import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, Detect, DFL
from torchvision.ops import deform_conv2d

from ...utils.tal import make_anchors, dist2bbox


class DeformableConv2d(nn.Module):
    """可变形卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding

        # 偏移量生成卷积层
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 每个位置x,y方向的偏移
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 权重调制生成卷积层
        self.modulator_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,  # 每个采样点的调制系数
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 常规卷积权重
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # 初始化参数
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.modulator_conv.weight)
        nn.init.constant_(self.modulator_conv.bias, 0.5)

    def forward(self, x):
        # 生成偏移量
        offset = self.offset_conv(x)
        # 生成调制系数（sigmoid限制在0-1之间）
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        # 应用可变形卷积
        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=modulator
        )
        return x


# ---------------------- DyDetect动态检测头 ----------------------

class DyDCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DyDCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()



class DyDetect(Detect):
    """动态感受野检测头（兼容YOLOv8框架）"""

    def __init__(self, nc=80, ch=()):
        super().__init__(nc,ch)
        self.nc = nc  # 类别数
        # 延迟初始化（根据输入特征图自动获取通道数）
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(DyDCNv2(x, c2, 3), DyDCNv2(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(DyDCNv2(x, c3, 3), DyDCNv2(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DyDCNv2(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DyDCNv2(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

# ---------------------- 带对比学习的检测头 ----------------------
class DyDetectWithCL(DyDetect):
    """带对比学习的动态检测头"""

    def __init__(self, nc=1,  ch=(), temp=0.1):
        super().__init__(nc,  ch)
        self.temp = temp
        # 为每个尺度添加投影层
        self.proj = nn.ModuleList([
            nn.Conv2d(c, 128, 1) for c in ch
        ])

    def _contrastive_loss(self, feats, targets):
        """多尺度对比损失"""
        total_loss = 0.0
        for feat, proj_layer in zip(feats, self.proj):
            proj_feat = F.normalize(proj_layer(feat), dim=1)
            # 生成目标掩码（需根据实际标签处理）
            mask = self._build_target_mask(feat.shape[2:], targets)
            # 计算对比损失（简化示例）
            pos_feat = proj_feat[mask > 0]
            neg_feat = proj_feat[mask == 0]
            if len(pos_feat) == 0:
                continue
            pos_sim = torch.mm(pos_feat, pos_feat.t()) / self.temp
            neg_sim = torch.mm(pos_feat, neg_feat.t()) / self.temp
            loss = -torch.log(torch.exp(pos_sim).sum() / (torch.exp(pos_sim).sum() + torch.exp(neg_sim).sum()))
            total_loss += loss * 0.1  # 损失权重
        return total_loss

    def forward(self, x, targets=None):
        # 检测输出
        p = super().forward(x)  # 返回多尺度检测结果

        if self.training:
            # 对比学习分支
            cl_loss = self._contrastive_loss(x, targets)
            return p, {'cl_loss': cl_loss}
        return p