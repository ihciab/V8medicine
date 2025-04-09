import torch
import torch.nn as nn

# 参考文献：https://arxiv.org/pdf/2010.03045

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,  # 输入通道数
        out_planes,  # 输出通道数
        kernel_size,  # 卷积核大小
        stride=1,  # 步长
        padding=0,  # 填充
        dilation=1,  # 膨胀系数
        groups=1,  # 组数，用于分组卷积
        relu=True,  # 是否使用ReLU激活函数
        bn=True,  # 是否使用Batch Normalization
        bias=False,  # 是否使用偏置
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # 卷积层定义
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # 定义BN层（可选）
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        # 定义ReLU激活函数（可选）
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    """用于通道池化，生成两个特征图：最大池化图和平均池化图。"""
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    """生成空间注意力的门控机制，基于输入特征的空间分布生成注意力图。"""
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7  # 卷积核大小
        # 使用ChannelPool压缩通道维度后，使用BasicConv层生成空间注意力图
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)  # 通道池化
        x_out = self.spatial(x_compress)  # 生成注意力图
        scale = torch.sigmoid_(x_out)  # 使用sigmoid激活以限制范围在0到1
        return x * scale  # 将输入乘以注意力权重


class TripletAttention(nn.Module):
    """三重注意力模块，通过通道方向和空间方向对特征图生成注意力。"""
    def __init__(
        self,
        no_spatial=False,  # 是否禁用空间注意力
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()  # 水平方向注意力
        self.ChannelGateW = SpatialGate()  # 垂直方向注意力
        self.no_spatial = no_spatial  # 控制是否使用空间注意力
        if not no_spatial:
            self.SpatialGate = SpatialGate()  # 空间注意力

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()  # 将通道和宽度维度互换
        x_out1 = self.ChannelGateH(x_perm1)  # 计算水平方向注意力
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()  # 恢复原始维度顺序

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()  # 将通道和高度维度互换
        x_out2 = self.ChannelGateW(x_perm2)  # 计算垂直方向注意力
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()  # 恢复原始维度顺序

        if not self.no_spatial:
            x_out = self.SpatialGate(x)  # 计算空间注意力
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)  # 三种注意力的平均加权
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)  # 两种注意力的平均加权
        return x_out  # 输出加权后的特征图



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck_TripletAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = TripletAttention()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_TripletAttention(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_TripletAttention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ =='__main__':

    TripletAttention = TripletAttention()
    #创建一个输入张量
    batch_size = 1
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =TripletAttention(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)