from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first
import torch
import torch.nn.functional as F
import sys, os, torch, math, time, warnings
import torch_pruning as tp
import matplotlib

matplotlib.use('AGG')


from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, clean_url, colorstr, emojis, yaml_save, callbacks, \
    __version__, LOCAL_RANK

from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, one_cycle, init_seeds, select_device

import math

class ContinualTrainer(BaseTrainer):
    """支持正则化持续学习的训练器，防止灾难性遗忘"""

    def __init__(self, model, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.pretrained_model = model  # 预训练模型（旧模型）
        #self.new_data_cfg = overrides.get('data', 'newdata.yaml')  # 新数据集配置
        self.device = select_device(self.args.device, self.args.batch)
        self.old_model = None  # 固定旧模型用于蒸馏
        self.ewc_lambda = 1.0  # EWC正则化强度
        self.kd_lambda = 0.5  # 蒸馏损失强度
        self._setup_train(world_size=1)

    def get_dataset(self):
        """加载新数据集（无需旧数据）"""
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")
    def setup_model(self):
        """初始化模型并冻结旧模型"""
        # 加载预训练模型
        self.old_model = attempt_load_weights(self.pretrained_model).eval().to(self.device)
        for param in self.old_model.parameters():
            param.requires_grad_(False)  # 固定旧模型参数

        # 初始化新模型（继承旧模型权重）
        self.model = attempt_load_weights(self.pretrained_model).to(self.device)
        self._adapt_model_for_new_data()  # 适配新数据

    def _adapt_model_for_new_data(self):
        """调整模型输出层（当类别数变化时）"""
        old_nc = self.old_model.nc
        new_nc = self.data['nc']
        if new_nc != old_nc:
            LOGGER.info(f"Adapting model from {old_nc} to {new_nc} classes")
            self.model = self.model._modify_output_layer(new_nc)

    def calculate_ewc_loss(self):
        """计算弹性权重巩固(EWC)正则化项"""
        ewc_loss = 0
        for (name, new_param), (_, old_param) in zip(self.model.named_parameters(),
                                                     self.old_model.named_parameters()):
            fisher = getattr(self.old_model, f'{name}_fisher', None)  # 需要预计算Fisher信息
            if fisher is not None:
                ewc_loss += (fisher * (new_param - old_param).pow(2)).sum()
        return self.ewc_lambda * ewc_loss

    def compute_distillation_loss(self, new_pred, old_pred):
        """计算知识蒸馏损失"""
        # 使用KL散度对齐分类logits
        cls_loss = F.kl_div(
            F.log_softmax(new_pred[..., 4:], dim=-1),
            F.softmax(old_pred[..., 4:], dim=-1),
            reduction='batchmean'
        )

        # 对齐边界框回归结果
        box_loss = F.mse_loss(new_pred[..., :4], old_pred[..., :4])
        return self.kd_lambda * (cls_loss + box_loss)

    def loss_function(self, pred, batch, new_loss_items):
        """组合损失函数"""
        # 原始检测损失
        original_loss = sum(new_loss_items)

        # 获取旧模型的预测
        with torch.no_grad():
            old_pred = self.old_model(batch['img'])

        # 蒸馏损失
        kd_loss = self.compute_distillation_loss(pred, old_pred)

        # EWC正则化
        ewc_loss = self.calculate_ewc_loss()

        # 总损失 = 新任务损失 + 蒸馏损失 + EWC正则化
        total_loss = original_loss + kd_loss + ewc_loss

        # 记录各项损失值
        loss_dict = {
            'loss/total': total_loss,
            'loss/original': original_loss,
            'loss/kd': kd_loss,
            'loss/ewc': ewc_loss
        }
        return total_loss, loss_dict

    def preprocess_batch(self, batch):
        """预处理批次数据（需返回旧模型预测）"""
        batch = super().preprocess_batch(batch)
        with torch.no_grad():
            batch['old_pred'] = self.old_model(batch['img'])  # 预计算旧模型输出
        return batch

    def training_step(self, batch, batch_idx):
        """重写训练步骤"""
        # 前向传播
        pred = self.model(batch['img'])

        # 计算原始检测损失
        _, loss_items = self.model.criterion(pred, batch)

        # 组合正则化损失
        total_loss, loss_dict = self.loss_function(pred, batch, loss_items)

        # 反向传播
        self.scaler.scale(total_loss).backward()
        return loss_dict

    def build_optimizer(self, model, **kwargs):
        """优化器配置（排除旧模型参数）"""
        # 只优化新模型参数
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.args.lr0 * 0.1)  # 使用更低学习率


# 使用示例 --------------------------------------------------
if __name__ == '__main__':
    overrides = {
        'data': 'newdata.yaml',
        'imgsz': 640,
        'epochs': 100,
        'batch': 4,
        'device': '0',
        'kd_lambda': 0.8,  # 可自定义正则化强度
        'ewc_lambda': 0.3
    }

    trainer = ContinualTrainer(
        model='./somemodels/lightyolo1/best.pt',  # 预训练模型路径
        overrides=overrides,
        cfg = 'D:/ultralytics-main/ultralytics/cfg/continuelearn.yaml'    )
    trainer.train()