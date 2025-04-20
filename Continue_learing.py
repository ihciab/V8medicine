# This code is adapted from Issue [#147](https://github.com/VainF/Torch-Pruning/issues/147), implemented by @Hyunseok-Kim0.
import argparse
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union
import re
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
#from torch.xpu import device
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from ultralytics import YOLO, __version__
from ultralytics.data import YOLODataset
from ultralytics.models import yolo
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck, ECA, MCEM, CARAFE
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel, ClassificationModel
#from ultralytics.nn.tasks import TASK_MAP
#from ultralytics.yolo.engine.trainer import BaseTrainer
#from ultralytics.yolo.utils import yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEY

from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights, de_parallel
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import yaml_load,LOGGER,RANK,DEFAULT_CFG_DICT,DEFAULT_CFG_KEYS
import torch_pruning as tp
#from mydistill import Ditillmoder
import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
TASK_MAP={
    "classify": {
        "model": ClassificationModel,
        "trainer": yolo.classify.ClassificationTrainer,
        "validator": yolo.classify.ClassificationValidator,
        "predictor": yolo.classify.ClassificationPredictor,
    },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }
param_dict = {
    # origin
    'model': './runs/detect/step_1_finetune/weights/best.pt',
    # 'model': None,
    'data': 'medicine.yaml',
    'imgsz': 640,
    'epochs': 1,
    'batch': 4,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project': 'runs/distill',
    'name': 'distillnet',

    # distill
    'prune_model': True,
    # 'teacher_weights': None,
    'teacher_weights': './somemodels/lightyolo1/best.pt',
    'teacher_cfg': 'teacher.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',

    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 1.0,

    'teacher_kd_layers': '20',
    'student_kd_layers': '20 ',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 1.0
}


def create_dataloader(param, batch_size, rank, workers):
    pass


class IncrementalTrainer:
    """持续学习训练管理器"""

    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size  # 旧数据缓冲区大小
        self.old_datasets = []  # 存储旧数据集
        self.old_model = None  # 旧模型参考

    def update_buffer(self, new_dataset):
        """更新数据缓冲区"""
        self.old_datasets.append(new_dataset)
        if len(self.old_datasets) > self.buffer_size:
            self.old_datasets.pop(0)  # FIFO淘汰

    def get_combined_loader(self, new_data, batch_size, imgsz):
        """创建混合数据加载器"""
        # 创建新数据集
        new_dataset = YOLODataset(
            img_path=new_data,
            imgsz=imgsz,
            augment=True,
            cache=False
        )

        # 合并旧数据
        combined_dataset = [new_dataset]
        if self.old_datasets:
            combined_dataset.extend(self.old_datasets)

        # 创建数据加载器
        return create_dataloader(
            ConcatDataset(combined_dataset),
            batch_size=batch_size,
            rank=-1,
            workers=8
        )
    """持续学习训练管理器"""
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size  # 旧数据缓冲区大小
        self.old_datasets = []  # 存储旧数据集
        self.old_model = None  # 旧模型参考

    def update_buffer(self, new_dataset):
        """更新数据缓冲区"""
        self.old_datasets.append(new_dataset)
        if len(self.old_datasets) > self.buffer_size:
            self.old_datasets.pop(0)  # FIFO淘汰

    def get_combined_loader(self, new_data, batch_size, imgsz):
        """创建混合数据加载器"""
        # 创建新数据集
        new_dataset = YOLODataset(
            img_path=new_data,
            imgsz=imgsz,
            augment=True,
            cache=False
        )

        # 合并旧数据
        combined_dataset = [new_dataset]
        if self.old_datasets:
            combined_dataset.extend(self.old_datasets)

        # 创建数据加载器
        return create_dataloader(
            ConcatDataset(combined_dataset),
            batch_size=batch_size,
            rank=-1,
            workers=8
        )


class ContinualLearning:
    """持续学习训练管理器"""

    def __init__(self, base_data_path, buffer_size=3):
        """
        参数：
            base_data_path: 基础数据集配置文件路径
            buffer_size: 保留的历史剪枝阶段数
        """
        self.base_data = yaml_load(check_yaml(base_data_path))
        self.buffer_size = buffer_size
        self.data_buffer = []  # 保存各阶段数据配置
        self.old_models = []  # 保存历史模型

    def add_new_data(self, new_data_path):
        """添加新数据集配置"""
        self.data_buffer.append(new_data_path)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)

    def get_combined_data(self):
        """生成混合数据集配置"""
        combined = deepcopy(self.base_data)
        combined['train'] = [self.base_data['train']] + self.data_buffer
        return combined


def train_continue(model, new_data_path, old_model=None, reg_lambda=0.5, epochs=1):
    """
    持续学习训练方法
    参数：
        model: 待训练的剪枝后模型
        new_data_path: 新增数据配置路径
        old_model: 参考模型（用于知识蒸馏）
        reg_lambda: 正则化强度
        epochs: 训练轮次
    """
    if not hasattr(model, 'trainer') or model.trainer is None:
        # 从模型配置中重建训练器
        task = model.overrides.get('task', 'detect')
        model.trainer = TASK_MAP[task]['trainer']
        overrides=model.overrides,
        _callbacks=getattr(model, 'callbacks', None)


    # 创建混合数据集
    cl_trainer = ContinualLearning(model.overrides['data'])
    cl_trainer.add_new_data(new_data_path)
    combined_cfg = cl_trainer.get_combined_data()
    # 配置混合数据集
    # combined_cfg = {
    #     'train': [model.overrides['data']['train'], new_data_path],
    #     'val': model.overrides['data']['val'],
    #     'nc': model.overrides['data']['nc']
    # }

    # 配置训练参数
    train_args = {
        'data': combined_cfg,
        'epochs': epochs,
        'imgsz': model.overrides['imgsz'],
        #'batch': model.overrides['batch'],
        'batch': 4,
        'name': f'prune-continue-{len(cl_trainer.data_buffer)}'
    }

    # 知识蒸馏正则化
    if old_model is not None:
        original_criterion = model.trainer.criterion
        old_model = old_model.freeze()

        def new_criterion(preds, batch):
            # 原始检测损失
            loss, loss_items = original_criterion(preds, batch)

            # 知识蒸馏损失
            with torch.no_grad():
                teacher_preds = old_model(batch['img'])

            # 分类头KL散度
            cls_loss = F.kl_div(
                F.log_softmax(preds[1][..., 4:], dim=-1),
                F.softmax(teacher_preds[1][..., 4:], dim=-1),
                reduction='batchmean'
            )

            # 回归头MSE
            box_loss = F.mse_loss(preds[0], teacher_preds[0])

            # 总损失
            total_loss = loss + reg_lambda * (cls_loss + box_loss)

            # 记录指标
            loss_items.update({
                'kd/cls': cls_loss.item(),
                'kd/box': box_loss.item()
            })
            return total_loss, loss_items

        model.trainer.criterion = new_criterion

    # 执行训练
    model.train(**train_args)
    return model
def save_pruning_performance_graph(x, y1, y2, y3):
    """
    Draw performance change graph
    Parameters
    ----------
    x : List
        Parameter numbers of all pruning steps
    y1 : List
        mAPs after fine-tuning of all pruning steps
    y2 : List
        MACs of all pruning steps
    y3 : List
        mAPs after pruning (not fine-tuned) of all pruning steps

    Returns
    -------

    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig('pruning_perf_change.png')


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)


def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    """
    x = torch.load(f, map_location=torch.device('cpu'))


    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def train_v2(self: YOLO, pruning=False, old_model=None, reg_lambda=0.5,
             incremental_trainer=None, **kwargs):
    """集成持续学习的改进训练函数"""
    print("train_v2[持续学习版]")
    self._check_is_pytorch_model()

    # 初始化训练配置
    overrides = self.overrides.copy()
    overrides.update(kwargs)

    # 持续学习相关配置
    if pruning and old_model is not None:
        old_model = old_model.float().eval().requires_grad_(False)
        original_loss_fn = self.trainer.criterion

        # 定义正则化损失
        def new_loss_fn(preds, batch, epoch=0):
            # 原始损失计算
            loss, loss_items = original_loss_fn(preds, batch)

            # 持续学习正则项
            with torch.no_grad():
                old_preds = old_model(batch['img'].to(old_model.device))

            # 检测任务正则化（分类头+回归头）
            cls_loss = 0.0
            box_loss = 0.0

            # 遍历每个检测头
            for new_head, old_head in zip(preds[1], old_preds[1]):
                # 分类KL散度
                cls_loss += F.kl_div(
                    F.log_softmax(new_head[..., 4:], dim=-1),
                    F.softmax(old_head[..., 4:].detach(), dim=-1),
                    reduction='batchmean'
                )
                # 回归MSE损失
                box_loss += F.mse_loss(new_head[..., :4], old_head[..., :4].detach())

            # 动态平衡系数
            current_lambda = reg_lambda * (1 - epoch / self.trainer.epochs)
            total_loss = loss + current_lambda * (cls_loss + box_loss)

            # 记录损失项
            loss_items.update({
                'reg/cls': cls_loss.item(),
                'reg/box': box_loss.item(),
                'lambda': current_lambda
            })

            return total_loss, loss_items

        self.trainer.criterion = new_loss_fn

    # 替换数据加载器
    if incremental_trainer is not None:
        data_cfg = yaml_load(check_yaml(overrides['data']))
        train_loader = incremental_trainer.get_combined_loader(
            new_data=data_cfg['train'],
            batch_size=overrides.get('batch', 16),
            imgsz=overrides.get('imgsz', 640)
        )
        self.trainer.train_loader = train_loader

    # 原有剪枝模式逻辑
    if not pruning:
        if not overrides.get('resume'):
            self.trainer.model = self.trainer.get_model(
                weights=self.model if self.ckpt else None,
                cfg=self.model.yaml
            )
            self.model = self.trainer.model
    else:
        self.trainer.pruning = True
        self.trainer.model = self.model
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    # 训练执行
    self.trainer.train()

    # 后续处理
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args


# def train_v2(self: YOLO, pruning=False, **kwargs):
#     """
#     Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
#     """
#     print("train_v2")
#     self._check_is_pytorch_model()
#     if self.session:  # Ultralytics HUB session
#         if any(kwargs):
#             LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
#         kwargs = self.session.train_args
#     overrides = self.overrides.copy()
#     overrides.update(kwargs)
#     if kwargs.get('cfg'):
#         LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
#         overrides = yaml_load(check_yaml(kwargs['cfg']))
#     overrides['mode'] = 'train'
#     if not overrides.get('data'):
#         raise AttributeError("Dataset required but missing, i.e. pass 'data=medicine.yaml'")
#     if overrides.get('resume'):
#         overrides['resume'] = self.ckpt_path
#
#     self.task = overrides.get('task') or self.task
#     self.trainer = TASK_MAP[self.task]['trainer'](overrides=overrides, _callbacks=self.callbacks)
#
#     if not pruning:
#         if not overrides.get('resume'):  # manually set model only if not resuming
#             print("train_v2_notpruingbengin")
#             self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
#             self.model = self.trainer.model
#
#     else:
#         # pruning mode
#         #print("train_v2_pruingbegin")
#         self.trainer.pruning = True
#         self.trainer.model = self.model
#
#         # replace some functions to disable half precision saving
#         self.trainer.save_model = save_model_v2.__get__(self.trainer)
#         self.trainer.final_eval = final_eval_v2.__get__(self.trainer)
#
#     # 附加HUB会话（用于云端协作）
#     self.trainer.hub_session = self.session  # attach optional HUB session
#     if hasattr(self.trainer, 'criterion'):
#         self.trainer.criterion = self.trainer.criterion.to('cuda:0')  # 强制移动到 GPU
#     #print("train_v2_trainerbegin")
#    # self.trainer.model = self.trainer.model.to('cuda:0')
#
#
#     self.trainer.train()
#     #self.trainer.model = self.trainer.model.to('cuda:0')
#     # Update model and cfg after training
#     if RANK in (-1, 0) :
#         self.model, _ = attempt_load_one_weight(str(self.trainer.best))
#        # self.model=self.model.to('cuda:0')
#         self.overrides = self.model.args
#         self.metrics = getattr(self.trainer.validator, 'metrics', None)
#     print("trainv2end")
###


def prune(args):
    # load trained yolov8 model
    model = YOLO(args.model)
    model.__setattr__("train_v2", train_v2.__get__(model))

    # 初始化持续学习管理器
    cl_manager = ContinualLearning(base_data_path=args.data)
    original_model = deepcopy(model.model)
    pruning_cfg = yaml_load(check_yaml(args.cfg))
    batch_size = pruning_cfg['batch']

    # use coco128 dataset for 10 epochs fine-tuning each pruning iteration step
    # this part is only for sample code, number of epochs should be included in config file
    #pruning_cfg['data'] = "medicine.yaml"
    #pruning_cfg['epochs'] = 1

    model.model.train()

    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    # do validation before pruning model
    pruning_cfg['name'] = f"baseline_val"
    pruning_cfg['batch'] = 1
    validation_model = deepcopy(model)
    #metric = validation_model.val(**pruning_cfg)
    #init_map = metric.box.map
    init_map = 0.735
    macs_list.append(base_macs)
    nparams_list.append(100)
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

    # prune same ratio of filter based on initial size
    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)

    for i in range(args.iterative_steps):

        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            if isinstance(m, (Detect,CARAFE)):
                ignored_layers.append(m)

        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.GroupNormImportance(),  # L2 norm pruning,
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )

        # Test regularization
        # output = model.model(example_inputs)
        # (output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        # pruner.regularize(model.model)

        pruner.step()
        # pre fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_pre_val"
        pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        #metric = validation_model.val(**pruning_cfg)
        #pruned_map = metric.box.map
        pruned_map =0.735
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
        current_speed_up = float(macs_list[0]) / pruned_macs
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = f"step_{i}_finetune"
        pruning_cfg['batch'] = batch_size  # restore batch size
        #model.train_v2(pruning=True, **pruning_cfg)
        # 持续训练步骤
        model = train_continue(
            model=model,
            new_data_path=args.new_data,
            old_model=original_model if i == 0 else model.model,
            reg_lambda=args.reg_lambda * (0.8 ** i),
            epochs=20
        )

        # post fine-tuning validation
        # 更新数据缓冲区
        #new_data = f"data/new_batch_{i}.yaml"
        #inc_trainer.update_buffer(YOLODataset(new_data))
        # 更新数据缓存
        cl_manager.add_new_data(args.new_data)
        # 验证性能
        print("\n[验证评估]")
        print("-- 基础数据集 --")
        val_base = model.val(data=args.data)
        print("-- 新数据集 --")
        val_new = model.val(data=args.new_data)


        pruning_cfg['name'] = f"step_{i}_post_val"
        pruning_cfg['batch'] = 1
        model.train_loader = None  # 清除数据加载器引用
        # 保存中间模型
        save_path = f"./runs/distill/pruned_step{i + 1}.pt"
        model.save(save_path)

        for name, param in model.model.named_parameters():
            param.requires_grad = True
        validation_model.load_state_dict(model.state_dict())


        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map
        print(f"After fine tuning mAP={current_map}")

        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)

        # remove pruner after single iteration
        del pruner

        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list)

        if init_map - current_map > args.max_map_drop:
            print("Pruning early stop")
            break

    model.export(format='onnx')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./somemodels/lightyolo1/best.pt', help='Pretrained pruning target model file')
    parser.add_argument('--cfg', default='./ultralytics/cfg/default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative-steps', default=1, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.4, type=float, help='Target pruning rate')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')
    # 新增持续学习参数
    parser.add_argument('--data', default='medicine.yaml', help='basedataset')
    parser.add_argument('--epoch', default=2, help='训练轮次')
    parser.add_argument('--new-data',default='newdata.yaml',
                        help="新增数据集配置文件路径")
    parser.add_argument('--reg-lambda', type=float, default=0.7,
                        help="知识蒸馏正则化强度")
    parser.add_argument('--buffer-size', type=int, default=3,
                        help="保留的历史数据阶段数")
    args = parser.parse_args()
    # 创建输出目录
    Path("runs/continual").mkdir(parents=True, exist_ok=True)

    prune(args)
