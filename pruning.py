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

from ultralytics import YOLO, __version__
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


def train_v2(self: YOLO, pruning=False, **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
    """
    print("train_v2")
    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = TASK_MAP[self.task]['trainer'](overrides=overrides, _callbacks=self.callbacks)

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            print("train_v2_notpruingbengin")
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

    else:
        # pruning mode
        #print("train_v2_pruingbegin")
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    if hasattr(self.trainer, 'criterion'):
        self.trainer.criterion = self.trainer.criterion.to('cuda:0')  # 强制移动到 GPU
    #print("train_v2_trainerbegin")
   # self.trainer.model = self.trainer.model.to('cuda:0')


    self.trainer.train()
    #self.trainer.model = self.trainer.model.to('cuda:0')
    # Update model and cfg after training
    if RANK in (-1, 0) :
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
       # self.model=self.model.to('cuda:0')
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)
    print("trainv2end")

def Prune_map_paramas_loss(init_map,init_params,nowmap,nowparams):  #作为剪枝的任务 还是尽量剪去参数
    maploss=float(((init_map-nowmap)/init_map))  #描述精度损失 越小越好
    init_paramsloss=float(((init_params-nowparams)/init_params))  #描述参数损失 越大越好 后续两者参数还可动态调整

    return maploss*500+init_paramsloss*500
def layer_adjust(layer_prune_idx:[[]]):
    result = []
    i = 0
   # result.append([])
    while i < len(layer_prune_idx):
        current = layer_prune_idx[i]
        if current is None:
            i=i+1
        elif current is not None:
            if len(current) <= 3:
                if i + 1 < len(layer_prune_idx):
                    merged = current + layer_prune_idx[i + 1]
                    result.append(merged)
                    i += 2
                else:
                    result.append(current)
                    i += 1
            else:
                result.append(current)
                i += 1
    return result
def get_layer_id(group):
    """
    从剪枝组中提取唯一的网络层编号

    参数：
        group (list): 包含依赖项(dep)的剪枝组，每个dep应包含target.name属性

    返回：
        List[int]: 唯一且排序后的层编号列表
    """
    layers_to_prune = []
    for dep in group:
        # 安全获取层名称
        modulename=dep.dep.target.name

        # 精确匹配 model.数字. 的格式（例如匹配 model.10. 中的10）

        parts = modulename.split('.')
        try:
            # 验证结构：至少包含model和数字段，且数字段紧随model之后
            if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
                layer_id=int(parts[1])
                if(layer_id in layers_to_prune):
                    continue
                else:
                    layers_to_prune.append(layer_id)
        except (IndexError, ValueError, AttributeError):
            pass
    return layers_to_prune



def prune(args):
    # load trained yolov8 model
    print("prune_begin")
    model = YOLO(args.model)
    #model=model.to('cuda:0')
    model.__setattr__("train_v2", train_v2.__get__(model))
    pruning_cfg = yaml_load(check_yaml(args.cfg))
    batch_size = pruning_cfg['batch']

    # use coco128 dataset for 10 epochs fine-tuning each pruning iteration step
    # this part is only for sample code, number of epochs should be included in config file
    pruning_cfg['data'] = "medicine.yaml"
    pruning_cfg['epochs'] = 1
    pruning_cfg['workers'] = 0

    model.model.train()
    #print(model)
    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    for name, param in model.model.named_parameters():
        param.requires_grad = True
   # print("prune_inpus")
    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)
    print(model.device)
        #改用metric.box.ap50
    # do validation before pruning model
    pruning_cfg['name'] = f"baseline_val"
    pruning_cfg['batch'] = 1
    validation_model = deepcopy(model)
    # metric = validation_model.val(**pruning_cfg)
    # init_map = metric.box.map50
    macs_list.append(base_macs)
    nparams_list.append(100)
    init_map=0.735
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    # print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

    # prune same ratio of filter based on initial size
    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)


    #######init pruner

    ignored_layers = []
    unwrapped_parameters = []



    ## 这里将所有操作在tempmodel上做
    ####获取可能减枝的层级？？
    for i in range(args.iterative_steps):

        bestmodel = deepcopy(model.model)
        layer_prune_idx=[]
        ##先将自定义模块全部忽略
        validation_model.model = deepcopy(model.model).to(model.device)
        pruning_cfg['name'] = None
        pruning_cfg['save'] = False
        pruning_cfg['save_json'] = False
        pruning_cfg['save_txt'] = False
        pruning_cfg['save_dir'] = None
        pruning_cfg['batch'] = 1

        metric = validation_model.val(**pruning_cfg)       # 禁用保存目录的自动创建
        before_map = metric.box.map50
        nowmac, before_nparams= tp.utils.count_ops_and_params(model.model, example_inputs)




        print("prune_forpruner")


        # Test regularization
        # output = model.model(example_inputs)
        # (output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        # pruner.regularize(model.model)

        #pruner.step(interactive=True)
        rep_prun_times=4
        rep_steps=50
        best_to_still = []
        layers_to_prune = []

        bestloss = 0
        for find_prun_times in range(rep_prun_times):

            print(f"Now in  {find_prun_times}:Times")
            tringi = 0  ##############要通过ixd修改！！！
            print(layers_to_prune)
            print("----------")
            best_to_still = list(set(best_to_still + layers_to_prune))
            print(best_to_still)
            rep_count=0
            while(rep_count<rep_steps):
                if(rep_count>=rep_steps):
                    break
                ###更新减枝器的model
                if(rep_count==0):


                    prunemodel = deepcopy(model.model)
                    prunemodel.train()

                    for name, param in prunemodel.named_parameters():
                        param.requires_grad = True

                else:
                    prunemodel = deepcopy(model.model)
                    prunemodel.train()

                    for name, param in prunemodel.named_parameters():
                        param.requires_grad = True

                for m in prunemodel.model.modules():
                    if isinstance(m, (Detect, CARAFE)):
                        ignored_layers.append(m)
                pruner = tp.pruner.GroupNormPruner(
                    prunemodel,
                    # model.model.to(model.device),
                    example_inputs,
                    # example_inputs.to(model.device),
                    importance=tp.importance.GroupNormImportance(),  # L2 norm pruning,

                    iterative_steps=8,
                    pruning_ratio=pruning_ratio,
                    ignored_layers=ignored_layers,
                    unwrapped_parameters=unwrapped_parameters
                )
                ###更新减枝器的model

                if (rep_count == 0):
                    for group in pruner.step(interactive=True):
                        dep, idxs = group[0]
                        if(idxs is not None):
                            layer_prune_idx.append(idxs)

                    ###---------- 调整idx 是大于三个为一组 减少循环次数
                    #layer_prune_idx = layer_adjust(layer_prune_idx)
                    #layer_prune_idx = layer_adjust(layer_prune_idx)
                    print(layer_prune_idx)
                        ####----------

                rep_steps=len(layer_prune_idx)
                tringi = tringi + 1
                ####  一些参数
                Is_large_layers=0
                for try_prune_i,group in enumerate(pruner.step(interactive=True)):  # Warning: groups must be handled sequentially. Do not keep them as a list.

                    #pruning_fn = dep.handler  # get the pruning function
                    #获得一下剪枝的层？  在group中获取最好的一步吗？

                    ###这个的作用是让group的迭代次数匹配idx的次数

                    if(try_prune_i<rep_count):
                        continue
                    else:
                        Is_large_layers+=len(layer_prune_idx[try_prune_i])
                        if(Is_large_layers<5):
                            group.prune(layer_prune_idx[try_prune_i])
                            layers_to_prune = layers_to_prune+get_layer_id(group)
                            rep_count=rep_count+1
                            continue
                        else:
                            if(layer_prune_idx[rep_count] is not None):
                                Is_large_layers=0
                                group.prune(layer_prune_idx[rep_count])

                                nowmac,nownparams  = tp.utils.count_ops_and_params(pruner.model, example_inputs)
                                print(
                                    f"Searing for  pruning group iter {rep_count}: MACs={nowmac / 1e9} G, #Params={nownparams / 1e6} M")
                                validation_model.model = deepcopy(pruner.model)

                                pruning_cfg['name'] = f"step_{rep_count}_{try_prune_i}_tring_val"
                                pruning_cfg['batch'] = 1

                                #metric = validation_model.val(**pruning_cfg)
                                #current_map = metric.box.map50

                                ####
                                current_map=before_map


                                ####


                                nowloss=Prune_map_paramas_loss(before_map,before_nparams /1e6 ,current_map,nownparams/1e6)

                                if(rep_count==0):
                                    layers_to_prune=get_layer_id(group)
                                    bestloss=nowloss
                                    bestmodel = deepcopy(pruner.model)
                                    break

                                else:
                                    if(nowloss>=bestloss):  ###更新最佳模型      思路：这里确定一次迭代要到什么程度 只剪枝一次吗？ 剪枝一次要distll训练吗？ 找到之后就得退出了？？
                                        print("find a better prune ")
                                        layers_to_prune=get_layer_id(group)
                                        bestloss=nowloss
                                        bestmodel=deepcopy(pruner.model)
                                        break
                                        #将模型还原为没有剪枝的状态

                                        #group.prune()
                                        ####保留此时的剪枝层数和剪枝模型
                                    if(nowloss<bestloss):
                                        print("nowloss<bestloss Something need to do")
                                        break
                                ##pruner.model 是剪枝器中的model 两者一样 要修改

                                #nowloss=Prune_map_paramas_loss(init_map,base_nparams /1e6 ,current_map,nownparams/1e6)


                    ###
                rep_count=rep_count+1
                # 合并需要的layers  更新剪枝模型为新的模型 同时还要更新最外层的model.model
                ##有点小问题


                del pruner
                #print(best_to_prune)



            model.model = deepcopy(bestmodel)

                #####需要的参数  1 currentmap 2.initmap  2.nowparametersize  3.initsize
                #print(f"After pruning iter {i + 1}: MACs={macs / 1e9} G, #Params={nparams/ 1e6} M, ")
                # group.prune(idxs=[0, 2, 6]) # It is even possible to change the pruning behaviour with the idxs parameter
        # pre fine-tuning validation
        print("Finish A search to a good model")
        #model.model=deepcopy(bestmodel)
        pruning_cfg['name'] = f"step_{i}_pre_val"
        pruning_cfg['batch'] = 1
        pruning_cfg['save'] = True
        pruning_cfg['save_json'] = True
        pruning_cfg['save_txt'] = True
        validation_model.model = deepcopy(model.model).to(model.device)

        metric = validation_model.val(**pruning_cfg)
        pruned_map = metric.box.map
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model.model,example_inputs)
        current_speed_up = float(macs_list[0]) / pruned_macs
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = f"step_{i}_finetune"
        pruning_cfg['batch'] = batch_size  # restore batch size
        model.train_v2(pruning=True, **pruning_cfg)

        # post fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_post_val"
        pruning_cfg['batch'] = 1
        validation_model = YOLO(model.trainer.best).to(model.device)
        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map50
        print(f"After fine tuning mAP={current_map}")

        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)

        # remove pruner after single iteration


        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list)
        print("next epoch")
        if init_map - current_map > args.max_map_drop:
            print("Pruning early stop")
            break

    model.export(format='engine')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./somemodels/lightyolo1/best.pt', help='Pretrained pruning target model file')
    parser.add_argument('--cfg', default='./ultralytics/cfg/cuttest.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative-steps', default=5, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.4, type=float, help='Target pruning rate')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')

    args = parser.parse_args()

    prune(args)
