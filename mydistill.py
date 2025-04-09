import warnings

from ultralytics import YOLO

warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
def Ditillmoder(feature_layers,distill_model,epochs=1, batch=4,losstype='feature'):
    param_dict = {
        # origin
        'model': None,
        'data': 'medicine.yaml',
        'imgsz': 640,
        'epochs':  epochs,
        'batch': batch,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project': 'runs/distill',
        'name': 'distillnet',

        # distill
        'prune_model': True,
        'teacher_weights': './somemodels/lightyolo1/best.pt',
        'teacher_cfg': 'teacher.yaml',
        'kd_loss_type': losstype,
        'kd_loss_decay': 'constant',

        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,

        'teacher_kd_layers': '20',
        'student_kd_layers': '20 ',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    print('------distilling now------')
    param_dict['teacher_kd_layers'] = feature_layers
    param_dict['student_kd_layers'] =feature_layers
    model = DetectionDistiller(overrides=param_dict)
    model.distill(distill_model)
    return 0
if __name__ == '__main__':
    param_dict = {
        # origin
        'model': './runs/detect/step_1_finetune/weights/best.pt',
        #'model': None,
        'data': 'medicine.yaml',
        'imgsz': 640,
        'epochs': 10,
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
        #'teacher_weights': None,
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

    model = DetectionDistiller(overrides=param_dict)
    print(model.model)
    Ymodel=YOLO('./runs/detect/step_1_finetune/weights/best.pt')
    #model.model = YOLO('./runs/detect/step_1_finetune/weights/best.pt').model
    #model.teacher_weights=YOLO('./runs/detect/step_1_finetune/weights/best.pt')

    # 问题出在setup prune函数中 要加载预训练的权重
    model.distill(Ymodel.model)