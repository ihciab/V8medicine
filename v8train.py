from ultralytics import YOLO
import os
import torch


if __name__ == '__main__':
    #model = YOLO("./yolov8n.pt", task="detect")


    # 在engine trainer文件夹中修改了resume代码 使得每次训练继承了之前的pt 每次训练前记得修改
    #model = YOLO("./myyolonet.yaml",verbose=True)
    model = YOLO("testlight.yaml", verbose=True)
    #model.train(data="medicine.yaml",resume=False,workers=0,epochs=15,batch=4)

    #model = YOLO('./somemodels/mydistillnet/best.pt')
    model.train(data="medicine.yaml", resume=False, workers=0, epochs=15, batch=4)
    print(model.info(detailed=True))
    #print(model.device)



    # 加载配置文件

    # 打印网络结构




