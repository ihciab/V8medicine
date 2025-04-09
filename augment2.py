import os
import cv2
import random
import numpy as np
import  shutil


from glob import glob
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

SHOW_SAVE_PATH = "./zhongyaodata/result2"
CLASSES = ['keli']
# 配置参数
class Config:
    # 路径配置
    looptimes=1
    img_dir = "./image"
    label_dir = "./label"
    output_img_dir ="./zhongyaodata/augimg2"
    output_label_dir = "./zhongyaodata/auglabel2"

    # 图像尺寸相关参数
    original_height = 2994
    original_width = 2174
    mosaic_crop_size = (1088, 1496)  # Mosaic子图尺寸（原图1/2）
    target_height = 1024  # 调整为适合模型输入的尺寸
    target_width = 768

    # 增强参数（根据大尺寸调整）
    mosaic_prob = 0.5  # 降低Mosaic概率
    copy_paste_prob = 0.3  # 降低复制概率
    min_area = 100  # 增大最小面积阈值
    min_visibility = 0.7  # 提高可见度阈值
# 创建输出目录
os.makedirs(Config.output_img_dir, exist_ok=True)
os.makedirs(Config.output_label_dir, exist_ok=True)


def validate_bbox(bbox, img_w, img_h):
    """校验边界框并修复尺寸问题"""
    x_min, y_min, x_max, y_max, class_id = bbox
    x_min = max(0, min(x_min, img_w-1))
    y_min = max(0, min(y_min, img_h-1))
    x_max = max(x_min+1, min(x_max, img_w))
    y_max = max(y_min+1, min(y_max, img_h))
    return [x_min, y_min, x_max, y_max, class_id]

def yolo_to_bbox(yolo_box, img_w, img_h):
    class_id, x_center, y_center, width, height = map(float, yolo_box)
    class_id = int(class_id)

    x_center = x_center * img_w
    y_center = y_center * img_h
    width = width * img_w
    height = height * img_h

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return validate_bbox([x_min, y_min, x_max, y_max, class_id], img_w, img_h)


def load_data(img_path):
    """加载图像和标注"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    label_path = os.path.join(Config.label_dir,
                              os.path.basename(img_path).replace('.jpg', '.txt'))
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                yolo_box = line.strip().split()
                if len(yolo_box) != 5:
                    continue
                try:
                    bbox = yolo_to_bbox(yolo_box, w, h)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area >= Config.min_area:
                        bboxes.append(bbox)
                except Exception as e:
                    print(f"解析标注错误: {line}, {str(e)}")
                    continue

    return img, np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)


# 针对大尺寸图像的增强管道
large_image_transform = A.Compose([
    # 轻度几何变换
    #A.Rotate(limit=5, interpolation=cv2.INTER_NEAREST, p=0.3),  # 减小旋转角度
    A.RandomScale(scale_limit=0.05, p=0.2),  # 减小缩放比例

    A.RandomCrop(
        height=int(Config.original_height * 0.8),
        width=int(Config.original_width * 0.8),
        p=0.5
    ),

    # 光学增强（降低强度）
    A.RandomGamma(gamma_limit=(90, 110), p=0.2),


    #A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
    #A.GlassBlur(sigma=0.7, max_delta=2, iterations=2, p=0.3),
    # 遮挡增强（降低强度）
    A.CoarseDropout (
        fill_value=0,
        p=0.1
    ),

    A.HorizontalFlip(p=0.2),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['class_labels'],
    min_area=Config.min_area,
    min_visibility=Config.min_visibility
))


def safe_micro_mosaic(images, bboxes_list):
    """修复尺寸问题的Mosaic增强"""
    # 统一调整子图尺寸
    target_h, target_w = Config.mosaic_crop_size
    output_img = np.zeros((target_h * 2, target_w * 2, 3), dtype=np.uint8)
    output_boxes = []

    indices = random.sample(range(len(images)), k=4)

    for i, idx in enumerate(indices):
        # 调整子图尺寸
        img = cv2.resize(images[idx], (target_w, target_h))
        boxes = bboxes_list[idx].copy()

        # 调整边界框坐标
        scale_x = target_w / images[idx].shape[1]
        scale_y = target_h / images[idx].shape[0]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # 确定拼接位置
        x_start = (i % 2) * target_w
        y_start = (i // 2) * target_h
        output_img[y_start:y_start + target_h, x_start:x_start + target_w] = img

        # 调整全局坐标
        boxes[:, [0, 2]] += x_start
        boxes[:, [1, 3]] += y_start

        # 校验边界框
        for box in boxes:
            validated = validate_bbox(box, output_img.shape[1], output_img.shape[0])
            output_boxes.append(validated)

    return output_img, np.array(output_boxes, dtype=np.float32)

def visualize_bbox(image, bboxes):
    img = image.copy()
    for box in bboxes:
        x1, y1, x2, y2, cls = map(int, box)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
    plt.imshow(img)
    plt.show()
def draw_detections(box, name, img):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))

    # 根据图像大小调整矩形框的线宽和文本的大小
    line_thickness = max(1, int(min(height, width) / 200))
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(min(height, width) / 200))
    # 根据图像大小调整文本的纵向位置
    text_offset_y = int(min(height, width) / 50)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), line_thickness)
    cv2.putText(img, str(name), (xmin, ymin - text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                font_thickness, lineType=cv2.LINE_AA)
    return img

def show_labels(images_base_path, labels_base_path):
    if os.path.exists(SHOW_SAVE_PATH):
        shutil.rmtree(SHOW_SAVE_PATH)
    os.makedirs(SHOW_SAVE_PATH, exist_ok=True)

    for images_name in tqdm(os.listdir(images_base_path)):
        file_heads, _ = os.path.splitext(images_name)
        # images_path = f'{images_base_path}/{images_name}'
        images_path = os.path.join(images_base_path, images_name)
        # labels_path = f'{labels_base_path}/{file_heads}.txt'
        labels_path = os.path.join(labels_base_path, f'{file_heads}.txt')
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float64), f.readlines())),
                                  dtype=np.float64)
            images = cv2.imread(images_path)
            height, width, _ = images.shape
            for cls, x_center, y_center, w, h in labels:
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                draw_detections([x_center - w // 2, y_center - h // 2, x_center + w // 2, y_center + h // 2],
                                CLASSES[int(cls)], images)
            # cv2.imwrite(f'{SHOW_SAVE_PATH}/{images_name}', images)
            cv2.imwrite(os.path.join(SHOW_SAVE_PATH, images_name), images)
            print(f'{SHOW_SAVE_PATH}/{images_name} save success...')
        else:
            print(f'{labels_path} label file not found...')
def bbox_to_yolo(bbox, img_w, img_h):
    """将像素坐标转换为YOLO格式"""
    x_min, y_min, x_max, y_max, class_id = bbox
    width = x_max - x_min
    height = y_max - y_min

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    return [
        class_id,
        x_center / img_w,
        y_center / img_h,
        width / img_w,
        height / img_h
    ]
def process_single(img_path,loop_idx):
    try:
        base_name = os.path.basename(img_path).split('.')[0]

        for _ in range(Config.looptimes):
            img, bboxes = load_data(img_path)
            if len(bboxes) == 0:
                continue

            # 生成唯一后缀
            suffix = f"_L{loop_idx}_{random.randint(1000, 9999)}"
            output_img_path = os.path.join(Config.output_img_dir, f"{base_name}{suffix}.jpg")
            output_label_path = os.path.join(Config.output_label_dir, f"{base_name}{suffix}.txt")
        # 应用增强
        if random.random() < Config.mosaic_prob:
            all_img_paths = glob(os.path.join(Config.img_dir, "*.jpg"))
            if len(all_img_paths) >= 4:
                other_paths = random.sample(all_img_paths, 3)
                other_imgs = []
                other_boxes = []
                for path in other_paths:
                    img_, boxes_ = load_data(path)
                    other_imgs.append(img_)
                    other_boxes.append(boxes_)
                img, bboxes = safe_micro_mosaic([img] + other_imgs, [bboxes] + other_boxes)

        # 基础增强
        transformed = large_image_transform(
            image=img,
            bboxes=bboxes[:, :4].tolist(),
            class_labels=bboxes[:, 4].tolist()
        )

        # 后处理校验
        valid_bboxes = []
        for box, cls in zip(transformed['bboxes'], transformed['class_labels']):
            validated = validate_bbox([*box, cls], transformed['image'].shape[1], transformed['image'].shape[0])
            area = (validated[2] - validated[0]) * (validated[3] - validated[1])
            if area >= Config.min_area:
                valid_bboxes.append(validated)

        bboxes = np.array(valid_bboxes, dtype=np.float32)
        img = transformed['image']

        # 保存结果
        # 转换回YOLO格式
        h, w = img.shape[:2]
        yolo_boxes = []
        for box in bboxes:
            yolo_box = bbox_to_yolo(box, w, h)
            yolo_boxes.append(yolo_box)
        base_name = os.path.basename(img_path).split('.')[0]
        suffix = f"_{random.randint(1000, 9999)}"
        output_img_path = os.path.join(Config.output_img_dir, f"{base_name}{suffix}.jpg")
        output_label_path = os.path.join(Config.output_label_dir, f"{base_name}{suffix}.txt")

        cv2.imwrite(output_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(output_label_path, 'w') as f:
            for box in yolo_boxes:
                line = f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)
        #debug_bboxes(img, bboxes)
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
def debug_bboxes(img, bboxes):
    plt.imshow(img)
    for box in bboxes:
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()

if __name__ == '__main__':
    img_paths = glob(os.path.join(Config.img_dir, "*.jpg"))
    print(f"Found {len(img_paths)} images")

    # 增加循环次数控制
    for loop_idx in range(Config.looptimes):
        print(f"Processing loop {loop_idx + 1}/{Config.looptimes}")
        for path in tqdm(img_paths, desc=f"Loop {loop_idx + 1}"):
            process_single(path, loop_idx)
    show_labels(Config.output_img_dir, Config.output_label_dir)
    print(f"Augmentation completed. Total loops: {Config.looptimes}")
    print(f"Images: {Config.output_img_dir}")
    print(f"Labels: {Config.output_label_dir}")