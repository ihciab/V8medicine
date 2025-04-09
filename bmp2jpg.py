# -*- coding: utf-8 -*-

import os
from PIL import Image


def convert_bmp_to_jpg(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置最大图像像素数量
    Image.MAX_IMAGE_PIXELS = None  # 设置为 None 以解除限制

    # 遍历输入目录中的文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".bmp"):
            # 构建文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")

            # 打开 BMP 文件并保存为 JPG
            with Image.open(input_path) as img:
                img.convert("RGB").save(output_path, "JPEG")


if __name__ == "__main__":
    input_folder = "./zhongyaodata/zhongyaodataset/data_batch3/image"  # 替换为实际的输入文件夹路径
    output_folder = "./zhongyaodata/zhongyaodataset/data_batch3/image_jpg"  # 替换为实际的输出文件夹路径
    convert_bmp_to_jpg(input_folder, output_folder)