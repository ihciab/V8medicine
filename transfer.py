import os


def modify_yolo_classes(label_dir):
    # 遍历标签目录中的所有文件
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue  # 跳过非txt文件

        filepath = os.path.join(label_dir, filename)

        # 读取文件内容
        with open(filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            parts = line.split()
            if len(parts) != 5:
                print(f"警告：文件 {filename} 中存在格式错误的行: '{line}'，已跳过")
                continue

            # 替换类别索引为0（如果原索引不是0）
            class_idx = parts[0]
            if class_idx != '0':
                parts[0] = '0'

            new_line = ' '.join(parts) + '\n'
            new_lines.append(new_line)

        # 写回修改后的内容
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        print(f"已处理文件: {filename}")


# 使用示例（修改为你的标签目录路径）
label_directory = 'D:/ultralytics-main/zhongyaodata/zhongyaodataset/data_batch3/label_yolo'  # YOLO标签所在文件夹路径
modify_yolo_classes(label_directory)