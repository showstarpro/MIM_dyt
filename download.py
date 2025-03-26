import os
import shutil

# 原始数据集路径
tiny_imagenet_dir = "./data/tiny-imagenet-200/train/"
output_dir = "./data/mini_imagenet_1class10img/train/"

# 选择第一个类别
class_id = os.listdir(tiny_imagenet_dir)[0]  # 取第一个类别
os.makedirs(os.path.join(output_dir, class_id), exist_ok=True)

# 复制 10 张图片
image_files = os.listdir(os.path.join(tiny_imagenet_dir, class_id, "images"))[:10]
for img in image_files:
    src = os.path.join(tiny_imagenet_dir, class_id, "images", img)
    dst = os.path.join(output_dir, class_id, img)
    shutil.copy(src, dst)

print(f"已提取 {class_id} 类别的 10 张图片")
