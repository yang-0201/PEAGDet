"""
说明：
该文件用于生成测试json文件
1. image_id：标号从1开始
2. 图像记录长款尺寸
3. 标记id和图片名对应，通过读取，文件名排序，能够使得id和文件名对应.
"""

import argparse
import json
import os

from PIL import Image

# 官方提供了类别ID
CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "van"},
    {"id": 4, "name": "truck"},
]


def generate_image_json(image_dir, output_json_path):
    image_list = []
    image_files = sorted(os.listdir(image_dir))  # 按文件名排序

    image_id = 1
    for file_name in image_files:
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            image_path = os.path.join(image_dir, file_name)
            with Image.open(image_path) as img:
                width, height = img.size

            image_info = {"id": image_id, "file_name": file_name, "width": width, "height": height}
            image_list.append(image_info)
            image_id += 1

    result_json = {"images": image_list, "categories": CATEGORIES}

    with open(output_json_path, "w") as f:
        json.dump(result_json, f, indent=4)
    print(f"✅ JSON saved to: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate image metadata JSON from a folder of images.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=r"E:\BaiduNetdiskDownload\2\images\val",
        help="Path to the folder containing images",
    )
    parser.add_argument("--output_json", type=str, default="test_info.json", help="Output path for the JSON file")
    args = parser.parse_args()

    generate_image_json(args.image_dir, args.output_json)


if __name__ == "__main__":
    main()
