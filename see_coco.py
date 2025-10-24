import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from tqdm import tqdm

# ==== 配置路径 ====
json_path = "./output_path/pred.json"  # 替换为你的COCO格式预测文件
image_dir = r"E:\tianchi\ultralytics\sliced_val_add_person\images\val"        # 图像原图目录
output_dir = "./vis_results"         # 输出目录
os.makedirs(output_dir, exist_ok=True)

# ==== 加载 COCO 格式数据 ====
with open(json_path, 'r') as f:
    coco_data = json.load(f)

images = coco_data["images"]
annotations = coco_data["annotations"]
categories = coco_data["categories"]

# ==== 构造类别映射 ====
cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

# ==== 构造 image_id -> file_name 映射 ====
img_id_to_file = {img["id"]: img["file_name"] for img in images}

# ==== 构造 image_id -> list[annotations] 映射 ====
img_id_to_anns = defaultdict(list)
for ann in annotations:
    img_id_to_anns[ann["image_id"]].append(ann)

# ==== 可视化并保存 ====
def visualize_and_save(image_id, file_name, anns):
    img_path = os.path.join(image_dir, file_name)
    image = Image.open(img_path).convert("RGB")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for ann in anns:
        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id, "unknown")
        score = ann.get("score", None)
        if score < 0.5:
            continue
        label = f"{cat_name}"
        if score is not None:
            label += f" {score:.2f}"

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 2, label, color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=0))

    ax.axis('off')
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ==== 批量执行 ====
print("开始批量可视化并保存图像...")
for img in tqdm(images):
    image_id = img["id"]
    file_name = img["file_name"]
    anns = img_id_to_anns.get(image_id, [])
    visualize_and_save(image_id, file_name, anns)

print(f"✅ 所有可视化图像已保存到：{output_dir}")
