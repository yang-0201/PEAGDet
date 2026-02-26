import json

from tqdm import tqdm


def convert_pred_to_coco(test_info_path, pred_path, output_path):
    # 加载测试集元数据
    with open(test_info_path) as f:
        test_info = json.load(f)

    # 创建image_id映射字典（去除.jpg后缀的文件名 -> 数字ID）
    image_id_map = {img["file_name"].replace(".jpg", ""): img["id"] for img in test_info["images"]}

    # 加载原始预测数据
    with open(pred_path) as f:
        predictions = json.load(f)

    # 构建COCO数据集结构
    dataset = {
        "images": test_info["images"],  # 直接使用test_info中的图片信息
        "annotations": [],
        "categories": [
            {"id": 2, "name": "person"},
            {"id": 1, "name": "car"},
            {"id": 4, "name": "truck"},
            {"id": 3, "name": "van"},
        ],
    }

    # 转换预测数据
    for ann_id, pred in tqdm(enumerate(predictions, 1)):
        # 获取映射后的数字image_id
        mapped_id = image_id_map.get(pred["image_id"] + ".png")
        if not mapped_id:
            continue  # 跳过无效的image_id

        # 转换bbox格式（COCO要求[x,y,width,height]）
        x, y, w, h = pred["bbox"]

        dataset["annotations"].append(
            {
                "id": ann_id,
                "image_id": mapped_id,
                "category_id": pred["category_id"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "score": pred["score"],
                "iscrowd": 0,
            }
        )

    # 保存结果
    with open(output_path, "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    convert_pred_to_coco(
        test_info_path="test_info.json",  # 测试集元数据文件
        pred_path=r"E:\tianchi\MHAF-yolov12\runs\detect\val22\predictions.json",  # 原始预测文件
        output_path="./output_path/pred.json",  # 输出文件路径
    )
