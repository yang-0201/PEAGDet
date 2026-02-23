import argparse
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate COCO mAP for predictions")
    parser.add_argument(
        "--ann",
        default=r"E:\BaiduNetdiskDownload\2\annotations/val_annotations.json",
        type=str,
        help="Path to ground truth annotation file (COCO format)",
    )
    parser.add_argument(
        "--pred", default="./output_path/pred.json", type=str, help="Path to predicted results file (COCO format)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load GT and prediction
    cocoGt = COCO(args.ann)
    with open(args.pred) as f:
        yolo_detections = json.load(f)

    # If pred is a dict (COCO format), convert to detection list
    if isinstance(yolo_detections, dict) and "annotations" in yolo_detections:
        # Convert annotation format to detection format
        yolo_detections = [
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": ann.get("score", 1.0),  # If no score, set to 1.0
            }
            for ann in yolo_detections["annotations"]
        ]

    # Check image_id type consistency
    gt_image_ids = set(img["id"] for img in cocoGt.dataset["images"])
    for det in yolo_detections:
        if det["image_id"] not in gt_image_ids:
            raise ValueError(f"Detection image_id {det['image_id']} not in GT images.")

    # Evaluate
    cocoDt = cocoGt.loadRes(yolo_detections)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    map, map50, map75 = cocoEval.stats[:3]
    print(f"map50:95 {map:.4f}  map50 {map50:.4f}  map75 {map75:.4f}  score {(map50 + map75) / 2:.4f}")

    # Per-category evaluation
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    {cat["id"]: cat["name"] for cat in cats}
    precisions = cocoEval.eval["precision"]  # [T, R, K, A, M]
    area_idx = 0  # area='all'
    max_det_idx = 2  # maxDets=100

    print("\nPer-class mAP@[IoU=0.50:0.95]:")
    for idx, cat in enumerate(cats):
        p_all = precisions[:, :, idx, area_idx, max_det_idx]
        valid = p_all > -1
        ap = np.mean(p_all[valid]) if valid.any() else float("nan")
        print(f"{cat['name']:>20s}: {ap:.3f}")

    print("\nPer-class AP@IoU=0.50 and AP@IoU=0.75:")
    for idx, cat in enumerate(cats):
        # AP50
        precision_50 = precisions[0, :, idx, area_idx, max_det_idx]
        precision_50 = precision_50[precision_50 > -1]
        ap50 = np.mean(precision_50) if precision_50.size else float("nan")

        # AP75
        precision_75 = precisions[5, :, idx, area_idx, max_det_idx]
        precision_75 = precision_75[precision_75 > -1]
        ap75 = np.mean(precision_75) if precision_75.size else float("nan")

        print(f"{cat['name']:>20s} | AP50 = {ap50:.3f} | AP75 = {ap75:.3f}")


if __name__ == "__main__":
    main()
