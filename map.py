import argparse
import json
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt", type=str, help="Assign the groud true path.", default=r"E:\datasets\annotations/val.json"
    )
    parser.add_argument(
        "--dt",
        type=str,
        help="Assign the detection result path.",
        default=r"E:\tianchi\ultralytics\runs\detect\val97\predictions.json",
    )

    args = parser.parse_args()
    gt_path = args.gt
    dt_path = args.dt

    anno_json = gt_path
    pred_json = dt_path
    with open(pred_json) as fp:
        pred_dict = json.load(fp)

    with open(anno_json) as fg:
        gt_dict = json.load(fg)
    name2id = OrderedDict()
    for image in gt_dict["images"]:
        name2id[image["file_name"].split(".")[0]] = image["id"]

    for annotations in pred_dict:
        image_id = annotations["image_id"]
        id = name2id[image_id]
        annotations["image_id"] = id
    with open("new_pred.json", "w") as fp:
        json.dump(pred_dict, fp)
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes("new_pred.json")  # init predictions api
    eval = COCOeval(anno, pred, "bbox")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50, map75 = eval.stats[:3]  # update results (mAP@0.5:0.95, mAP@0.5)
    print(f"map50:95 {map}  map50 {map50}  map75 {map75}  score {(map50 + map75) / 2}")
