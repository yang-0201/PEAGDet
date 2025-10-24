import torch

from ultralytics import YOLO

ckpt = YOLO(r"best_133_796.pt")
new_model = YOLO("dual-MAF-YOLOv2-n.yaml")
# idx = 0
for k, v in new_model.model.state_dict().items():
    idx = int(k.split(".")[1])
    kr = k.replace(f"model.{idx}.", f"model.{idx - 1}.")
    # kr = k
    new_model.model.state_dict()[k] -= new_model.model.state_dict()[k]
    print(k)
    new_model.model.state_dict()[k] += ckpt.model.state_dict()[kr]
torch.save(new_model, "./exp.pt")
