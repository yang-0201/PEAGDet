import torch
from ultralytics import YOLO

ckpt = YOLO(r'MAF-YOLOv2-N.pt')
new_model = YOLO('dual-MAF-YOLOv2-n-v2.yaml')
# idx = 0
for k, v in new_model.model.state_dict().items():
    idx = int(k.split('.')[1])
    if idx >=4 and idx <= 13:
        kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx - 3))
        # kr = k
        new_model.model.state_dict()[k] -= new_model.model.state_dict()[k]
        print(k)
        new_model.model.state_dict()[k] += ckpt.model.state_dict()[kr]
    elif idx >=15 and idx <= 24:
        kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx - 14))   #第一个卷积为3通道不是6
        # kr = k
        new_model.model.state_dict()[k] -= new_model.model.state_dict()[k]
        print(k)
        new_model.model.state_dict()[k] += ckpt.model.state_dict()[kr]
    # else:
    #     del new_model.model.state_dict()[k]
    # elif idx in [31, 36, 41, 44, 45, 49, 54]:
    #     continue
    elif idx ==54:
        try:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx - 18))
            # kr = k
            new_model.model.state_dict()[k] -= new_model.model.state_dict()[k]
            print(k)
            new_model.model.state_dict()[k] += ckpt.model.state_dict()[kr]
        except:
            continue
    elif idx >=29:
        print(k)
        kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx - 18))
        # kr = k
        new_model.model.state_dict()[k] -= new_model.model.state_dict()[k]
        print(k)
        new_model.model.state_dict()[k] += ckpt.model.state_dict()[kr]

new_model.save('dual_n_origin.pt')
# torch.save(new_model, "./dual.pt")

