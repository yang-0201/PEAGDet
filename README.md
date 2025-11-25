# PEAGDet



This is the official code for the paper: PEAG-Det: A Dual-Stream Phase- and Edge-Aware Framework with Adaptive Gating for Cross-Phase Tumor Detection
<div align="center">
 <a href="./">
     <img src="exp3.png" width="100%"/>
</a>
 
</div>

## Train
Single GPU training
```python
# train.py
from ultralytics import YOLOv10
if __name__ == '__main__':
    model = YOLOv10('PEAG-Det.yaml')
    model.train(data="dual_yixian.yaml", epochs=300,batch=64,imgsz=640 ,device='0', workers=4,val_period= 1, scale=0.9)

```
## Val
```python
# val.py
from ultralytics import YOLOv10
if __name__ == '__main__':
    model = YOLOv10('best.pt')
    model.val(data='dual_yixian.yaml', device=0,split='val', batch=8)
```
## Model Architecture
```python
# Parameters
ch: 6
nc: 5 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
#  n: [0.33, 0.25, 1024]
  n: [0.33, 0.25, 1024]
# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Dual_in, []] # 0
  - [-1, 1, Dual_out, [1]] # 1
  - [-1, 1, Conv, [64, 3, 2]] # 2-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 3-P2/4
  - [-1, 1, RepHEA, [128, 3, 1, 3, 3]] #4
  - [-1, 1, Conv, [256, 3, 2]] # 5-P3/8
  - [-1, 1, RepHEA, [256, 3, 1, 3, 5]]
  - [-1, 1, SCDown, [512, 3, 2]] # 7-P4/16
  - [-1, 1, RepHEA, [512, 3, 1, 3, 7]]
  - [-1, 1, SCDown, [768, 3, 2]] # 9-P5/32
  - [-1, 1, RepHEA, [768, 3, 1, 3, 9]]
  - [-1, 1, SPPF, [768, 5]] # 11
  - [-1, 1, PSA, [768]] # 12

  - [0, 1, Dual_out, [0]] #
  - [-1, 1, Conv, [64, 3, 2]] #
  - [-1, 1, Conv, [128, 3, 2]] #
  - [-1, 1, RepHEA, [128, 3, 1, 3, 3]]  #17  16
  # - [[4, -1], 1, TFCF_GTFA, [32, 30, 30]]   #old_6_20  30   25
  - [[4, -1], 1, Concat, [1]] #
  - [-1, 1, Conv, [256, 3, 2]] #
  - [-1, 1, RepHEA, [256, 3, 1, 3, 5]] # 20  19
  - [[6, -1], 1, PCAF_DPAG, [48, 40, 40, 8, 3, 1 ]]   #old_6_20  30   20
  - [-1, 1, SCDown, [512, 3, 2]] # 5-P4/16
  - [-1, 1, RepHEA, [512, 3, 1, 3, 7]] # 23  22
  - [[8,-1], 1, PCAF_DPAG, [96, 30, 30, 8, 3, 1]]    # old_8_23  31   23
  - [-1, 1, SCDown, [768, 3, 2]] # 7-P5/32
  - [-1, 1, RepHEA, [768, 3, 1, 3, 9]] # 26  25
  - [-1, 1, SPPF, [768, 5]] #
  - [-1, 1, PSA, [768]] # 29
  - [[12,-1], 1, PCAF_DPAG, [192, 20, 20, 8, 3, 1]]    # old_12_29  32   28

head:
  - [23, 1, AVG, []]
  - [[-1, 28], 1, Concat, [1]]
  - [-1, 1, RepHEA, [512, 3, 1, 3, 9]] #31

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [28 ,1, nn.Upsample, [None, 2, "nearest"]]
  - [20, 1, AVG, []]
  - [[-1, 23, -2, -3], 1, Concat, [1]]
  - [-1, 1, RepHEA, [384, 3, 1, 3, 7]] #36

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [23, 1, nn.Upsample, [None, 2, "nearest"]]
  - [17, 1, AVG, []]
  - [[-1, 20, -2, -3], 1, Concat, [1]]
  - [-1, 1, RepHEA, [256, 3, 1, 3, 5]] #41

  - [36, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, -2], 1, Concat, [1]]
  - [-1, 1, RepHEA, [256, 3, 1, 3, 5]] # 59  45

  - [-1, 1, Conv, [384, 3, 2]]
  - [41, 1, AVG, []]
  - [31, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-2, -1, 36, -3], 1, Concat, [1]]
  - [-1, 1, RepHEA, [384, 3, 1, 3, 7]] # 66  50

  - [-1, 1, Conv, [512, 3, 2]]
  - [36, 1, AVG, []]
  - [[-2, -1, 31], 1, Concat, [1]]
  - [-1, 1, RepHEA, [512, 3, 1, 3, 9]] # 71   54
  - [[44, 49, 53], 1, v10Detect, [nc]] # Detect(P3, P4, P5)

```
## Model  Figures
<div align="center">
 <a href="./">
     <img src="exp1.png" width="100%"/>

</a>
 
</div>
