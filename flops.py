from ultralytics import YOLO

model = YOLO('PEAG-Det.yaml')
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
model.fuse()