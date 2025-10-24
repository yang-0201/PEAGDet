from PIL.ImageOps import scale

from ultralytics import YOLO
if __name__ == '__main__':
    # model = YOLO(r'yolov10m_aux.yaml').load("628_945_739_640.pt")
    model = YOLO(r"E:\tianchi\ultralytics\runs\detect\train238/weights/last.pt")  # 41 person
    model.export(format="onnx", opset=15, half=True, imgsz=640)
    # model.train(data='qiyuan.yaml', imgsz=1024, batch=12, device=0, mixup=0.1, epochs=300)