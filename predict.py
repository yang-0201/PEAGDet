from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r'best_882.pt') # 41 person    #80.98   [512, 1728]
    # model = YOLO(r'E:\tianchi\Dual-MHAF-yolov12\runs\detect\train5/weights/last.pt') # 41 person    #81.11   [512, 1728]

    model.predict(r"/input/datasets/cut-230-610/images/testB", device='0',save=False,imgsz=[480,1632], save_conf=True,conf=0.14,save_txt=True,project="runs",name="last", exist_ok=True)
    # model.train(data='qiyuan.yaml', imgsz=1024, batch=12, device=0, mixup=0.1, epochs=300)
