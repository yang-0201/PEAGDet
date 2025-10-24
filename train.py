from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO(r'yolov10m_aux.yaml').load("628_945_739_640.pt")
    # model = YOLO(r'E:\tianchi\ultralytics\runs\detect\train259/weights/last.pt')  # 41 person
    model = YOLO(r"PEAG-Det.yaml")
    # model = YOLO(r'dual-MAF-YOLOv2-n-v2.1.yaml').load('best_882.pt') # 41 person
    # model = YOLO(r'E:\tianchi\Dual-MHAF-yolov12\runs\detect\train5/weights/last.pt')
    # model.train(resume=True, data='tianchi.yaml', imgsz=1024, batch=24, device=0, mixup=0.1, copy_paste=0.2, scale=0.9, cutmix=0.1, epochs=200)
    model.train(
        resume=False,
        data="yixian.yaml",
        imgsz=640,
        epochs=200,
        lr0=0.01,
        optimizer="SGD",
        multi_scale=False,
        batch=16,
        device=0,
        copy_paste=0,
        scale=0.5,
        name="train-exp1",
        exist_ok=True,
    )
