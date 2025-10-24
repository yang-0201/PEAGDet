from ultralytics import YOLO

# docker run -v E:/datasets:/input/datasets -v ./:/output/ --gpus all --memory 48g --shm-size 16G infer python predict.py
if __name__ == "__main__":
    # model = YOLO(r'E:\tianchi\MHAF-yolov12\runs\detect\train33\weights/last.pt')
    # model = YOLO('best_133_796_change.pt')
    # model = YOLO(r'dual-MAF-YOLOv2-n.yaml').load('best_133_796.pt')  # 41 person
    # model = YOLO(r'best_dual_63_1280_837.pt') # 1728  原图  86/679  230-610 856/653  861(conf 0.15)  240-600  853/647 858(conf 0.15)
    model = YOLO(r"best_882.pt")  # 1728  原图  87/679  230-610 856/653  861(conf 0.15)  240-600  853/647 858(conf 0.15)

    model.val(
        data="tianchi.yaml",
        split="val",
        imgsz=1632,
        agnostic_nms=False,
        batch=16,
        device=0,
        conf=0.14,
        save_json=False,
        save_txt=False,
        save_conf=False,
        half=True,
    )
# best  1280  851  672
