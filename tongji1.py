from collections import defaultdict
from pathlib import Path

# 目录与文件
path_pred = Path(r"E:\tianchi\Dual-MHAF-yolov12\runs\res191\labels_resume")  # 预测(含conf)的labels
path_orig = Path(r"E:\tianchi\Dual-MHAF-yolov12\runs\res191\labels")  # 原始labels(用于top_y检查)
image_path = Path(r"E:\datasets\images\testB")
result_path = Path("./result_cut.txt")


# 线性映射：把[0.14, 1.0]单调映射到[0.25, 1.0]
def remap_conf(conf, old_min=0.14, old_max=1.0, new_min=0.25, new_max=1.0):
    c = min(max(conf, old_min), old_max)
    scale = (new_max - new_min) / (old_max - old_min)  # 0.75 / 0.86  (按你给的0.14修正)
    new_c = new_min + (c - old_min) * scale
    if new_c < new_min:
        new_c = new_min
    if new_c > new_max:
        new_c = new_max
    return new_c


def get_top_y(yolo_bbox, img_h=720):
    cx, cy, w, h = yolo_bbox
    return (cy - h / 2) * img_h


# 收集图片基名（按文件名顺序写结果，缺label时写空行）
filenames = sorted(p.stem for p in image_path.iterdir() if p.is_file())

# ====== 新增：类别-置信度累加器（原始/重映射） ======
sum_conf_raw = defaultdict(float)  # {cls: sum of raw conf}
sum_conf_mapped = defaultdict(float)  # {cls: sum of mapped conf}
count_conf = defaultdict(int)  # {cls: count}

idx = 0  # 统计top_y<1的框数量
with result_path.open("w", encoding="utf-8") as out:
    out.write("3358655 19.8\n")
    # out.write("2662420 19.2\n")

    for name in filenames:
        pred_file = path_pred / f"{name}.txt"
        orig_file = path_orig / f"{name}.txt"

        # 先做top_y检查（若存在原始label）
        if orig_file.exists():
            with orig_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        box = list(map(float, parts[1:5]))
                        ty = get_top_y(box)
                        if ty < 1:
                            idx += 1
                            print(orig_file.name)
                            print(ty)

        # 写结果行：img.jpg + (x y w h conf class)...
        if not pred_file.exists():
            out.write(f"{name}.jpg\n")
            continue

        tokens = []
        with pred_file.open("r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip().split()
                # 期望格式：cls cx cy w h conf
                if len(s) < 6:
                    continue
                cls_token = s[0]
                # 尽量把类别转为int；如果失败就保持原字符串键
                try:
                    cls_id = int(float(cls_token))
                except:
                    cls_id = cls_token

                x, y, w, h = s[1:5]
                conf_raw = float(s[-1])  # 原始置信度
                conf_new = remap_conf(conf_raw)  # 重映射置信度

                # ====== 新增：累加统计 ======
                sum_conf_raw[cls_id] += conf_raw
                sum_conf_mapped[cls_id] += conf_new
                count_conf[cls_id] += 1

                tokens.extend([x, y, w, h, f"{conf_new:.6f}", str(cls_id)])

        if tokens:
            out.write(f"{name}.jpg {' '.join(tokens)}\n")
        else:
            out.write(f"{name}.jpg\n")

print(idx)

# ====== 新增：输出每个类别的平均置信度 ======
if count_conf:
    print("\n=== 每个类别的平均置信度（原始 / 重映射） ===")
    for cls_id in sorted(count_conf, key=lambda k: (isinstance(k, str), k)):
        cnt = count_conf[cls_id]
        avg_raw = sum_conf_raw[cls_id] / cnt if cnt else 0.0
        avg_map = sum_conf_mapped[cls_id] / cnt if cnt else 0.0
        print(f"cls {cls_id}: count={cnt}, avg_raw={avg_raw:.6f}, avg_mapped={avg_map:.6f}")
else:
    print("\n未找到任何预测框，无法统计类别置信度均值。")
