r"""
将“分类/浆液性”里的图片对应的VOC标注 (<object><name>…</name>) 全部改为 SCN，
并写入到新的标注输出目录。.

用法示例（Windows 路径带空格与中文都OK）：
python fix_voc_ann_to_scn.py ^
  --class_dir "F:\datasets\第一批胰腺\胰腺数据 (2)\胰腺数据\分类\浆液性" ^
  --ann_dir   "F:\datasets\第一批胰腺\胰腺数据 (2)\胰腺数据\检测\VOCdevkit\VOC2007\Annotations" ^
  --out_dir   "F:\datasets\第一批胰腺\胰腺数据 (2)\胰腺数据\检测\VOCdevkit\VOC2007\Annotations_SCN"

可选参数：
  --label SCN            # 目标类别名，默认 SCN
  --copy_others          # 额外把 ann_dir 里其余未命中的 XML 原样拷贝到 out_dir
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_basenames(class_dir: Path):
    """收集‘分类/浆液性’目录下所有图片的basename（不含扩展名）."""
    names = set()
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            names.add(p.stem)
    return names


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rewrite_xml_to_label(xml_in: Path, xml_out: Path, target_label: str):
    """把 xml_in 中所有 <object><name>xxx</name> 改为 target_label，并写到 xml_out."""
    try:
        # 尽量保留原有结构
        ET.register_namespace("", "")  # 防止多余 ns
    except Exception:
        pass

    tree = ET.parse(xml_in)
    root = tree.getroot()

    changed = False
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is not None and name_node.text != target_label:
            name_node.text = target_label
            changed = True

    # 如果没有<object>节点，也照常写出（有些文件可能是空目标；按需保留原文件）
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    # 指定编码与 XML 声明
    tree.write(xml_out, encoding="utf-8", xml_declaration=True)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--class_dir",
        default="F:\datasets\第一批胰腺\胰腺数据\分类\黏液性",
        type=Path,
        help="分类/浆液性 文件夹路径（包含图片）",
    )
    ap.add_argument(
        "--ann_dir",
        default="F:\datasets\第一批胰腺\胰腺数据\检测\VOCdevkit\VOC2007\Annotations",
        type=Path,
        help="VOC Annotations 原标注文件夹（XML）",
    )
    ap.add_argument(
        "--out_dir",
        default="F:\datasets\第一批胰腺\胰腺数据\检测\VOCdevkit\VOC2007\Annotations_SCN",
        type=Path,
        help="新的标注输出文件夹",
    )
    ap.add_argument("--label", default="MCN", help="目标类别名，默认 SCN")
    ap.add_argument("--copy_others", action="store_true", help="把未命中的其它 XML 也原样拷贝到 out_dir")
    args = ap.parse_args()

    class_dir: Path = args.class_dir
    ann_dir: Path = args.ann_dir
    out_dir: Path = args.out_dir
    target_label: str = args.label

    assert class_dir.exists(), f"class_dir 不存在: {class_dir}"
    assert ann_dir.exists(), f"ann_dir 不存在: {ann_dir}"
    ensure_outdir(out_dir)

    # 1) 取“浆液性”目录的图片 basenames
    basenames = collect_basenames(class_dir)
    if not basenames:
        print(f"[WARN] 在 {class_dir} 未找到图片；检查路径或扩展名。")
        return

    print(f"[INFO] 将处理 {len(basenames)} 个图片对应的 XML，全部改为 {target_label}")

    # 2) 遍历 basenames，定位对应 XML 并改写
    num_ok, num_missing, num_changed = 0, 0, 0
    for stem in sorted(basenames):
        xml_in = ann_dir / f"{stem}.xml"
        if not xml_in.exists():
            num_missing += 1
            print(f"[MISS] 未找到标注: {xml_in}")
            continue

        xml_out = out_dir / xml_in.name
        changed = rewrite_xml_to_label(xml_in, xml_out, target_label)
        num_ok += 1
        if changed:
            num_changed += 1

    # 3) 可选：把未命中的其它 XML 也原样拷贝过去（方便形成一套“完整的”Annotations）
    if args.copy_others:
        all_xmls = set(p.name for p in ann_dir.glob("*.xml"))
        done_xmls = set(f"{s}.xml" for s in basenames if (ann_dir / f"{s}.xml").exists())
        others = sorted(all_xmls - done_xmls)
        for name in others:
            src = ann_dir / name
            dst = out_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)
        print(f"[INFO] 其余未命中的 XML 已拷贝：{len(others)} 个")

    print(f"[DONE] 总计：命中 {num_ok}，改写 {num_changed}，缺失XML {num_missing}。输出目录：{out_dir}")


if __name__ == "__main__":
    main()
