import os
import glob
import re
import json
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# config
mask_root = "./KI_dataset_demo"
output_json = "./KI.json"
os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

MIN_AREA = 100  

# modality filter (None = keep all)
# TARGET_MODALITIES = {"he", "pas"}
TARGET_MODALITIES = None

# class filter (None = keep all)
# TARGET_CLASSES = {"cap", "pt"}
TARGET_CLASSES = None

category_ids = {"cap": 1, "dt": 2, "pt": 3, "ptc": 4, "tuft": 5, "ves": 6}


def create_category_annotation(d):
    return [{"id": v, "name": k, "supercategory": k} for k, v in d.items()]

def create_image_annotation(fn, w, h, i, stain, mn):
    info = {"id": i, "file_name": fn, "width": w, "height": h, "stain": stain}
    if mn:
        info["mask_name"] = mn
    return info


def find_original_image(mask_path):
    m = re.match(r"(.+)_mask.*\.png", os.path.basename(mask_path), flags=re.IGNORECASE)
    if not m:
        return None
    base = m.group(1)
    d = os.path.dirname(mask_path)
    for ext in (".tif", ".png", ".jpg", ".jpeg"):
        cand = os.path.join(d, base + ext)
        if os.path.exists(cand):
            return cand
    return None


def get_segmentation_with_holes(inst_mask):
    contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []
    segs = []
    h0 = hierarchy[0]
    for idx, h in enumerate(h0):
        if h[3] == -1:
            outer = contours[idx]
            if len(outer) < 3:
                continue
            segs.append(outer.reshape(-1, 2).astype(float).flatten().tolist())
            child = h[2]
            while child != -1:
                hole = contours[child]
                if len(hole) >= 3:
                    segs.append(hole.reshape(-1, 2).astype(float).flatten().tolist())
                child = h0[child][0]
    return segs


def process_one_mask(args):
    mask_path, img_id = args
    rel = os.path.relpath(mask_path, mask_root)
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return None, []
    category_name, stain = parts[0], parts[1]
    if TARGET_CLASSES and category_name not in TARGET_CLASSES:
        return None, []
    if TARGET_MODALITIES and stain not in TARGET_MODALITIES:
        return None, []
    cat_id = category_ids.get(category_name)
    if cat_id is None:
        return None, []
    orig = find_original_image(mask_path)
    if not orig:
        return None, []
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if msk is None:
        return None, []
    h, w = msk.shape
    imginfo = create_image_annotation(
        os.path.relpath(orig, mask_root), w, h, img_id, stain, rel
    )
    mask_bin = (msk > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    annos = []
    for lbl in range(1, num_labels):
        area_px = int(stats[lbl, cv2.CC_STAT_AREA])
        if area_px < MIN_AREA:
            continue
        inst = (labels == lbl).astype(np.uint8) * 255
        segs = get_segmentation_with_holes(inst)
        if not segs:
            continue
        outer_pts = np.array(segs[0], dtype=np.float32).reshape(-1, 2)
        x, y, bw, bh = cv2.boundingRect(outer_pts.astype(np.int32))
        area = float(cv2.contourArea(outer_pts))
        annos.append({
            "iscrowd": 0,
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "area": max(area, 0.0),
            "segmentation": segs
        })
    return imginfo, annos


if __name__ == "__main__":
    if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
        with open(output_json, "r") as f:
            data = json.load(f)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", create_category_annotation(category_ids))
        max_img_id = max((img.get("id", -1) for img in images), default=-1)
    else:
        images = []
        annotations = []
        categories = create_category_annotation(category_ids)
        max_img_id = -1

    all_masks = glob.glob(os.path.join(mask_root, "**", "*_mask*.png"), recursive=True)
    filtered_masks = []
    for p in all_masks:
        rel = os.path.relpath(p, mask_root)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            continue
        category_name, modality = parts[0], parts[1]
        if TARGET_CLASSES and category_name not in TARGET_CLASSES:
            continue
        if TARGET_MODALITIES and modality not in TARGET_MODALITIES:
            continue
        filtered_masks.append(p)

    args = [(p, max_img_id + 1 + idx) for idx, p in enumerate(filtered_masks)]

    max_workers = min(16, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for imginfo, annos in tqdm(
            ex.map(process_one_mask, args, chunksize=8),
            total=len(args), desc="Processing"
        ):
            if imginfo is not None:
                images.append(imginfo)
            if annos:
                annotations.extend(annos)

    for idx, ann in enumerate(annotations):
        ann["id"] = idx

    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
