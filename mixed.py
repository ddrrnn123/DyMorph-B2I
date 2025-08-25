import os
import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy import ndimage
from combination import separate_objects_skeleton_watershed, filter_by_length
from classic_method import find_contour, watershed, skeleton, morphology

# config
mask_root = "./KI_dataset_demo"
json_path = "./KI.json"
output_vis_dir = "./output"

JPEG_QUALITY = 1   

# drawing colors/thickness
ORIG_LINE_COLOR = (0, 255, 0)  
MASK_LINE_COLOR = (0, 0, 255)  
LINE_THICKNESS = 2            
category_ids = {"cap": 1, "dt": 2, "pt": 3, "ptc": 4, "tuft": 5, "ves": 6}
ORIG_EXTS = ['.png', '.jpg', '.jpeg']


def separate_objects_watershed(binary_mask,
                               dist_kernel,
                               fg_thresh,
                               length_filter_enable,
                               min_len,
                               max_len,
                               min_area):
    if len(binary_mask.shape) == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, dist_kernel)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(dist, fg_thresh * dist.max(), 255, cv2.THRESH_BINARY)
    fg = fg.astype(np.uint8)
    unknown = cv2.subtract(binary_mask, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_mask, markers)
    contours = []
    for v in range(2, markers.max() + 1):
        comp = (markers == v).astype(np.uint8) * 255
        if cv2.countNonZero(comp) < min_area:
            continue
        cts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours.extend(cts)
        if length_filter_enable:
            contours = filter_by_length(contours, min_len, max_len)
        contours.extend(cts)
    return contours

def load_coco(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def save_coco(data, json_file):
    for ann in data.get('annotations', []):
        if isinstance(ann.get('segmentation_rings'), np.ndarray):
            ann['segmentation_rings'] = ann['segmentation_rings'].tolist()
        if isinstance(ann.get('segmentation_lines'), np.ndarray):
            ann['segmentation_lines'] = ann['segmentation_lines'].tolist()
        lines = ann.get('cut_lines')
        if lines is not None:
            new_lines = []
            for ln in lines:
                if isinstance(ln, np.ndarray):
                    new_lines.append(ln.tolist())
                else:
                    new_lines.append(ln)
            ann['cut_lines'] = new_lines

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def find_image_id(data, mask_relpath):
    for img in data['images']:
        if img.get('mask_name') == mask_relpath:
            return img['id']
    return None

def determine_category_id(mask_path):
    rel = os.path.relpath(mask_path, mask_root)
    parts = rel.split(os.sep)
    return category_ids.get(parts[0], 0)

def update_segmentation_for_mask(
    mask=None,             
    mask_path=None,        
    json_file=None,        
    method='ws',           
    dist_kernel=5,          
    fg_thresh=0.35,
    min_len=0,
    max_len=200,
    length_filter_enable=False,
    min_area=0,
    inter_collect=False,
    return_lines=False,      # True: only return lines; do not write JSON
    all_lines=None,         
    write_json=False        # True: perform JSON writing branch
):
    # apply provided lines and write JSON
    if all_lines is not None and write_json:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        data = load_coco(json_file)
        rel = os.path.relpath(mask_path, mask_root)
        img_id = find_image_id(data, rel)
        if img_id is None:
            raise ValueError(f"Mask {rel} not in JSON")
        mask_bin = (mask > 0).astype(np.uint8) * 255
        mask_cut = mask_bin.copy()

        for ln in all_lines:
            pts = np.array(ln, dtype=np.int32).reshape(-1, 2)
            line_mask = np.zeros_like(mask_bin)
            if pts.shape[0] == 2:
                p1, p2 = tuple(pts[0]), tuple(pts[1])
                cv2.line(line_mask, p1, p2, 255, thickness = 2)
            else:
                cv2.polylines(line_mask, [pts], False, 255, thickness = 2)
            mask_cut[line_mask == 255] = 0

        cnts, hierarchy = cv2.findContours(
            mask_cut,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        seg_rings = []
        if hierarchy is not None:
            for idx, h in enumerate(hierarchy[0]):
                parent = h[3]
                collect = inter_collect or (parent == -1)
                if not collect:
                    continue
                area = cv2.contourArea(cnts[idx])
                if area < min_area:
                    continue
                pts = cnts[idx].reshape(-1, 2)
                if pts.shape[0] >= 3:
                    seg_rings.append(pts.flatten().astype(int).tolist())

        seg_lines = []
        for ln in all_lines:
            pts = np.array(ln, dtype=np.int32).reshape(-1, 2)
            if pts.shape[0] == 2:
                seg_lines.append([int(pts[0][0]), int(pts[0][1]),
                                  int(pts[1][0]), int(pts[1][1])])
            else:
                seg_lines.append(pts.flatten().astype(int).tolist())
        segmentation = seg_rings + seg_lines

        data['annotations'] = [a for a in data['annotations'] if a['image_id'] != img_id]
        ann_id = max((a['id'] for a in data['annotations']), default=0) + 1
        x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
        area = float(cv2.countNonZero(mask))
        data['annotations'].append({
            'id': ann_id,
            'image_id': img_id,
            'category_id': determine_category_id(mask_path),
            'iscrowd': 0,
            'bbox': [int(x), int(y), int(w), int(h)],
            'area': area,
            'segmentation': seg_rings,
            'cut_lines': seg_lines
        })
        save_coco(data, json_file)
        print(f"[json] Replace {len(segmentation)} elements for image_id={img_id}")
        return

    # create lines according to method/params
    if mask is None:
        raise ValueError("mask is required when generating lines")
    if method == 'ws':
        lines = separate_objects_watershed(mask, dist_kernel=dist_kernel, fg_thresh=fg_thresh,
                                           length_filter_enable=length_filter_enable, min_len=min_len, max_len=max_len,min_area=min_area)
    elif method == 'com':
        lines = separate_objects_skeleton_watershed(mask, dist_kernel=dist_kernel, fg_thresh=fg_thresh,
                                                   length_filter_enable=length_filter_enable, min_len=min_len, max_len=max_len,min_area=min_area)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    if return_lines:
        return lines
    print(f"Replaced {len(segmentation)} segmentation elements for image_id={img_id}")

def composite_visualization(mask_path, json_file,
                            orig_image_path=None,
                            line_thickness=LINE_THICKNESS,
                            jpeg_quality=JPEG_QUALITY,
                            out_dir=None):
    data = load_coco(json_file)
    rel = os.path.relpath(mask_path, mask_root)
    img_id = find_image_id(data, rel)
    if img_id is None:
        raise ValueError(f"Mask {rel} not in JSON")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if orig_image_path and os.path.exists(orig_image_path):
        orig = cv2.imread(orig_image_path)
    else:
        base = os.path.splitext(os.path.basename(mask_path))[0].replace('_mask', '')
        dirp = os.path.dirname(mask_path)
        orig = None
        for ext in ORIG_EXTS:
            p = os.path.join(dirp, base + ext)
            if os.path.exists(p):
                orig = cv2.imread(p)
                break
        if orig is None:
            raise FileNotFoundError("Original image not found")
    canvas_orig = orig.copy()
    canvas_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for ann in data['annotations']:
        if ann['image_id'] == img_id:
            for seg in ann['segmentation']:
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(canvas_orig, [pts], True, ORIG_LINE_COLOR, line_thickness)
                cv2.polylines(canvas_mask, [pts], True, MASK_LINE_COLOR, line_thickness)
    h1, w1 = canvas_orig.shape[:2]
    canvas_mask = cv2.resize(canvas_mask, (w1, h1))
    comp = np.concatenate([canvas_orig, canvas_mask], axis=1)
    if out_dir is None:
        out_dir = os.path.dirname(json_file)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(mask_path))[0].replace('_mask', '')
    out_file = os.path.join(out_dir, f"{base}_composite.jpg")
    cv2.imwrite(out_file, comp, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    print(f"Composite image saved: {out_file}")


def main():
    global mask_root, json_path, output_vis_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_root', default=mask_root)
    parser.add_argument('--json_path', default=json_path)
    parser.add_argument('--output_vis_dir', default=output_vis_dir)
    parser.add_argument('--nums', nargs='+', type=int, required=True,
                        help='Image indices to process (e.g., --nums 51 167)')
    parser.add_argument('--mix_params', nargs='+',
                        help='List of segmentation parameter sets: method,dist_kernel,fg_thresh,min_len,max_len')
    parser.add_argument('--default_method', choices=['fc', 'ws', 'sk','mor'], default='ws',
                    help='Classic method used when --mix_params is not given')
    parser.add_argument('--min_area', type=int, default=200,
                        help='Minimum contour area to include')
    parser.add_argument('--inter_collect', action='store_true',
                        help='If set, collect inner contours as well')
    args = parser.parse_args()

    mask_root = args.mask_root
    json_path = args.json_path
    out_dir = args.output_vis_dir
    os.makedirs(out_dir, exist_ok=True)

    for n in args.nums:
        mask_rel = mask_pattern.format(n=n)
        orig_rel = orig_pattern.format(n=n)
        mask_p = os.path.join(mask_root, mask_rel)
        orig_p = os.path.join(mask_root, orig_rel)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        all_lines = []

        if args.mix_params:
            for param_str in args.mix_params:
                items = param_str.split(',')
                if len(items) != 5:
                    print(f'Invalid param set: {param_str}; expected 5 items: method,dist_kernel,fg_thresh,min_len,max_len')
                    continue
                method, dk, ft, mn, mx = items
                dk, ft, mn, mx = int(dk), float(ft), int(mn), int(mx)
                lines = update_segmentation_for_mask(
                    mask=mask,
                    method=method,
                    dist_kernel=dk,
                    fg_thresh=ft,
                    min_len=mn,
                    max_len=mx,
                    length_filter_enable=True,
                    return_lines=True 
                )
                print(f"[Debug] method={method}, dk={dk}, ft={ft}, min_len={mn}, max_len={mx} -> {len(lines)} lines")  
                all_lines.extend(lines)
        else:
            method = args.default_method   
            if method == 'fc':
                lines = find_contour(mask_p)
            elif method == 'ws':
                lines = watershed(mask_p)
            elif method == 'sk':
                lines = skeleton(mask_p)
            elif method == 'mor':
                lines = morphology(mask_p)
            else:
                raise ValueError(f"Unknown classic method: {method}")

            print(f"[Debug] classic method {method} -> {len(lines)} 条线")
            all_lines.extend(lines)

        update_segmentation_for_mask(
            mask_path=mask_p,
            json_file=json_path,
            all_lines=all_lines,
            inter_collect=args.inter_collect,
            min_area=args.min_area,
            write_json=True
        )
        composite_visualization(mask_p, json_file=json_path, orig_image_path=orig_p, out_dir=output_vis_dir)

if __name__ == '__main__':
    mask_pattern = "ves/he/im_{n}_mask_vessel.png"
    orig_pattern = "ves/he/im_{n}.png"
    main()

# python ./mixed.py --mix_params  ws,5,0.2,0,20 com,3,0.25,125,130 --nums 152