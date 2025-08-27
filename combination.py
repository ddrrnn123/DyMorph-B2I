import os
import glob
import cv2
import numpy as np
import json
import re
from tqdm import tqdm
import random
import glob
import cv2
import numpy as np
import json
import re
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy import ndimage


def filter_by_length(items, min_len, max_len):
    kept = []
    for item in items:
        # line as tuple of pts
        if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(pt, (list, tuple)) for pt in item):
            p1, p2 = item
        else:
            # contour array
            arr = np.array(item)
            pts = arr.reshape(-1, 2)
            if pts.shape[0] < 2:
                continue
            p1, p2 = tuple(pts[0]), tuple(pts[-1])
        L = np.linalg.norm(np.array(p1) - np.array(p2))
        if min_len <= L <= max_len:
            kept.append(item)
    return kept

def find_main_defect_line(region_mask, thickness=7):
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull) if hull is not None and len(hull) >= 3 else None
    if defects is None or len(defects) < 2:
        return None, None, None
    # choose the two deepest concave points
    d1, d2 = sorted(defects, key=lambda x: x[0][3], reverse=True)[:2]
    pt1 = tuple(cnt[d1[0][2]][0])
    pt2 = tuple(cnt[d2[0][2]][0])
    mask_line = np.zeros_like(region_mask)
    cv2.line(mask_line, pt1, pt2, 255, thickness)
    return mask_line, pt1, pt2

def skeleton_watershed_joint(mask,
                             max_width=7,
                             fg_thresh=0.35,
                             kernel=5,
                             min_area=200):
    mask_bin = (mask > 0).astype(np.uint8)
    # distance transform
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, kernel)
    sure_fg = (dist > fg_thresh * dist.max()).astype(np.uint8)
    seeds, _ = ndimage.label(sure_fg)
    markers = seeds + 1
    markers[(mask_bin - sure_fg) > 0] = 0
    mask_3c = cv2.cvtColor(mask_bin * 255, cv2.COLOR_GRAY2BGR)
    wshed = cv2.watershed(mask_3c, markers)

    cut_lines = []
    for lbl in [l for l in np.unique(wshed) if l > 1]:
        region = ((wshed == lbl) & (mask_bin == 1)).astype(np.uint8)
        if cv2.countNonZero(region) < min_area:
             continue
        # for very large regions, derive a defect line
        if region.sum() > (min_area * 10):
            region255 = (region * 255).astype(np.uint8)
            mask_line, pt1, pt2 = find_main_defect_line(region255)
            if pt1 and pt2:
                cut_lines.append((pt1, pt2))
    return cut_lines

def separate_objects_skeleton_watershed(binary_mask,
                                        min_len,
                                        max_len,
                                        length_filter_enable=False,
                                        dist_kernel=5,
                                        fg_thresh=0.35,
                                        min_area=200):
    if len(binary_mask.shape) == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)
    cut_lines = skeleton_watershed_joint(mask_bin,
                                         fg_thresh=fg_thresh,
                                         kernel=dist_kernel,
                                         min_area=min_area)
    if length_filter_enable:
        kept = filter_by_length(cut_lines, min_len, max_len)
    else:
        kept = cut_lines
    contours = []
    for pt1, pt2 in kept:
        pts = np.array([pt1, pt2], dtype=np.int32)
        contours.append(pts)
    return contours

def combination(
    mask_path,   
    min_len,
    max_len,               
    dist_kernel=5,
    fg_thresh=0.35,
    length_filter_enable=False,
    min_area=200
):
   
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = ((mask > 0).astype(np.uint8) * 255)

    cut_line_contours = separate_objects_skeleton_watershed(
        mask_bin,
        min_len,
        max_len, 
        length_filter_enable=length_filter_enable,
        dist_kernel=dist_kernel,
        fg_thresh=fg_thresh,
        min_area=min_area
    )
    cut_lines = []
    for c in cut_line_contours:
        pts = np.asarray(c, dtype=np.int32).reshape(-1, 2)
        if pts.shape[0] == 2:
            cut_lines.append((tuple(pts[0]), tuple(pts[1])))

    mask_cut = mask_bin.copy()
    if cut_lines:
        line_mask = np.zeros_like(mask_cut)
        for (p1, p2) in cut_lines:
            cv2.line(line_mask, p1, p2, 255, line_thickness)
        mask_cut[line_mask == 255] = 0

    # extract external contours as final instances
    cnts, _ = cv2.findContours(mask_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_rings = []
    for c in seg_rings:
        if cv2.contourArea(c) >= min_area:
            continue
        if len(c) >= 3:
            seg_rings.append(c.reshape(-1, 2).tolist())


    return seg_rings


