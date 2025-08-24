import os
import glob
import cv2
import numpy as np
import json
import re
from tqdm import tqdm
import random
from skimage.morphology import skeletonize

def find_contour(mask_path,           
                min_area=200):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary = (mask > 0).astype(np.uint8) * 255
    seg_rings, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_rings = [c for c in seg_rings if cv2.contourArea(c) >= min_area]
    return seg_rings

def watershed(mask_path, 
            dist_kernel=5,
            fg_thresh=0.35,
            min_area=200):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary = (mask > 0).astype(np.uint8) * 255
    # Distance transform + foreground seeds
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, dist_kernel)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(dist, fg_thresh * dist.max(), 255, cv2.THRESH_BINARY)
    fg = fg.astype(np.uint8)
    unknown = cv2.subtract(binary, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_mask, markers)

    # Extract contours from separated labels
    seg_rings = []
    max_lbl = markers.max()
    for lbl in range(2, max_lbl + 1):          
        comp = (markers == lbl).astype(np.uint8) * 255
        if cv2.countNonZero(comp) < min_area:
            continue
        cts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cts:
            if cv2.contourArea(c) >= min_area and len(c) >= 3:
                pts = c.reshape(-1, 2).tolist()
                seg_rings.append(pts)

    return seg_rings


def skeleton(mask_path,
            min_area=200,
            local_radius=5,
            cut_length=50):
   

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary = (mask > 0).astype(np.uint8)

    skel = skeletonize(binary.astype(bool)).astype(np.uint8)
    # 8-neighborhood count
    kernel8 = np.ones((3,3), np.uint8)
    neigh = cv2.filter2D(skel, -1, kernel8)
    degree = neigh - skel

    # keypoints
    endpoints = np.where((skel==1) & (degree==1))
    junctions = np.where((skel==1) & (degree>2))
    pts_yx = list(zip(*endpoints)) + list(zip(*junctions))

    # cut along local normals around keypoints
    mask_cut = binary.copy()*255
    h, w = binary.shape
    for (y, x) in pts_yx:
        y0, x0 = y, x
        y1, y2 = max(0,y0-local_radius), min(h, y0+local_radius+1)
        x1, x2 = max(0,x0-local_radius), min(w, x0+local_radius+1)
        local = skel[y1:y2, x1:x2]
        ys, xs = np.nonzero(local)
        if len(xs) < 2:
            continue
        pts = np.stack([xs + x1, ys + y1], axis=1).astype(np.float32)
        vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)

        # normal direction
        nx, ny = -vy, vx
        norm = np.hypot(nx, ny)
        nx, ny = nx / norm, ny / norm

        # cut along both sides of the normal
        for sign in (-1, 1):
            for d in range(1, cut_length):
                xx = int(round(x0 + sign * nx * d))
                yy = int(round(y0 + sign * ny * d))
                if xx < 0 or yy < 0 or xx >= w or yy >= h or binary[yy, xx] == 0:
                    break
                mask_cut[yy, xx] = 0

    # find external contours on mask_cut
    seg_rings, hierarchy = cv2.findContours(
        mask_cut, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    seg_rings = []
    for c in seg_rings:
        if cv2.contourArea(c) >= min_area:
            continue
        if len(c) >= 3:
            seg_rings.append(c.reshape(-1, 2).tolist())

    return seg_rings

def morphology(mask_path,
            min_area=200,
            erode_size=(15,15),
            open_iter=1):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 0).astype(np.uint8)

    # morphological opening to get "bridge" regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erode_size)
    opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    bridge = mask_bin - opened   
    split_mask = (bridge * 255).astype(np.uint8)

    # cut bridges from the original mask
    mask_cut = mask_bin.copy()
    mask_cut[bridge > 0] = 0
    mask_cut = (mask_cut * 255).astype(np.uint8)

    # find external contours on mask_cut
    cnts, _ = cv2.findContours(mask_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_rings = []
    for c in seg_rings:
        if cv2.contourArea(c) >= min_area:
            continue
        if len(c) >= 3:
            seg_rings.append(c.reshape(-1, 2).tolist())

    return seg_rings

