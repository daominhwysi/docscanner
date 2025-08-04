import cv2
import requests
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from io import BytesIO 
import concurrent.futures
from functools import lru_cache
import base64
from dotenv import load_dotenv
load_dotenv()
import os
from collections import OrderedDict
# Load model

# --------- CACHE CHO COLORS VÀ CONSTANTS ----------
COLORS = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

@lru_cache(maxsize=None)
def get_text_size_cached(text: str, font_scale: float, thickness: int):
    return cv2.getTextSize(text, FONT, font_scale, thickness)


def calculate_max_font_scale(text: str, max_width: int, max_height: int,
                             font_face=cv2.FONT_HERSHEY_SIMPLEX,
                             initial_scale: float = 1.0,
                             scale_step: float = 0.5,
                             min_scale_step: float = 0.01,
                             padding: int = 2) -> Tuple[float, int]:
    def fits(s: float) -> bool:
        th = max(1, int(s * 2))
        (w_text, h_text), baseline = cv2.getTextSize(text, font_face, s, th)
        total_h = h_text + baseline
        return (w_text + 2 * padding) <= max_width and (total_h + 2 * padding) <= max_height

    scale = initial_scale
    if not fits(scale):
        while scale > min_scale_step and not fits(scale):
            scale /= 2
    else:
        while fits(scale):
            scale += scale_step
        scale -= scale_step

    step = scale_step
    while step >= min_scale_step:
        if fits(scale + step):
            scale += step
        else:
            step /= 2

    thickness = max(1, int(scale * 2))
    return scale, thickness


def draw_boxes(
    img: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    class_names: Optional[List[str]] = None,
    base_box_thickness: int = 2,
    skip_class_ids: Optional[List[int]] = None,
    skip_class_names: Optional[List[str]] = None,
    draw_labels: bool = True,
    sort_by_coordinate: bool = True,
    row_threshold: int = 10
):
    if boxes is None or len(boxes) == 0:
        return img.copy(), {}

    h_img, w_img = img.shape[:2]
    box_thickness = max(1, int(base_box_thickness * min(w_img, h_img) / 1000))
    out = img.copy()
    skip_ids = set(skip_class_ids or [])
    skip_names = set(skip_class_names or [])

    bboxes = boxes.astype(np.int32)
    valid_indices = []
    for i, bbox in enumerate(bboxes):
        cls = int(labels[i])
        if cls in skip_ids:
            continue
        name = (class_names[cls] if class_names and cls < len(class_names) else str(cls))
        if name in skip_names:
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        valid_indices.append(i)

    # Sort theo y-then-x
    if sort_by_coordinate and valid_indices:
        entries = [(i, *bboxes[i][:4]) for i in valid_indices]
        entries.sort(key=lambda e: e[2])  # sort theo y1
        rows = []
        for i, x1, y1, x2, y2 in entries:
            placed = False
            for row in rows:
                y_min, y_max, lst = row
                if y1 <= y_max + row_threshold and y2 >= y_min - row_threshold:
                    row[0] = min(y_min, y1);
                    row[1] = max(y_max, y2)
                    lst.append((i, x1))
                    placed = True
                    break
            if not placed:
                rows.append([y1, y2, [(i, x1)]])
        rows.sort(key=lambda r: r[0])
        sorted_indices = []
        for _, _, group in rows:
            group.sort(key=lambda e: e[1])
            sorted_indices += [i for i, _ in group]
    else:
        sorted_indices = valid_indices

    # Tạo dict cho ROI
    cropped_objects_np = OrderedDict()

    for order_idx, i in enumerate(sorted_indices):
        x1, y1, x2, y2 = bboxes[i]
        cls = int(labels[i])
        name = (class_names[cls] if class_names and cls < len(class_names) else str(cls))
        prefix = name[:2].upper() if len(name) >= 2 else name.upper()
        key = f"{prefix}{order_idx + 1}"

        # Vẽ khung
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)

        if draw_labels:
            font_scale, font_thickness = calculate_max_font_scale(
                key, x2-x1, y2-y1, font_face=FONT
            )
            (w_text, h_text), baseline = get_text_size_cached(key, font_scale, font_thickness)
            text_x = np.clip(x1 + (x2-x1 - w_text)//2, x1+2, x2 - w_text - 2)
            text_y = np.clip(y1 + h_text + 2, y1 + h_text + 2, y2 - baseline - 2)

            # ========== Layer 1: NỀN ==========
            background_overlay = out.copy()
            cv2.rectangle(background_overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            alpha_background = 0.4  # nền mờ nhẹ
            cv2.addWeighted(background_overlay, alpha_background, out, 1 - alpha_background, 0, out)

            # ========== Layer 2: CHỮ ==========
            text_overlay = out.copy()
            cv2.putText(text_overlay, key, (text_x, text_y), FONT, font_scale, (0, 0, 255), font_thickness)
            alpha_text = 0.75  # chữ gần như rõ ràng
            cv2.addWeighted(text_overlay, alpha_text, out, 1 - alpha_text, 0, out)



        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            cropped_objects_np[key] = roi.copy()

    return out, cropped_objects_np