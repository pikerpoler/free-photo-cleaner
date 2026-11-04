"""Feature schema for the offline scoring experiment.

This mirrors the 59-number vector in the feature-scoring plan. Formulas the plan
leaves unspecified are fixed here and documented in FEATURE_FORMULAS so train and
predict stay consistent. Empty slots stay 0.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

FEATURE_DIM = 59

CATEGORY_NAMES = [
    "person",
    "vehicle",
    "outdoor",
    "animal",
    "accessory",
    "sports",
    "kitchen",
    "furniture",
    "electronic",
    "appliance",
    "indoor",
]

# Each COCO class is in exactly one category. Names match Ultralytics COCO.
CATEGORY_MEMBERS: dict[str, list[str]] = {
    "person": ["person"],
    "vehicle": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ],
    "outdoor": [
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
    ],
    "animal": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "sports": [
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    ],
    "kitchen": [
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    ],
    "furniture": ["chair", "couch", "bed", "dining table"],
    "electronic": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "appliance": [
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "hair drier",
    ],
    "indoor": [
        "toilet",
        "potted plant",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "toothbrush",
    ],
}

CANONICAL_COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

CLASS_NAME_ALIASES = {
    "aeroplane": "airplane",
    "motorbike": "motorcycle",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "tvmonitor": "tv",
    "diningtable": "dining table",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "hair dryer": "hair drier",
    "hairdryer": "hair drier",
}

PHOTOMETRIC_NAMES = [
    "mean_luminance",
    "luminance_std",
    "frac_dark_0.05",
    "frac_bright_0.95",
    "laplacian_var",
    "center_border_sharpness_ratio",
    "mean_saturation",
    "saturation_std",
    "colorfulness_hasler",
    "warm_cool_cast",
    "luminance_entropy",
    "hf_noise_mad",
    "lum_mass_q1",
    "lum_mass_q2",
    "lum_mass_q3",
    "lum_mass_q4",
]

GEOM_NAMES = [
    "log_det_count",
    "max_conf",
    "largest_box_area",
    "largest_box_center_dist",
    "frac_boxes_touch_border",
    "sum_box_area_clipped",
]

VISION_NAMES = [
    "log_face_count",
    "max_face_conf",
    "largest_face_area",
    "largest_face_center_dist",
    "log_text_count",
    "text_area_clipped",
]

META_NAMES = [
    "screenshot",
    "whatsapp",
    "live_photo",
    "portrait",
    "panorama",
    "hdr",
    "burst",
    "aspect_ratio",
    "has_gps",
]

AUX_NAMES = [
    "file_size",
    "megapixels",
    "creation_date_ms",
    "catalog_screenshot",
    "filename_screenshot",
    "catalog_whatsapp",
    "filename_whatsapp",
    "exif_gps",
    "oriented_width",
    "oriented_height",
    "center_sharpness",
    "border_sharpness",
    "n_faces",
    "smaller_than_640",
]


def _feature_names() -> list[str]:
    names = list(PHOTOMETRIC_NAMES) + list(GEOM_NAMES)
    for cat in CATEGORY_NAMES:
        names.append(f"{cat}_max_conf")
        names.append(f"{cat}_sum_conf")
    names.extend(VISION_NAMES)
    names.extend(META_NAMES)
    return names


FEATURE_NAMES = _feature_names()

PHOTO_SLICE = slice(0, 16)
GEOM_SLICE = slice(16, 22)
CAT_SLICE = slice(22, 44)
VISION_SLICE = slice(44, 50)
FACE_SLICE = slice(44, 48)
TEXT_SLICE = slice(48, 50)
META_SLICE = slice(50, 59)

FEATURE_FORMULAS = {
    "luminance": "Rec.709: 0.2126 R + 0.7152 G + 0.0722 B on RGB in 0..1",
    "laplacian_var": "variance of cv2.Laplacian ksize=3 on luminance",
    "center_region": "central half of width and height; border is the rest",
    "sharpness_ratio_eps": "center_var / (border_var + 1e-8)",
    "saturation": "HSV saturation (max-min)/max",
    "colorfulness": "Hasler metric on RGB 0..1 (paper uses 0..255; pure scale)",
    "entropy_bins": 32,
    "high_pass": "MAD of luminance minus 3x3 box blur",
    "histogram_quarters": "equal-width bins [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]",
    "log": "natural log",
    "box_distance": "euclidean distance of the largest box center from (0.5, 0.5)",
    "border_touch": "x1<=0.02 or y1<=0.02 or x2>=0.98 or y2>=0.98",
}


def _assert_schema() -> None:
    if len(FEATURE_NAMES) != FEATURE_DIM:
        raise AssertionError(f"expected {FEATURE_DIM} features, got {len(FEATURE_NAMES)}")
    if len(CANONICAL_COCO) != 80:
        raise AssertionError("canonical COCO list is not 80")
    seen: list[str] = []
    for cat in CATEGORY_NAMES:
        seen.extend(CATEGORY_MEMBERS[cat])
    if len(seen) != len(set(seen)):
        raise AssertionError("a COCO class is in more than one category")
    missing = set(CANONICAL_COCO) - set(seen)
    extra = set(seen) - set(CANONICAL_COCO)
    if missing or extra:
        raise AssertionError(f"category map mismatch missing={missing} extra={extra}")
    expected = {
        16: "log_det_count",
        22: "person_max_conf",
        44: "log_face_count",
        48: "log_text_count",
        50: "screenshot",
        58: "has_gps",
    }
    for idx, name in expected.items():
        if FEATURE_NAMES[idx] != name:
            raise AssertionError(f"index {idx} is {FEATURE_NAMES[idx]}, expected {name}")


_assert_schema()


def normalize_class_name(name: str) -> str:
    key = str(name).lower().strip()
    return CLASS_NAME_ALIASES.get(key, key)


def class_to_category_index(class_names: list[str]) -> np.ndarray:
    lookup = {}
    for cat_i, cat in enumerate(CATEGORY_NAMES):
        for member in CATEGORY_MEMBERS[cat]:
            lookup[member] = cat_i
    out = np.full(len(class_names), -1, dtype=np.int16)
    for i, name in enumerate(class_names):
        key = normalize_class_name(name)
        if key not in lookup:
            raise KeyError(f"detector class {name!r} is not in the category map")
        out[i] = lookup[key]
    return out


def is_screenshot_filename(filename: str) -> bool:
    """iOS rule from src/services/media.ts."""
    if not filename:
        return False
    lower = filename.lower()
    if "screenshot" in lower:
        return True
    return re.fullmatch(r"img_\d+\.png", filename, flags=re.IGNORECASE) is not None


def is_whatsapp_filename(filename: str) -> bool:
    """iOS rule from src/services/media.ts."""
    if not filename:
        return False
    lower = filename.lower()
    if "whatsapp" in lower:
        return True
    return re.match(r"img-\d+-wa", lower) is not None


def resize_longest(img: np.ndarray, longest: int) -> np.ndarray:
    """Aspect-fit downscale. Does not upscale."""
    height, width = img.shape[:2]
    long_edge = max(height, width)
    if long_edge <= longest:
        return img
    scale = longest / float(long_edge)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def photometric(img_rgb_uint8: np.ndarray) -> tuple[np.ndarray, float, float]:
    """16 photometric features plus raw center and border Laplacian variance."""
    if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {img_rgb_uint8.shape}")
    rgb = img_rgb_uint8.astype(np.float64) / 255.0
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    lum_f = np.ascontiguousarray(lum.astype(np.float32))

    mean_l = float(lum.mean())
    std_l = float(lum.std())
    frac_dark = float(np.mean(lum < 0.05))
    frac_bright = float(np.mean(lum > 0.95))

    lap = cv2.Laplacian(lum_f, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())
    center_var, border_var = _center_border_var(lap)
    ratio = center_var / (border_var + 1e-8)

    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    sat = np.where(max_c > 1e-6, (max_c - min_c) / np.maximum(max_c, 1e-6), 0.0)
    mean_sat = float(sat.mean())
    std_sat = float(sat.std())

    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    warm_cool = float(red.mean() - blue.mean())

    hist, _ = np.histogram(lum, bins=32, range=(0.0, 1.0))
    total = hist.sum()
    if total <= 0:
        entropy = 0.0
    else:
        p = hist[hist > 0] / total
        entropy = float(-(p * np.log2(p)).sum())

    blur = cv2.blur(lum_f, (3, 3))
    high = lum_f - blur
    med = float(np.median(high))
    noise_mad = float(np.median(np.abs(high - med)))

    q1 = float(np.mean(lum < 0.25))
    q2 = float(np.mean((lum >= 0.25) & (lum < 0.5)))
    q3 = float(np.mean((lum >= 0.5) & (lum < 0.75)))
    q4 = float(np.mean(lum >= 0.75))

    vec = np.array(
        [
            mean_l,
            std_l,
            frac_dark,
            frac_bright,
            lap_var,
            ratio,
            mean_sat,
            std_sat,
            colorfulness,
            warm_cool,
            entropy,
            noise_mad,
            q1,
            q2,
            q3,
            q4,
        ],
        dtype=np.float64,
    )
    return vec, center_var, border_var


def _center_border_var(lap: np.ndarray) -> tuple[float, float]:
    height, width = lap.shape
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    if y1 <= y0 or x1 <= x0:
        var = float(lap.var())
        return var, var
    center = lap[y0:y1, x0:x1]
    mask = np.ones(lap.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    border = lap[mask]
    center_var = float(center.var()) if center.size else 0.0
    border_var = float(border.var()) if border.size else 0.0
    return center_var, border_var


def aggregate_detections(
    dets: np.ndarray,
    class_to_cat: np.ndarray,
    conf_thresh: float = 0.25,
    max_boxes: int = 50,
) -> np.ndarray:
    """Geometry (6) plus max and sum confidence for each category (22).

    dets columns: x1, y1, x2, y2, conf, class id. Boxes are normalized 0..1.
    """
    out = np.zeros(6 + 2 * len(CATEGORY_NAMES), dtype=np.float64)
    if dets is None or len(dets) == 0:
        return out
    dets = np.asarray(dets, dtype=np.float64)
    dets = dets[dets[:, 4] >= conf_thresh]
    if len(dets) == 0:
        return out
    if len(dets) > max_boxes:
        keep = np.argsort(-dets[:, 4])[:max_boxes]
        dets = dets[keep]

    xy = dets[:, :4].copy()
    xy[:, [0, 2]] = np.clip(xy[:, [0, 2]], 0.0, 1.0)
    xy[:, [1, 3]] = np.clip(xy[:, [1, 3]], 0.0, 1.0)
    widths = np.clip(xy[:, 2] - xy[:, 0], 0.0, None)
    heights = np.clip(xy[:, 3] - xy[:, 1], 0.0, None)
    areas = widths * heights
    largest = int(np.argmax(areas))
    cx = (xy[largest, 0] + xy[largest, 2]) * 0.5
    cy = (xy[largest, 1] + xy[largest, 3]) * 0.5
    dist = float(np.hypot(cx - 0.5, cy - 0.5))
    touch = (
        (xy[:, 0] <= 0.02)
        | (xy[:, 1] <= 0.02)
        | (xy[:, 2] >= 0.98)
        | (xy[:, 3] >= 0.98)
    )
    out[0] = float(np.log1p(len(dets)))
    out[1] = float(dets[:, 4].max())
    out[2] = float(areas[largest])
    out[3] = dist
    out[4] = float(touch.mean())
    out[5] = float(min(areas.sum(), 1.0))

    class_ids = dets[:, 5].astype(np.int32)
    cat_ids = class_to_cat[class_ids]
    for cat_i in range(len(CATEGORY_NAMES)):
        sel = dets[cat_ids == cat_i]
        if len(sel) == 0:
            continue
        out[6 + 2 * cat_i] = float(sel[:, 4].max())
        out[6 + 2 * cat_i + 1] = float(sel[:, 4].sum())
    return out


def coco_max_confidence(
    dets: np.ndarray,
    n_classes: int,
    conf_thresh: float = 0.25,
    max_boxes: int = 50,
) -> np.ndarray:
    """Per-class max confidence. Alternative to the 11 category buckets."""
    vec = np.zeros(n_classes, dtype=np.float64)
    if dets is None or len(dets) == 0:
        return vec
    dets = np.asarray(dets, dtype=np.float64)
    dets = dets[dets[:, 4] >= conf_thresh]
    if len(dets) == 0:
        return vec
    if len(dets) > max_boxes:
        dets = dets[np.argsort(-dets[:, 4])[:max_boxes]]
    for row in dets:
        cls_id = int(row[5])
        if 0 <= cls_id < n_classes:
            if row[4] > vec[cls_id]:
                vec[cls_id] = row[4]
    return vec


def face_feature_vector(faces: np.ndarray | None, width: int, height: int) -> tuple[np.ndarray, int]:
    """Four face numbers. `faces` is Nx5: x, y, w, h, score in pixels."""
    vec = np.zeros(4, dtype=np.float64)
    if faces is None or len(faces) == 0 or width <= 0 or height <= 0:
        return vec, 0
    faces = np.asarray(faces, dtype=np.float64)
    areas = (faces[:, 2] / width) * (faces[:, 3] / height)
    areas = np.clip(areas, 0.0, None)
    largest = int(np.argmax(areas))
    cx = (faces[largest, 0] + faces[largest, 2] * 0.5) / width
    cy = (faces[largest, 1] + faces[largest, 3] * 0.5) / height
    vec[0] = float(np.log1p(len(faces)))
    vec[1] = float(faces[:, 4].max())
    vec[2] = float(areas[largest])
    vec[3] = float(np.hypot(cx - 0.5, cy - 0.5))
    return vec, int(len(faces))


def metadata_vector(screenshot: bool, whatsapp: bool, aspect: float, has_gps: bool) -> np.ndarray:
    """Nine metadata slots. Live, portrait, panorama, HDR, and burst are 0 in this export."""
    vec = np.zeros(9, dtype=np.float64)
    vec[0] = 1.0 if screenshot else 0.0
    vec[1] = 1.0 if whatsapp else 0.0
    vec[7] = float(aspect)
    vec[8] = 1.0 if has_gps else 0.0
    return vec


def assemble(photo: np.ndarray, yolo: np.ndarray, vision: np.ndarray, meta: np.ndarray) -> np.ndarray:
    vec = np.concatenate([photo, yolo, vision, meta]).astype(np.float64, copy=False)
    if vec.shape != (FEATURE_DIM,):
        raise AssertionError(f"feature vector has shape {vec.shape}")
    if not np.isfinite(vec).all():
        raise ValueError("non-finite feature value")
    return vec


def iter_detections(flat: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    offset = 0
    for count in counts:
        n = int(count)
        rows.append(flat[offset : offset + n])
        offset += n
    if offset != len(flat):
        raise AssertionError("detection counts do not match the flat array")
    return rows


def matrix_from_parts(
    photo: np.ndarray,
    vision: np.ndarray,
    meta: np.ndarray,
    dets_list: list[np.ndarray],
    class_to_cat: np.ndarray,
    conf_thresh: float,
) -> np.ndarray:
    n = len(photo)
    out = np.empty((n, FEATURE_DIM), dtype=np.float64)
    for i in range(n):
        yolo = aggregate_detections(dets_list[i], class_to_cat, conf_thresh=conf_thresh)
        out[i] = assemble(photo[i], yolo, vision[i], meta[i])
    return out


def _self_check() -> None:
    black = np.zeros((160, 160, 3), dtype=np.uint8)
    photo, _, _ = photometric(black)
    if not np.isclose(photo[0], 0.0) or not np.isclose(photo[2], 1.0):
        raise AssertionError("black image photometric failed")
    if not np.isclose(photo[12:16].sum(), 1.0):
        raise AssertionError("histogram masses must sum to 1")

    white = np.full((80, 80, 3), 255, dtype=np.uint8)
    photo_w, _, _ = photometric(white)
    if not np.isclose(photo_w[0], 1.0) or not np.isclose(photo_w[3], 1.0):
        raise AssertionError("white image photometric failed")
    if not np.isclose(photo_w[15], 1.0):
        raise AssertionError("white image should sit in the brightest quarter")

    gray = np.full((40, 40, 3), 128, dtype=np.uint8)
    photo_g, _, _ = photometric(gray)
    if not np.isclose(photo_g[12:16].sum(), 1.0):
        raise AssertionError("gray histogram masses must sum to 1")

    class_to_cat = np.array([0, 1], dtype=np.int16)
    empty = aggregate_detections(np.zeros((0, 6)), class_to_cat)
    if empty.shape != (28,) or empty.sum() != 0:
        raise AssertionError("empty detections should be zeros")

    one = np.array([[0.1, 0.1, 0.4, 0.5, 0.9, 0.0]])
    vec = aggregate_detections(one, class_to_cat)
    if not np.isclose(vec[0], np.log1p(1)) or not np.isclose(vec[1], 0.9):
        raise AssertionError("single box count/conf failed")
    if not np.isclose(vec[2], 0.3 * 0.4):
        raise AssertionError("single box area failed")
    if not np.isclose(vec[6], 0.9) or not np.isclose(vec[7], 0.9):
        raise AssertionError("person category failed")
    if vec[4] != 0:
        raise AssertionError("this box should not touch the border")

    border = np.array([[0.0, 0.2, 0.3, 0.4, 0.8, 1.0]])
    vec_b = aggregate_detections(border, class_to_cat)
    if not np.isclose(vec_b[4], 1.0):
        raise AssertionError("border touch failed")
    if not np.isclose(vec_b[8], 0.8):
        raise AssertionError("vehicle max conf failed")

    low = np.array([[0.2, 0.2, 0.4, 0.4, 0.2, 0.0]])
    if aggregate_detections(low, class_to_cat, conf_thresh=0.25).sum() != 0:
        raise AssertionError("confidence threshold failed")

    two = np.array(
        [
            [0.1, 0.1, 0.3, 0.3, 0.8, 0.0],
            [0.4, 0.4, 0.6, 0.7, 0.4, 0.0],
        ]
    )
    vec_t = aggregate_detections(two, class_to_cat)
    if not np.isclose(vec_t[0], np.log1p(2)):
        raise AssertionError("two-box count failed")
    if not np.isclose(vec_t[6], 0.8) or not np.isclose(vec_t[7], 1.2):
        raise AssertionError("summed person confidence failed")

    faces = np.array([[10, 20, 30, 40, 0.77]])
    face_vec, n_faces = face_feature_vector(faces, 100, 200)
    if n_faces != 1 or not np.isclose(face_vec[1], 0.77):
        raise AssertionError("face vector failed")
    if not np.isclose(face_vec[2], 0.3 * 0.2):
        raise AssertionError("face area failed")

    meta = metadata_vector(True, False, 0.75, True)
    if meta.shape != (9,) or meta[0] != 1 or meta[1] != 0 or meta[2] != 0 or meta[8] != 1:
        raise AssertionError("metadata vector failed")
    if not is_screenshot_filename("IMG_0610.PNG"):
        raise AssertionError("png screenshot rule failed")
    if is_screenshot_filename("IMG_0109.JPG"):
        raise AssertionError("jpg should not match the png screenshot rule")
    if not is_screenshot_filename("ScreenShot 1.JPG"):
        raise AssertionError("screenshot substring rule failed")
    if not is_whatsapp_filename("IMG-1234-WA005.jpg"):
        raise AssertionError("whatsapp rule failed")

    vision = np.zeros(6)
    vision[:4] = face_vec
    yolo = aggregate_detections(one, class_to_cat)
    assembled = assemble(photo, yolo, vision, meta)
    if assembled.shape != (FEATURE_DIM,):
        raise AssertionError("assemble failed")
    print("photo_features self-check ok")


if __name__ == "__main__":
    _self_check()
