"""Decode the exported library once and cache the planned feature blocks.

Detector: Ultralytics YOLO11n (COCO), standing in for YOLOX-Nano. Boxes are
aggregated with the plan's schema, so a later script can refit models without
running the detector again. Apple Vision is not available here. Faces use
OpenCV YuNet when the model downloads; text slots stay 0. Live Photo, portrait,
panorama, HDR, and burst stay 0 because this export does not carry them.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm

from photo_features import (
    AUX_NAMES,
    FEATURE_FORMULAS,
    FEATURE_NAMES,
    class_to_category_index,
    face_feature_vector,
    is_screenshot_filename,
    is_whatsapp_filename,
    metadata_vector,
    photometric,
    resize_longest,
)

register_heif_opener()

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/Users/nadavnissim/Downloads/photocleaner-export")
YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


def load_labels(csv_path: Path) -> list[dict[str, str]]:
    import csv

    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def photos_uuid(asset_id: str) -> str:
    """Labels use the bare Photos UUID. The SQLite catalog stores ph://UUID/L0/001."""
    if asset_id.startswith("ph://"):
        asset_id = asset_id[5:]
    return asset_id.split("/")[0]


def load_catalog(db_path: Path) -> dict[str, dict]:
    if not db_path.exists():
        return {}
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT id, filename, width, height, file_size, creation_date,
               has_exif, is_screenshot, is_whatsapp
        FROM asset_catalog
        """
    )
    catalog = {}
    for row in rows:
        catalog[photos_uuid(row[0])] = {
            "filename": row[1] or "",
            "width": row[2] or 0,
            "height": row[3] or 0,
            "file_size": row[4] or 0,
            "creation_date": row[5] if row[5] is not None else np.nan,
            "has_exif": row[6] or 0,
            "is_screenshot": int(row[7] or 0),
            "is_whatsapp": int(row[8] or 0),
        }
    con.close()
    return catalog


def image_has_gps(image: Image.Image) -> bool:
    try:
        exif = image.getexif()
    except Exception:
        return False
    if not exif:
        return False
    try:
        gps = exif.get_ifd(0x8825)
    except Exception:
        return False
    if not gps:
        return False
    return any(key in gps for key in (1, 2, 3, 4))


def load_rgb(path: Path) -> tuple[np.ndarray, bool]:
    image = Image.open(path)
    has_gps = image_has_gps(image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"unexpected image shape {arr.shape}")
    return arr, has_gps


def download_yunet(dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 100_000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in YUNET_URLS:
        try:
            print(f"downloading YuNet from {url}")
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 100_000:
                return True
        except Exception as exc:
            print(f"YuNet download failed: {exc}")
    return False


def load_face_detector(model_path: Path):
    if not download_yunet(model_path):
        print("YuNet unavailable; face features stay 0")
        return None
    if not hasattr(cv2, "FaceDetectorYN"):
        print("this OpenCV build has no FaceDetectorYN; face features stay 0")
        return None
    try:
        return cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.5, 0.3, 50)
    except Exception as exc:
        print(f"YuNet failed to load: {exc}")
        return None


def detect_faces(detector, bgr: np.ndarray) -> np.ndarray | None:
    if detector is None:
        return None
    height, width = bgr.shape[:2]
    even_h = height - (height % 2)
    even_w = width - (width % 2)
    if even_h < 2 or even_w < 2:
        return None
    crop = bgr[:even_h, :even_w]
    try:
        detector.setInputSize((even_w, even_h))
        _, faces = detector.detect(crop)
    except Exception:
        return None
    if faces is None or len(faces) == 0:
        return None
    # YuNet rows: x, y, w, h, landmarks..., score. Score is the last column.
    return np.column_stack([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 3], faces[:, -1]])


def load_yolo(weights_dir: Path):
    import torch
    from ultralytics import YOLO

    weights_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    errors = []
    for name in ("yolo11n.pt", "yolov8n.pt"):
        dest = weights_dir / name
        cwd = os.getcwd()
        try:
            if not dest.exists():
                os.chdir(weights_dir)
                model = YOLO(name)
            else:
                model = YOLO(str(dest))
            os.chdir(cwd)
            names = [model.names[i] for i in range(len(model.names))]
            class_to_cat = class_to_category_index(names)
            print(f"loaded {name} on {device} with {len(names)} classes")
            return model, name, names, class_to_cat, device
        except Exception as exc:
            os.chdir(cwd)
            errors.append(f"{name}: {exc}")
            print(f"could not load {name}: {exc}")
    raise SystemExit("no COCO YOLO model could be loaded\n" + "\n".join(errors))


def predict_batch(model, images: list[np.ndarray], imgsz: int, device: str):
    kwargs = dict(
        source=images,
        imgsz=imgsz,
        conf=0.05,
        iou=0.45,
        max_det=100,
        verbose=False,
        device=device,
    )
    try:
        return model.predict(**kwargs), device
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"{device} predict failed ({exc}); retrying on cpu")
        kwargs["device"] = "cpu"
        return model.predict(**kwargs), "cpu"


def run_detector(model, images: list[np.ndarray], imgsz: int, device: str, batch_size: int = 8):
    per_image: list[np.ndarray] = []
    for start in tqdm(range(0, len(images), batch_size), desc=f"yolo {imgsz}"):
        batch = images[start : start + batch_size]
        results, device = predict_batch(model, batch, imgsz, device)
        if len(results) != len(batch):
            raise RuntimeError(f"detector returned {len(results)} results for {len(batch)} images")
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                per_image.append(np.zeros((0, 6), dtype=np.float32))
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy()
            height, width = result.orig_shape
            xyxy = xyxy.copy()
            xyxy[:, [0, 2]] /= max(width, 1)
            xyxy[:, [1, 3]] /= max(height, 1)
            det = np.column_stack([xyxy, conf, cls]).astype(np.float32)
            per_image.append(det)
    return per_image, device


def pack_detections(per_image: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(rows) for rows in per_image], dtype=np.int32)
    if counts.sum() == 0:
        flat = np.zeros((0, 6), dtype=np.float32)
    else:
        flat = np.concatenate([rows for rows in per_image if len(rows)], axis=0).astype(np.float32)
    return flat, counts


def save_previews(
    images: list[np.ndarray],
    dets: list[np.ndarray],
    ids: list[str],
    labels: np.ndarray,
    screenshots: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    person_score = []
    for det in dets:
        if len(det) == 0:
            person_score.append(0.0)
            continue
        person = det[(det[:, 5].astype(int) == 0) & (det[:, 4] >= 0.25)]
        person_score.append(float(person[:, 4].max()) if len(person) else 0.0)
    person_score = np.array(person_score)
    chosen = []
    for index in np.argsort(-person_score)[:3]:
        chosen.append(int(index))
    for index in np.argsort(-screenshots.astype(float))[:3]:
        if int(index) not in chosen and screenshots[index] > 0:
            chosen.append(int(index))
    zero = [i for i, det in enumerate(dets) if len(det) == 0 or det[:, 4].max() < 0.25]
    for index in zero[:3]:
        if index not in chosen:
            chosen.append(index)

    for index in chosen:
        img = images[index].copy()
        height, width = img.shape[:2]
        det = dets[index]
        if len(det):
            keep = det[det[:, 4] >= 0.25]
            for x1, y1, x2, y2, conf, cls_id in keep:
                p1 = (int(x1 * width), int(y1 * height))
                p2 = (int(x2 * width), int(y2 * height))
                cv2.rectangle(img, p1, p2, (0, 220, 0), 2)
                name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))
                cv2.putText(
                    img,
                    f"{name} {conf:.2f}",
                    (p1[0], max(14, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 0),
                    1,
                    cv2.LINE_AA,
                )
        decision = "delete" if labels[index] == 1 else "keep"
        ss = "ss" if screenshots[index] else "photo"
        path = out_dir / f"{decision}_{ss}_{ids[index][:8]}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print(f"wrote {len(chosen)} previews to {out_dir}")


def class_histogram(dets: list[np.ndarray], class_names: list[str], conf: float = 0.25) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for det in dets:
        if len(det) == 0:
            continue
        hit = set()
        for row in det:
            if row[4] < conf:
                continue
            cls_id = int(row[5])
            if 0 <= cls_id < len(class_names):
                hit.add(class_names[cls_id])
        for name in hit:
            counts[name] += 1
    return {name: n for name, n in counts.items() if n > 0}


def extract(data_dir: Path, out_dir: Path, limit: int, force: bool) -> None:
    cache_path = out_dir / "features.npz"
    meta_path = out_dir / "features_meta.json"
    if cache_path.exists() and meta_path.exists() and not force and limit == 0:
        print(f"cache exists at {cache_path}; pass --force to rebuild")
        return

    labels = load_labels(data_dir / "dataset" / "labels.csv")
    if limit > 0:
        labels = labels[:limit]
    catalog = load_catalog(data_dir / "aiphotocleaner.db")
    dataset_root = (data_dir / "dataset").resolve()

    ids: list[str] = []
    y_list: list[int] = []
    filenames: list[str] = []
    photo_rows = []
    vision_rows = []
    meta_rows = []
    aux_rows = []
    images_bgr: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    n_small = 0

    face_detector = load_face_detector(out_dir / "models" / "face_detection_yunet_2023mar.onnx")
    started = time.time()

    for row in tqdm(labels, desc="decode"):
        asset_id = row["uuid"]
        local = row["local_file"]
        path = (dataset_root / local).resolve()
        if not str(path).startswith(str(dataset_root)) or not path.is_file():
            failures.append({"id": asset_id, "error": f"missing file {local}"})
            continue
        original_name = row.get("original_filename") or ""
        info = catalog.get(asset_id, {})
        catalog_name = info.get("filename") or original_name
        try:
            rgb, has_gps = load_rgb(path)
        except Exception as exc:
            failures.append({"id": asset_id, "error": f"decode: {exc}"})
            continue
        height, width = rgb.shape[:2]
        if width < 2 or height < 2:
            failures.append({"id": asset_id, "error": "image too small"})
            continue

        img640 = resize_longest(rgb, 640)
        if max(height, width) <= 640:
            n_small += 1
        photo, center_var, border_var = photometric(resize_longest(img640, 160))
        bgr = cv2.cvtColor(img640, cv2.COLOR_RGB2BGR)
        faces = detect_faces(face_detector, bgr)
        face_vec, n_faces = face_feature_vector(faces, bgr.shape[1], bgr.shape[0])
        vision = np.zeros(6, dtype=np.float64)
        vision[:4] = face_vec

        filename_ss = is_screenshot_filename(original_name) or is_screenshot_filename(catalog_name)
        filename_wa = is_whatsapp_filename(original_name) or is_whatsapp_filename(catalog_name)
        catalog_ss = int(info.get("is_screenshot") or 0)
        catalog_wa = int(info.get("is_whatsapp") or 0)
        screenshot = bool(catalog_ss) or filename_ss
        whatsapp = bool(catalog_wa) or filename_wa
        aspect = float(width) / float(height)
        meta = metadata_vector(screenshot, whatsapp, aspect, has_gps)

        file_size = float(info.get("file_size") or 0)
        creation = info.get("creation_date", np.nan)
        creation = float(creation) if creation is not None else np.nan
        cat_w = float(info.get("width") or 0)
        cat_h = float(info.get("height") or 0)
        megapixels = (cat_w * cat_h / 1e6) if cat_w and cat_h else (width * height / 1e6)

        aux = np.array(
            [
                file_size,
                megapixels,
                creation,
                float(catalog_ss),
                float(filename_ss),
                float(catalog_wa),
                float(filename_wa),
                float(has_gps),
                float(width),
                float(height),
                center_var,
                border_var,
                float(n_faces),
                float(max(height, width) <= 640),
            ],
            dtype=np.float64,
        )

        ids.append(asset_id)
        y_list.append(int(row["label"]))
        filenames.append(original_name)
        photo_rows.append(photo)
        vision_rows.append(vision)
        meta_rows.append(meta)
        aux_rows.append(aux)
        images_bgr.append(np.ascontiguousarray(bgr))

    if not ids:
        raise SystemExit("no images decoded")

    model, detector_name, class_names, class_to_cat, device = load_yolo(out_dir / "weights")
    dets640, device = run_detector(model, images_bgr, 640, device)
    flat640, counts640 = pack_detections(dets640)
    dets416, device = run_detector(model, images_bgr, 416, device)
    flat416, counts416 = pack_detections(dets416)

    photo = np.stack(photo_rows).astype(np.float32)
    vision = np.stack(vision_rows).astype(np.float32)
    meta = np.stack(meta_rows).astype(np.float32)
    aux = np.stack(aux_rows).astype(np.float64)
    y = np.array(y_list, dtype=np.int8)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        photo=photo,
        vision=vision,
        meta=meta,
        aux=aux,
        y=y,
        det640=flat640,
        det640_counts=counts640,
        det416=flat416,
        det416_counts=counts416,
        class_to_cat=class_to_cat.astype(np.int16),
        ids=np.array(ids),
        filenames=np.array(filenames),
    )
    histogram = class_histogram(dets640, class_names)
    disagreements = int(np.sum((aux[:, 3] > 0) != (aux[:, 4] > 0)))
    payload = {
        "detector": detector_name,
        "detector_note": (
            "Ultralytics COCO nano model used in place of YOLOX-Nano. "
            "Same 80 classes and the same box aggregation. "
            "imgsz 640 is this model's native size; imgsz 416 approximates the plan's YOLOX letterbox."
        ),
        "face_detector": "opencv-yunet-2023mar" if face_detector is not None else "none",
        "text_detector": "none",
        "zero_metadata": ["live_photo", "portrait", "panorama", "hdr", "burst"],
        "n_images": len(ids),
        "n_keep": int((y == 0).sum()),
        "n_delete": int((y == 1).sum()),
        "n_failed": len(failures),
        "failures": failures,
        "n_smaller_than_640": n_small,
        "screenshot_flag_disagreements": disagreements,
        "n_screenshot": int(meta[:, 0].sum()),
        "n_whatsapp": int(meta[:, 1].sum()),
        "n_gps": int(meta[:, 8].sum()),
        "n_with_face": int((aux[:, 12] > 0).sum()),
        "device": device,
        "elapsed_sec": round(time.time() - started, 1),
        "conf_stored_min": 0.05,
        "iou": 0.45,
        "max_det_stored": 100,
        "feature_names": FEATURE_NAMES,
        "aux_names": AUX_NAMES,
        "class_names": class_names,
        "class_image_hits_640_conf0.25": histogram,
        "formulas": FEATURE_FORMULAS,
        "limit": limit,
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    save_previews(
        images_bgr,
        dets640,
        ids,
        y,
        meta[:, 0],
        class_names,
        out_dir / "preview",
    )
    print(
        f"cached {len(ids)} images ({payload['n_keep']} keep / {payload['n_delete']} delete), "
        f"{payload['n_failed']} failed, {payload['n_screenshot']} screenshots, "
        f"{payload['n_gps']} with gps, {payload['n_with_face']} with a YuNet face, "
        f"in {payload['elapsed_sec']}s"
    )
    print("class hits at conf>=0.25:")
    for name, count in sorted(histogram.items(), key=lambda item: -item[1])[:20]:
        print(f"  {name:20s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--limit", type=int, default=0, help="decode only the first N labels")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extract(args.data, args.out, args.limit, args.force)


if __name__ == "__main__":
    main()
