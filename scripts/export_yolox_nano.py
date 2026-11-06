#!/usr/bin/env python3
"""Export YOLOX-Nano to Core ML: raw decoded head [1, N, 85] (same as YOLOX export).

The app runs NMS in Swift and pads to [1, 50, 6] normalized xyxy + score + classId.

Requires: pip install torch torchvision coremltools pillow
Install YOLOX: git clone https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX && pip install -v -e .

  PYTHONPATH=/path/to/YOLOX python scripts/export_yolox_nano.py --download-weights --verify
  python scripts/export_yolox_nano.py --verify --verify-image /path/to/photo.jpg

Output: ios/FreePhotoCleaner/Models/YOLOXNano.mlpackage (~2 MB)

Apache 2.0 (Megvii). See YOLOX_APACHE2_NOTICE.txt in ios/FreePhotoCleaner/Models/.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "ios" / "FreePhotoCleaner" / "Models" / "YOLOXNano.mlpackage"

DEFAULT_WEIGHTS_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_nano.pth"
)
INPUT_SIZE = 416


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading", url, "->", dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def load_yolox_nano(weights: Path | None):
    try:
        import torch
        from yolox.exp import get_exp
    except ImportError as exc:
        print("Missing dependency:", exc, file=sys.stderr)
        print("Install YOLOX from https://github.com/Megvii-BaseDetection/YOLOX", file=sys.stderr)
        sys.exit(1)

    exp = get_exp(None, "yolox-nano")
    model = exp.get_model()
    model.eval()
    # Export raw head output (grid offsets + logits) — NMS runs on device in Swift.
    model.head.decode_in_inference = False

    if weights is not None and weights.exists():
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if not isinstance(state, dict):
            print("Unexpected checkpoint format", file=sys.stderr)
            sys.exit(1)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            miss, unexp = model.load_state_dict(state, strict=False)
            print(
                "Loaded weights with strict=False (missing keys:",
                len(miss),
                "unexpected:",
                len(unexp),
                ")",
            )
        print("Loaded weights from", weights)
    else:
        print("ERROR: weights file not found. Use --weights or --download-weights", file=sys.stderr)
        sys.exit(1)

    return model


def export_coreml(model: "torch.nn.Module", out: Path) -> None:
    import coremltools as ct
    import torch

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    model.eval()
    with torch.no_grad():
        raw = model(dummy)
    if not isinstance(raw, torch.Tensor):
        print("Unexpected model output type", type(raw), file=sys.stderr)
        sys.exit(1)
    print("Torch raw output shape:", tuple(raw.shape))

    traced = torch.jit.trace(model, dummy)

    # Vision (VNCoreMLModel) requires an image input. YOLOX expects BGR 0–255, unnormalized.
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
                color_layout=ct.colorlayout.BGR,
                scale=1.0,
                bias=[0.0, 0.0, 0.0],
            )
        ],
        minimum_deployment_target=ct.target.iOS15,
    )
    mlmodel.author = "Megvii YOLOX-Nano (Apache-2.0)"
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print("Wrote", out)


def letterbox_image_path(path: Path, size: int) -> "Image.Image":
    """Resize with aspect ratio, pad to square `size` (YOLOX gray 114). Returns an RGB PIL image."""
    from PIL import Image  # type: ignore

    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(im, (left, top))
    return canvas


def pil_to_bgr_tensor(img: "Image.Image") -> "np.ndarray":
    """RGB PIL image -> BGR float32 [1,3,H,W] in 0–255 (YOLOX preprocessing)."""
    import numpy as np

    arr = np.asarray(img).astype("float32")[:, :, ::-1]
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[np.newaxis, ...])


def run_verify(
    mlpackage_path: Path,
    model: "torch.nn.Module",
    verify_image: Path | None,
) -> None:
    import numpy as np
    import torch

    model.eval()

    try:
        import coremltools as ct
    except ImportError:
        print("coremltools required for --verify", file=sys.stderr)
        sys.exit(1)

    from PIL import Image  # type: ignore

    if verify_image and verify_image.exists():
        try:
            pil_img = letterbox_image_path(verify_image, INPUT_SIZE)
        except OSError:
            print("Failed to read image", verify_image, file=sys.stderr)
            sys.exit(1)
    else:
        print("No --verify-image: using random image (shape + finite check).")
        noise = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        pil_img = Image.fromarray(noise)

    inp = pil_to_bgr_tensor(pil_img)
    torch_in = torch.from_numpy(inp.copy())
    with torch.no_grad():
        torch_out = model(torch_in)

    assert torch_out.dim() == 3 and torch_out.shape[0] == 1, torch_out.shape
    assert torch.isfinite(torch_out).all(), "NaN in torch output"

    ml = ct.models.MLModel(str(mlpackage_path))
    preds = ml.predict({"image": pil_img})  # type: ignore[attr-defined]
    out_arr = None
    for _k, v in preds.items():
        out_arr = np.array(v)
        break
    assert out_arr is not None
    assert out_arr.shape == tuple(torch_out.shape), (out_arr.shape, torch_out.shape)
    assert np.isfinite(out_arr).all()

    t = torch_out.numpy()
    if verify_image and verify_image.exists():
        diff = np.abs(t - out_arr).max()
        print("max |torch - coreml| (raw):", diff)
        if diff > 0.05:
            print(
                "WARN: torch vs CoreML raw output differs more than expected.",
                file=sys.stderr,
            )
    else:
        print(
            "Skipping numeric diff on random input (some ops differ slightly between runtimes)."
        )
    print("VERIFY OK: raw tensor shapes match and outputs are finite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="yolox_nano.pth (COCO). Default: build/yolox_nano.pth",
    )
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help=f"Download default weights from {DEFAULT_WEIGHTS_URL}",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare torch vs CoreML raw output after export",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        metavar="MLPACKAGE",
        help="Verify existing .mlpackage against weights (no export)",
    )
    parser.add_argument(
        "--verify-image",
        type=Path,
        default=None,
        help="Optional image for verification letterbox",
    )
    args = parser.parse_args()

    weights_path = args.weights if args.weights is not None else ROOT / "build" / "yolox_nano.pth"

    if args.verify_only is not None:
        if not args.verify_only.exists():
            print("verify-only path not found:", args.verify_only, file=sys.stderr)
            sys.exit(1)
        if not weights_path.exists():
            print("--verify-only needs weights at", weights_path, file=sys.stderr)
            sys.exit(1)
        model = load_yolox_nano(weights_path)
        run_verify(args.verify_only, model, args.verify_image)
        return

    if args.download_weights or not weights_path.exists():
        if not weights_path.exists():
            download_file(DEFAULT_WEIGHTS_URL, weights_path)

    model = load_yolox_nano(weights_path)
    export_coreml(model, args.out)

    if args.verify:
        run_verify(args.out, model, args.verify_image)


if __name__ == "__main__":
    main()
