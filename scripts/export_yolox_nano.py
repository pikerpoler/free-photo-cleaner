#!/usr/bin/env python3
"""Export YOLOX-Nano to Core ML with fixed [1, 50, 6] NMS-style output.

Run once on a dev machine (requires torch, onnx, coremltools, yolox):

  pip install torch onnx coremltools
  git clone https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX && pip install -v -e .

  python scripts/export_yolox_nano.py

Output: ios/FreePhotoCleaner/Models/YOLOXNano.mlpackage

The bundled model is Apache 2.0 (Megvii). See YOLOX_APACHE2_NOTICE.txt in the same folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ios" / "FreePhotoCleaner" / "Models" / "YOLOXNano.mlpackage"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    try:
        import coremltools as ct
        import numpy as np
        import torch
    except ImportError as exc:
        print("Missing dependency:", exc, file=sys.stderr)
        print("Install: pip install torch coremltools", file=sys.stderr)
        sys.exit(1)

    try:
        from yolox.exp import get_exp
    except ImportError:
        print(
            "Install YOLOX from https://github.com/Megvii-BaseDetection/YOLOX",
            file=sys.stderr,
        )
        sys.exit(1)

    exp = get_exp(None, "yolox-nano")
    model = exp.get_model()
    model.eval()

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            raw = self.inner(x)
            return postprocess_fixed(raw, num_classes=80, max_boxes=50)

    wrapped = Wrapped(model)
    dummy = torch.randn(1, 3, 416, 416)
    traced = torch.jit.trace(wrapped, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, 416, 416))],
        minimum_deployment_target=ct.target.iOS15,
    )
    mlmodel.author = "Megvii YOLOX-Nano (Apache-2.0)"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print("Wrote", args.out)


def postprocess_fixed(raw, num_classes: int, max_boxes: int):
    """Placeholder: real export should decode YOLOX head + NMS in TorchScript.

    Replace with the project's NMS when wiring the full YOLOX export pipeline.
    """
    del raw, num_classes
    import torch

    pad = torch.full((1, max_boxes, 6), -1.0)
    return pad


if __name__ == "__main__":
    main()
