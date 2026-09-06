"""
ml/scripts/verify_onnx.py

Smoke test: loads netrapulse_resnet50.onnx with onnxruntime,
runs a random dummy input through it, and prints the output shape.

This is NOT a real prediction — it only verifies that:
  1. The ONNX file is loadable by onnxruntime (i.e., the MATLAB export worked)
  2. The expected input/output shapes are correct

Usage:
    python ml/scripts/verify_onnx.py
    python ml/scripts/verify_onnx.py --model path/to/custom.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def verify(model_path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("        Run MATLAB training and export first:")
        print("            exportONNXNetwork(net, 'netrapulse_resnet50.onnx')")
        print("        Then move the file to: ml/deploy/models/netrapulse_resnet50.onnx")
        sys.exit(1)

    print(f"[INFO] Loading model: {model_path}")
    session = ort.InferenceSession(str(model_path))

    # Inspect inputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"[INFO] Model inputs:")
    for inp in inputs:
        print(f"       {inp.name}  shape={inp.shape}  dtype={inp.type}")
    print(f"[INFO] Model outputs:")
    for out in outputs:
        print(f"       {out.name}  shape={out.shape}  dtype={out.type}")

    # Build a dummy input matching the expected shape
    # Expected: (batch=1, channels=3, height=512, width=512) — NCHW
    inp_name = inputs[0].name
    inp_shape = inputs[0].shape
    # Replace symbolic dims (None / strings) with concrete values
    concrete_shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp_shape]
    # Make sure spatial dims are 512 (override if symbolic)
    if len(concrete_shape) == 4:
        concrete_shape[0] = 1   # batch
        concrete_shape[1] = 3   # channels
        concrete_shape[2] = 512 # height
        concrete_shape[3] = 512 # width

    dummy = np.random.rand(*concrete_shape).astype(np.float32)
    print(f"\n[INFO] Running dummy input of shape {dummy.shape} ...")
    result = session.run(None, {inp_name: dummy})

    print(f"[OK] Inference succeeded!")
    print(f"     Output shape: {result[0].shape}")
    print(f"     Output values (first 5): {result[0].flatten()[:5]}")
    print(f"\n✅  ONNX model verified successfully — the export worked.")


def main() -> None:
    default_model = (
        Path(__file__).resolve().parents[2]
        / "ml" / "deploy" / "models" / "netrapulse_resnet50.onnx"
    )
    parser = argparse.ArgumentParser(description="Smoke-test the NetraPulse ONNX model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help="Path to the .onnx file (default: ml/deploy/models/netrapulse_resnet50.onnx)",
    )
    args = parser.parse_args()
    verify(args.model)


if __name__ == "__main__":
    main()
