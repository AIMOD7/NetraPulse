import onnx
import onnxruntime as ort
import os
import sys

# Define model path relative to the 'ml' directory
MODEL_PATH = "deploy/models/netrapulse_resnet50.onnx"

def verify_model():
    print(f"Loading ONNX model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at {MODEL_PATH}")
        sys.exit(1)

    try:
        # 1. Check the model's structural integrity
        onnx_model = onnx.load(MODEL_PATH)
        onnx.checker.check_model(onnx_model)
        
        # 2. Test loading into the ONNX Runtime engine
        ort_session = ort.InferenceSession(MODEL_PATH)
        
        # 3. Print the required success message
        print("\n[SUCCESS] ONNX model verified successfully")

    except Exception as e:
        print(f"\n[ERROR] Error verifying ONNX model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_model()
