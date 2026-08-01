from pathlib import Path

from PIL import Image

from src.inference import load_model_bundle, predict_image

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "best_EfficientNetB0_FastKAN.pth"

bundle = load_model_bundle(ARTIFACT_PATH)

# A plain gray placeholder image just verifies the pipeline runs end to end
# (correct shapes, valid probabilities). It does not test real accuracy —
# use test_samples from the training notebook for that.
dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

result = predict_image(dummy_image, bundle=bundle)

probs_sum = sum(result["probabilities"].values())
assert 0.99 <= probs_sum <= 1.01, f"Probabilities do not sum to 1: {probs_sum}"
assert result["top_class"] in bundle["class_names"]

print("SMOKE TEST PASSED")
print(result["summary"])
