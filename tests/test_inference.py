from pathlib import Path

import pytest
from PIL import Image

from src.inference import CLASS_NAMES, load_model_bundle, predict_image, validate_image

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT / "artifacts" / "best_EfficientNetB0_FastKAN.pth"


def test_validate_image_rejects_none():
    with pytest.raises(ValueError):
        validate_image(None)


def test_validate_image_accepts_rgb():
    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    out = validate_image(img)
    assert out.mode == "RGB"


@pytest.mark.skipif(not ARTIFACT_PATH.is_file(), reason="Model artifact not present locally")
def test_predict_image_returns_known_classes():
    bundle = load_model_bundle(ARTIFACT_PATH)
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = predict_image(img, bundle=bundle)

    assert result["top_class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)
