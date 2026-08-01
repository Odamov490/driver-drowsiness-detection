from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# =========================================================
# Inference contract
# =========================================================

CLASS_NAMES = ["notdrowsy", "sleepyCombination", "slowBlinkWithNodding", "yawning"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
MODEL_VERSION = "EfficientNetB0_FastKAN_v1"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must stay identical to the Colab training notebook's eval_transform.
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# =========================================================
# Model architecture — must match the training notebook exactly,
# including attribute names, so the saved state_dict keys line up.
# =========================================================


class FastKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 8, sigma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        grid = torch.linspace(-1, 1, grid_size)
        self.register_buffer("grid", grid)
        self.base = nn.Linear(input_dim, output_dim)
        self.spline_weight = nn.Parameter(torch.randn(output_dim, input_dim, grid_size) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_expanded = x.unsqueeze(-1)
        rbf = torch.exp(-((x_expanded - self.grid) ** 2) / (2 * self.sigma ** 2))
        spline_out = torch.einsum("big,oig->bo", rbf, self.spline_weight)
        return base_out + spline_out


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: list[int],
        grid_size: int = 8,
        sigma: float = 0.5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                FastKANLayer(input_dim=in_dim, output_dim=out_dim, grid_size=grid_size, sigma=sigma)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.silu(x)
            x = self.dropout(x)
        return self.layers[-1](x)


class EfficientNetB0_FastKAN(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280

        self.fastkan = FastKAN(
            layers_hidden=[self.feature_dim, 512, 128, num_classes],
            grid_size=8,
            sigma=0.5,
            dropout=0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fastkan(x)
        return x


# =========================================================
# Public API used by app.py and smoke_test.py
# =========================================================


@lru_cache(maxsize=1)
def load_model_bundle(artifact_path: str | Path) -> dict[str, Any]:
    """Load the saved model weights once for the app process."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact was not found at '{path}'. "
            "Confirm that artifacts/best_EfficientNetB0_FastKAN.pth is committed "
            "to the repository (see README.md for upload instructions)."
        )

    model = EfficientNetB0_FastKAN(num_classes=NUM_CLASSES, pretrained=False).to(_DEVICE)

    checkpoint = torch.load(path, map_location=_DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return {
        "model": model,
        "class_names": CLASS_NAMES,
        "model_version": MODEL_VERSION,
    }


def validate_image(image: Image.Image) -> Image.Image:
    """Basic validation for one uploaded image."""
    if image is None:
        raise ValueError("No image was provided.")
    if image.size[0] < 10 or image.size[1] < 10:
        raise ValueError("Image is too small to analyze.")
    return image.convert("RGB")


def predict_image(image: Image.Image, *, bundle: dict[str, Any]) -> dict[str, Any]:
    """Run inference for one uploaded image using the saved model."""
    img = validate_image(image)
    model = bundle["model"]
    class_names = bundle["class_names"]

    input_tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    top_class = max(probabilities, key=probabilities.get)
    top_confidence = probabilities[top_class]

    label = "Drowsy" if top_class != "notdrowsy" else "Not drowsy"

    return {
        "label": label,
        "top_class": top_class,
        "confidence": top_confidence,
        "probabilities": probabilities,
        "summary": f"{label} — predicted class: {top_class} ({top_confidence:.1%}).",
        "model_version": bundle["model_version"],
    }
