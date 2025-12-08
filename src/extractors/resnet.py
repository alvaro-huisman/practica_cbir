import os
from functools import lru_cache
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_device = torch.device("cpu")


@lru_cache(maxsize=1)
def _load_model() -> torch.nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(_device)
    return model


@lru_cache(maxsize=1)
def _preprocess() -> transforms.Compose:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    return weights.transforms()


def extract_resnet50(image: Union[Image.Image, str]) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        pil_img = Image.open(image).convert("RGB")

    preprocess = _preprocess()
    tensor = preprocess(pil_img).unsqueeze(0).to(_device)

    model = _load_model()
    with torch.no_grad():
        features = model(tensor)

    vector = features.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    vector /= norm
    return vector


__all__ = ["extract_resnet50"]
