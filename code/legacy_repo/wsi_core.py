"""Thumbnail filtering and training-free Kaiko patch selection for the WSI demo."""

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from safetensors.torch import load_file


def tissue_mask(overview):
    hsv = cv2.cvtColor(overview, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 250)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def patch_grid(width, height, overview_shape, mask, labels, patch_size):
    thumb_h, thumb_w = overview_shape[:2]
    candidates = []
    for row in range(math.ceil(height / patch_size)):
        for col in range(math.ceil(width / patch_size)):
            x = min(col * patch_size, width - patch_size)
            y = min(row * patch_size, height - patch_size)
            x1 = max(0, round(x * thumb_w / width))
            y1 = max(0, round(y * thumb_h / height))
            x2 = min(thumb_w, max(x1 + 1, round((x + patch_size) * thumb_w / width)))
            y2 = min(thumb_h, max(y1 + 1, round((y + patch_size) * thumb_h / height)))
            patch_mask = mask[y1:y2, x1:x2]
            positive_labels = labels[y1:y2, x1:x2][patch_mask > 0]
            component = (int(np.bincount(positive_labels).argmax())
                         if len(positive_labels) else 0)
            candidates.append({"row": row, "col": col, "x": x, "y": y,
                               "tissue_fraction": round(float(patch_mask.mean()), 4),
                               "component": component})
    return candidates


def spatial_shortlist(candidates, components, per_fragment=8, min_tissue=0.5):
    """Seed each tissue fragment with its richest patch, then spread spatially."""
    selected = []
    for component in components:
        options = [item for item in candidates if item["component"] == component
                   and item["tissue_fraction"] >= min_tissue]
        if len(options) < per_fragment:
            raise ValueError(f"Fragment {component} has only {len(options)} eligible positions")
        picked = [max(options, key=lambda item: item["tissue_fraction"])]
        while len(picked) < per_fragment:
            remaining = [item for item in options if item not in picked]
            def value(item):
                distance = min(math.hypot(item["x"] - old["x"],
                                          item["y"] - old["y"]) for old in picked)
                return (distance, item["tissue_fraction"], -item["row"], -item["col"])
            picked.append(max(remaining, key=value))
        selected.extend(picked)
    return selected


def load_pathology_encoder(weights_path):
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Kaiko weights missing: {weights_path}. Run download_assets.py --kaiko")
    model = timm.create_model("vit_small_patch16_224", pretrained=False,
                              num_classes=0, global_pool="token")
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    return model.eval(), weights_path


def encode(model, patches):
    # Match Kaiko's model card: resize to 224 and normalize RGB to [-1, 1].
    tensors = []
    for rgb in patches:
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        tensors.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()
                       .div_(127.5).sub_(1.0))
    features = []
    with torch.inference_mode():
        for batch in torch.stack(tensors).split(4):
            features.append(model(batch))
    return torch.nn.functional.normalize(torch.cat(features), dim=1).numpy()


def select_diverse_one_per_fragment(features, groups):
    """Choose one per fragment to maximize summed pairwise cosine distance."""
    similarity = features @ features.T
    best, best_score = None, -float("inf")
    for group in itertools.product(*groups):
        score = sum(1.0 - similarity[a, b]
                    for a, b in itertools.combinations(group, 2))
        if score > best_score:
            best, best_score = group, score
    return list(best), float(best_score)
