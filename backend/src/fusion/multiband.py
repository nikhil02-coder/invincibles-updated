"""
INVINCIBLES - Final Multi-Sensor Output (Spec Section 25)
============================================================
Individual sensor layers are preserved separately rather than collapsed
into a single blended grayscale image. Because the five sensors differ in
channel count/dtype, a single naive stack is not always safe; a
scientifically safe multi-layer NPZ archive is produced (each layer kept
at its own dtype/valid-mask) plus a human-viewable PNG stack for quick
inspection, together with a metadata JSON describing every layer.
"""
from __future__ import annotations
import os
import json
import numpy as np
import cv2


def save_multiband_output(run_dir: str, reference_key: str, layers: dict):
    """
    layers: { sensor_key: {"image": ndarray, "mask": ndarray, "role": str,
                            "interpolation": str, "dtype": str} }
    Writes:
      - multiband.npz  (all layers, safe multi-dtype archive)
      - layer_<sensor>.png (viewable grayscale preview per layer)
      - layers_metadata.json
    """
    os.makedirs(run_dir, exist_ok=True)
    npz_payload = {}
    metadata = {"reference_sensor": reference_key, "layers": {}}

    for key, entry in layers.items():
        img = entry["image"]
        mask = entry["mask"]
        npz_payload[f"{key}_image"] = img
        npz_payload[f"{key}_mask"] = mask

        preview = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img[..., 0]
        preview_norm = cv2.normalize(preview.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(run_dir, f"layer_{key}.png"), preview_norm)

        valid_fraction = float(np.mean(mask > 0))
        metadata["layers"][key] = {
            "role": entry.get("role"),
            "interpolation": entry.get("interpolation"),
            "dtype": str(img.dtype),
            "shape": list(img.shape),
            "valid_pixel_fraction": valid_fraction,
        }

    np.savez_compressed(os.path.join(run_dir, "multiband.npz"), **npz_payload)
    with open(os.path.join(run_dir, "layers_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    return {
        "npz_path": os.path.join(run_dir, "multiband.npz"),
        "metadata_path": os.path.join(run_dir, "layers_metadata.json"),
        "metadata": metadata,
    }
