"""
INVINCIBLES - Composite Overview & Comparison Grid Generation
================================================================
Produces two ACTUAL rendered PNG artifacts from real pixel data (not a
client-side CSS overlay):

  1. composite_overview.png
     A single true-pixel composite showing every successfully registered
     sensor layer overlaid on the OHRC base in a distinct color, so an
     evaluator can see at a glance that all sensors truly line up in one
     image, in the OHRC reference frame.

  2. comparison_grid.png
     A contact-sheet image: one row per sensor, each row showing
     [ORIGINAL / UNREGISTERED] next to [REGISTERED / OHRC FRAME], with the
     OHRC reference shown once at the top for visual reference. Sensors
     whose registration failed are shown with their original image and an
     explicit "REGISTRATION FAILED" panel instead of a fabricated result.

Both are generated once per run, from the same warped arrays / masks the
evaluation stage already computed - no re-computation of geometry here.
"""
from __future__ import annotations
import cv2
import numpy as np

# Distinct, readable BGR colors per source sensor for the composite overlay.
SENSOR_OVERLAY_COLOR_BGR = {
    "TMC-Azimuth": (60, 180, 240),   # amber
    "TMC-Slope": (70, 70, 235),      # red
    "IIRS": (140, 210, 80),          # green
    "SAR": (235, 150, 70),           # blue-ish
}

TILE_W = 340
TILE_H = 400
LABEL_H = 26
FOOTER_H = 16
PAD = 10


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY) if gray.shape[-1] == 4 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def build_composite_overview(ref_gray: np.ndarray, registered_layers: dict) -> np.ndarray:
    """
    registered_layers: { sensor: {"warped_gray": ndarray, "mask": ndarray uint8} }
    only sensors that actually succeeded should be passed in.
    Returns a BGR uint8 image, same size as ref_gray.
    """
    base = cv2.cvtColor(ref_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    composite = base.copy()

    for sensor, entry in registered_layers.items():
        color = np.array(SENSOR_OVERLAY_COLOR_BGR.get(sensor, (200, 200, 200)), dtype=np.float32)
        warped = entry["warped_gray"].astype(np.float32)
        mask = (entry["mask"] > 0).astype(np.float32)

        gx = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx ** 2 + gy ** 2)
        edge_norm = edge / (edge.max() + 1e-6)
        alpha = np.clip(edge_norm, 0, 1) * mask * 0.55
        alpha3 = alpha[..., None]

        composite = composite * (1 - alpha3) + color[None, None, :] * alpha3

    return np.clip(composite, 0, 255).astype(np.uint8)


def _labeled_tile(img_bgr: np.ndarray, label: str, sub_label: str = "", failed: bool = False) -> np.ndarray:
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    tile[:] = (10, 12, 14)

    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        scale = min((TILE_W - 2 * PAD) / w, (TILE_H - LABEL_H - FOOTER_H - 2 * PAD) / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = (TILE_W - nw) // 2
        y0 = LABEL_H + (TILE_H - LABEL_H - FOOTER_H - nh) // 2
        tile[y0:y0 + nh, x0:x0 + nw] = resized
        cv2.rectangle(tile, (x0, y0), (x0 + nw - 1, y0 + nh - 1), (60, 65, 72), 1)
    elif failed:
        cv2.putText(tile, "REGISTRATION", (PAD, TILE_H // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 235), 1, cv2.LINE_AA)
        cv2.putText(tile, "FAILED", (PAD, TILE_H // 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 235), 1, cv2.LINE_AA)

    cv2.putText(tile, label, (PAD, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 230, 235), 1, cv2.LINE_AA)
    if sub_label:
        cv2.putText(tile, sub_label, (PAD, TILE_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 145, 150), 1, cv2.LINE_AA)
    return tile


def build_comparison_grid(ref_gray: np.ndarray, rows: list) -> np.ndarray:
    """
    rows: list of dicts, one per source sensor:
      {
        "sensor": str,
        "status": "SUCCESS" | "FAILED",
        "original_gray": ndarray (sensor's own native resolution, unregistered),
        "registered_gray": ndarray or None (OHRC-frame warped image; None if failed),
        "sub_label": str (e.g. "RMSE 4.20px" or failure reason, kept short)
      }
    Returns a single BGR uint8 contact-sheet image.
    """
    ref_bgr = _to_bgr(ref_gray)
    header_tile = _labeled_tile(ref_bgr, "OHRC - REFERENCE FRAME")

    grid_rows = [header_tile]
    for row in rows:
        original_tile = _labeled_tile(_to_bgr(row["original_gray"]), f"{row['sensor']} - ORIGINAL (UNREGISTERED)")
        if row["status"] == "SUCCESS" and row.get("registered_gray") is not None:
            registered_tile = _labeled_tile(_to_bgr(row["registered_gray"]), f"{row['sensor']} - REGISTERED (OHRC FRAME)", row.get("sub_label", ""))
        else:
            registered_tile = _labeled_tile(None, f"{row['sensor']} - REGISTERED (OHRC FRAME)", row.get("sub_label", ""), failed=True)
        row_img = np.hstack([original_tile, registered_tile])
        grid_rows.append(row_img)

    # pad header to match width of the two-tile rows
    if grid_rows:
        target_w = max(r.shape[1] for r in grid_rows)
        padded_rows = []
        for r in grid_rows:
            if r.shape[1] < target_w:
                pad = np.zeros((r.shape[0], target_w - r.shape[1], 3), dtype=np.uint8)
                pad[:] = (10, 12, 14)
                r = np.hstack([r, pad])
            padded_rows.append(r)
        grid_rows = padded_rows

    return np.vstack(grid_rows)
