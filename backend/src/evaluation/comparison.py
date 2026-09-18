"""
INVINCIBLES - Registered-vs-Original Comparison Metrics
==========================================================
Answers, with an actual measured number rather than a visual impression,
the question every evaluator asks: "how much did registration actually
improve alignment?"

For each source sensor we compute the Structural Similarity Index (SSIM -
Wang et al. 2004) between the OHRC reference and the source image in two
states:

  BEFORE: the source's ORIGINAL image, naively resampled to OHRC's pixel
          grid with a simple resize (no rotation/translation correction) -
          this is what a naive "just stack the images" approach would give.
  AFTER:  the source image warped into the OHRC frame using the actual
          estimated registration transform.

SSIM on the raw sensor pair is not expected to be high in absolute terms
(different sensors have entirely different radiometry), so we do not
report SSIM as "accuracy" - we report the RELATIVE IMPROVEMENT
(after - before), which is meaningful even when both absolute values are
low, because both are measured under the exact same radiometric
mismatch and differ only in geometric alignment. A positive improvement
is real evidence the estimated transform moved the source image closer to
true spatial agreement with OHRC; a non-positive value is reported
honestly, not hidden.
"""
from __future__ import annotations
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def _prep(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)


def compute_alignment_comparison(ref_gray: np.ndarray, src_gray_original: np.ndarray,
                                  warped_gray: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    ref_gray: OHRC grayscale (H, W)
    src_gray_original: the SOURCE sensor's own original grayscale (its own resolution)
    warped_gray: the source image warped into OHRC's frame by the actual estimated transform
    valid_mask: uint8 mask (0/255) of pixels the warp actually populated
    """
    h, w = ref_gray.shape[:2]
    naive_resize = cv2.resize(src_gray_original, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = valid_mask > 0
    coverage = float(np.mean(mask))
    if coverage < 0.05:
        return {
            "ssim_before": None,
            "ssim_after": None,
            "improvement": None,
            "valid_coverage": coverage,
            "note": "Valid registered coverage too small (<5% of frame) for a meaningful comparison.",
        }

    ref_p = _prep(ref_gray)
    naive_p = _prep(naive_resize)
    warped_p = _prep(warped_gray)

    # Full-frame SSIM for the naive (unregistered) case
    ssim_before = float(ssim(ref_p, naive_p, data_range=255.0))

    # SSIM restricted to the actually-valid registered region for a fair
    # apples-to-apples comparison (padding/no-data regions would otherwise
    # penalise the AFTER case for pixels that were never claimed to be
    # registered in the first place).
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    ref_crop = ref_p[y0:y1, x0:x1]
    warped_crop = warped_p[y0:y1, x0:x1]
    naive_crop = naive_p[y0:y1, x0:x1]

    ssim_after = float(ssim(ref_crop, warped_crop, data_range=255.0))
    ssim_before_same_region = float(ssim(ref_crop, naive_crop, data_range=255.0))

    improvement = ssim_after - ssim_before_same_region

    return {
        "ssim_before": ssim_before_same_region,
        "ssim_after": ssim_after,
        "improvement": improvement,
        "valid_coverage": coverage,
        "note": (
            f"SSIM measured within the actually-registered region only, "
            f"so both figures reflect the same geographic footprint."
        ),
    }
