import os
import sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("ANTHROPIC_API_KEY", None)  # force rule-based fallback for deterministic tests

from src.lunar_ai import interpreter as lunar_ai
from src.fusion import composite
from src.evaluation import comparison as comparison_mod


def _fake_success(model="affine", rmse=3.5, inlier_ratio=0.6, n_inliers=20, n_total=33, confidence="MEDIUM"):
    return {
        "status": "SUCCESS",
        "sensor_evidence_description": "Structural information.",
        "metrics": {
            "n_keypoints": 1200, "n_keypoints_reference": 600, "n_keypoints_source": 600,
            "n_candidate_matches": 50, "n_accepted_matches": 40,
            "n_inliers": n_inliers, "n_total_fit": n_total, "inlier_ratio": inlier_ratio,
            "transformation_model": model, "checkpoint_rmse_final": rmse,
        },
        "confidence": {"level": confidence, "score": 3, "max_score": 5, "factors": [("PASS", "ok")]},
    }


def _fake_failure(stage="FEATURE_MATCHING", reason="Insufficient correspondences"):
    return {"status": "FAILED", "failed_stage": stage, "failure_reason": reason}


SENSOR_RESULTS = {
    "TMC-Azimuth": _fake_success(rmse=8.7, inlier_ratio=0.65, confidence="MEDIUM"),
    "TMC-Slope": _fake_failure(),
    "IIRS": _fake_success(rmse=4.2, inlier_ratio=0.94, confidence="MEDIUM"),
    "SAR": _fake_success(rmse=4.7, inlier_ratio=0.88, confidence="MEDIUM"),
}

COMPARISON_METRICS = {
    "TMC-Azimuth": {"ssim_before": 0.37, "ssim_after": 0.34, "improvement": -0.03, "valid_coverage": 0.94},
    "TMC-Slope": {"ssim_before": None, "ssim_after": None, "improvement": None, "valid_coverage": 0.0, "note": "Registration failed."},
    "IIRS": {"ssim_before": 0.73, "ssim_after": 0.91, "improvement": 0.18, "valid_coverage": 0.95},
    "SAR": {"ssim_before": 0.68, "ssim_after": 0.81, "improvement": 0.13, "valid_coverage": 0.99},
}


def test_scientific_report_includes_ssim_comparison_language():
    report = lunar_ai.build_scientific_report(SENSOR_RESULTS, COMPARISON_METRICS)
    iirs_lines = [l for l in report["scientific_summary"] if l.startswith("Registration of IIRS")]
    assert iirs_lines and "improved" in iirs_lines[0].lower()
    tmc_lines = [l for l in report["scientific_summary"] if l.startswith("Registration of TMC-Azimuth")]
    assert tmc_lines and "did not" in tmc_lines[0].lower()


def test_scientific_report_never_claims_geolocation_or_composition():
    report = lunar_ai.build_scientific_report(SENSOR_RESULTS, COMPARISON_METRICS)
    full_text = " ".join(report["scientific_summary"]).lower() + " ".join(report["limitations"]).lower()
    assert "latitude" not in full_text or "no geolocation" in full_text
    assert "mineral composition" not in full_text.replace("no mineral composition", "")


def test_compare_sensors_identifies_best_and_worst():
    result = lunar_ai.compare_sensors(SENSOR_RESULTS, COMPARISON_METRICS)
    assert "IIRS" in result["cross_sensor_interpretation"]  # best (lowest RMSE)
    statuses = {c["sensor"]: c["status"] for c in result["chain"]}
    assert statuses["TMC-Slope"] == "FAILED"
    assert statuses["OHRC"] == "LOCKED"


def _ask(question):
    context = {
        "sensor_results": SENSOR_RESULTS,
        "lunar_ai_report": lunar_ai.build_scientific_report(SENSOR_RESULTS, COMPARISON_METRICS),
        "comparison_metrics": COMPARISON_METRICS,
    }
    return lunar_ai.ask_lunar_ai(question, context)


def test_ask_lunar_ai_uses_rule_based_fallback_without_api_key():
    res = _ask("What is the overall confidence?")
    assert res["engine"] == "rule-based-fallback"
    assert "MEDIUM" in res["answer"] or "LOW" in res["answer"] or "HIGH" in res["answer"]


def test_ask_lunar_ai_answers_definitions():
    res = _ask("What does RMSE mean?")
    assert "checkpoint" in res["answer"].lower()

    res2 = _ask("Explain FSC")
    assert "consensus" in res2["answer"].lower() or "outlier" in res2["answer"].lower()


def test_ask_lunar_ai_answers_best_worst_sensor():
    res = _ask("Which sensor performed best?")
    assert "IIRS" in res["answer"]


def test_ask_lunar_ai_answers_sensor_comparison():
    res = _ask("Compare IIRS and SAR")
    assert "IIRS" in res["answer"] and "SAR" in res["answer"]


def test_ask_lunar_ai_answers_visual_comparison_question():
    res = _ask("Did registration actually improve alignment compared to the original image?")
    assert "SSIM" in res["answer"]
    assert "IIRS" in res["answer"]


def test_ask_lunar_ai_answers_failures():
    res = _ask("Which sensors failed?")
    assert "TMC-Slope" in res["answer"]


def test_ask_lunar_ai_handles_unrecognized_question_gracefully():
    res = _ask("asdkjaslkdj random gibberish")
    assert res["engine"] == "rule-based-fallback"
    assert len(res["answer"]) > 20


# ---------------------------------------------------------------------------
# Composite image generation
# ---------------------------------------------------------------------------
def _synthetic_gray(size=200, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.uniform(50, 200, (size, size))).astype(np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, 255, -1)
    return img


def test_composite_overview_matches_reference_shape():
    ref = _synthetic_gray(200, seed=1)
    src = _synthetic_gray(200, seed=2)
    mask = np.full((200, 200), 255, dtype=np.uint8)
    layers = {"IIRS": {"warped_gray": src, "mask": mask}}
    result = composite.build_composite_overview(ref, layers)
    assert result.shape == (200, 200, 3)
    assert result.dtype == np.uint8


def test_comparison_grid_has_one_row_per_sensor_plus_header():
    ref = _synthetic_gray(120, seed=3)
    rows = [
        {"sensor": "IIRS", "status": "SUCCESS", "original_gray": _synthetic_gray(100, 4),
         "registered_gray": _synthetic_gray(120, 5), "sub_label": "RMSE 4.2px"},
        {"sensor": "TMC-Slope", "status": "FAILED", "original_gray": _synthetic_gray(90, 6),
         "registered_gray": None, "sub_label": "FEATURE_MATCHING"},
    ]
    grid = composite.build_comparison_grid(ref, rows)
    assert grid.shape[0] == composite.TILE_H * 3  # header + 2 sensor rows
    assert grid.shape[1] == composite.TILE_W * 2


# ---------------------------------------------------------------------------
# SSIM-based alignment comparison
# ---------------------------------------------------------------------------
def test_alignment_comparison_detects_improvement_after_correction():
    base = _synthetic_gray(150, seed=7)
    ref = base.copy()
    # source is a shifted version of the same content (simulating misalignment)
    shifted = np.roll(base, 15, axis=1)
    # "warped" simulates a corrected version much closer to the reference
    corrected = base.copy()
    mask = np.full((150, 150), 255, dtype=np.uint8)

    result = comparison_mod.compute_alignment_comparison(ref, shifted, corrected, mask)
    assert result["improvement"] is not None
    assert result["ssim_after"] > result["ssim_before"]


def test_alignment_comparison_handles_low_coverage_honestly():
    ref = _synthetic_gray(100, seed=8)
    src = _synthetic_gray(100, seed=9)
    tiny_mask = np.zeros((100, 100), dtype=np.uint8)
    tiny_mask[:2, :2] = 255
    result = comparison_mod.compute_alignment_comparison(ref, src, src, tiny_mask)
    assert result["ssim_before"] is None
    assert "too small" in result["note"]
