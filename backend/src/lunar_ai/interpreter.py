"""
INVINCIBLES - Lunar AI (Spec Sections 36-41)
===============================================
Lunar AI is NOT the registration engine. It never performs geometry - it
only interprets outputs that the registration engine has already
computed. Every statement is tagged OBSERVED / DERIVED / UNCERTAIN, and
phrasing strength is explicitly gated by measured registration confidence
(Section 38). No mineral composition, elevation, latitude/longitude, or
crater-age claims are ever made, because the supplied dataset does not
carry the calibration/geolocation metadata that would support them.

An optional interactive layer lets a scientist ask free-form questions.
When an ANTHROPIC_API_KEY is configured, the question is answered by
Claude - but Claude is given ONLY the actual structured, computed
registration/analysis data as context and is explicitly instructed to
stay within it and to flag anything it cannot support from that data as
uncertain. If no API key is configured, a transparent rule-based answer
engine (pattern matching over the same structured data) is used instead,
so the feature degrades gracefully rather than fabricating a response.
"""
from __future__ import annotations
import os
import re
import json
import numpy as np

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except Exception:
    _ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT = """You are "Lunar AI", the scientific interpretation layer of the INVINCIBLES
lunar multi-sensor image registration prototype. You are speaking to a scientist/evaluator.

STRICT RULES:
1. You may only make claims that are directly supported by the structured registration
   and image-analysis data provided to you in the user message as JSON. Treat that JSON
   as your only source of truth about this dataset and this registration run.
2. Classify every substantive claim as one of: OBSERVED (directly read from the data),
   DERIVED (a reasonable inference/computation from the data), or UNCERTAIN (plausible but
   not supported strongly enough by the data to be confident).
3. NEVER state a mineral composition, geological age, absolute latitude/longitude, or
   elevation value unless such information is explicitly present in the provided JSON -
   it is not, in this dataset, so do not invent it.
4. Explicitly reflect registration confidence in how strongly you phrase cross-sensor
   comparisons: if confidence is LOW, say so plainly and hedge; if HIGH, you may speak
   with more confidence about spatial correspondence claims (but never about semantic/
   compositional claims that the sensors here cannot support).
5. Be concise, technical, and honest. If asked something the data cannot answer, say so
   directly rather than guessing.
6. You may hold a natural conversation about the project, the algorithms used, and the
   results - answer like a knowledgeable colleague, not a canned template - but never
   drift from rules 1-4.
"""


def _safe_get(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def build_scientific_report(sensor_results: dict, comparison_metrics: dict = None) -> dict:
    """
    sensor_results: { sensor_key: { registration metrics dict } } for all
    SOURCE sensors that were processed in this run, as produced by the
    pipeline (real computed values only).
    comparison_metrics: optional { sensor_key: {ssim_before, ssim_after,
        improvement, ...} } from src/evaluation/comparison.py - real,
        measured structural-similarity comparison between the registered
        output and the original unregistered image, restricted to the
        same footprint. When provided, folded into the summary/limitations
        so Lunar AI can speak to "how much did registration actually help"
        with a real number, not an impression.
    Produces the structured Lunar AI output required by Section 37.
    """
    comparison_metrics = comparison_metrics or {}
    summary_lines = []
    detected_features = {}
    sensor_contributions = {}
    cross_sensor = []
    quality_section = {}
    confidence_overall = []
    limitations = [
        "No geolocation (latitude/longitude) metadata accompanies the supplied imagery; "
        "no absolute coordinates are claimed anywhere in this report.",
        "No calibrated reflectance/mineral spectral library was supplied for IIRS; "
        "no mineral composition claims are made.",
        "No independently dated reference is available; no crater-age claims are made.",
    ]

    for sensor, res in sensor_results.items():
        if res.get("status") != "SUCCESS":
            quality_section[sensor] = {
                "status": "FAILED",
                "stage": res.get("failed_stage"),
                "reason": res.get("failure_reason"),
            }
            summary_lines.append(
                f"{sensor}: registration to OHRC FAILED at stage '{res.get('failed_stage')}' "
                f"({res.get('failure_reason')}). No cross-sensor interpretation attempted for {sensor}."
            )
            continue

        conf = res["confidence"]["level"]
        rmse_final = res["metrics"]["checkpoint_rmse_final"]
        inlier_ratio = res["metrics"]["inlier_ratio"]
        n_inliers = res["metrics"]["n_inliers"]
        cmp = comparison_metrics.get(sensor, {})

        detected_features[sensor] = {
            "keypoints_detected": res["metrics"]["n_keypoints"],
            "accepted_matches": res["metrics"]["n_accepted_matches"],
            "inliers": n_inliers,
            "classification": "OBSERVED",
        }

        sensor_contributions[sensor] = {
            "evidence": res.get("sensor_evidence_description", "Spatial structural correspondence with OHRC."),
            "classification": "OBSERVED" if inlier_ratio and inlier_ratio > 0 else "UNCERTAIN",
        }

        if conf == "HIGH":
            phrasing = (f"Registration of {sensor} to the OHRC reference achieved HIGH confidence "
                        f"(checkpoint RMSE {rmse_final:.2f}px, inlier ratio {inlier_ratio:.1%}). "
                        f"The available registration metrics provide stronger support for cross-sensor "
                        f"spatial comparison in this region. [DERIVED]")
        elif conf == "MEDIUM":
            phrasing = (f"Registration of {sensor} to OHRC reached MEDIUM confidence "
                        f"(checkpoint RMSE {rmse_final:.2f}px, inlier ratio {inlier_ratio:.1%}). "
                        f"Cross-sensor spatial comparisons should be treated as indicative rather than "
                        f"precise. [DERIVED]")
        else:
            phrasing = (f"Registration of {sensor} to OHRC reached only LOW confidence "
                        f"(checkpoint RMSE {rmse_final:.2f}px, inlier ratio {inlier_ratio:.1%}). "
                        f"Spatial registration uncertainty limits reliable cross-sensor interpretation "
                        f"for {sensor}; findings below should be treated as UNCERTAIN. [UNCERTAIN]")

        if cmp.get("improvement") is not None:
            imp = cmp["improvement"]
            if imp > 0.02:
                phrasing += (f" Structural similarity (SSIM) to OHRC improved by {imp:+.3f} after registration "
                             f"compared to a naive unregistered resize, measured over the actually-registered "
                             f"footprint - direct pixel-level evidence the estimated transform helped. [OBSERVED]")
            elif imp < -0.02:
                phrasing += (f" SSIM to OHRC did NOT improve after registration ({imp:+.3f} vs. a naive resize) - "
                             f"this suggests either the true geometric offset for {sensor} was already very small, "
                             f"or the estimated transform is dominated by correspondence noise; treat spatial "
                             f"claims for {sensor} cautiously despite the reported confidence level. [UNCERTAIN]")
            else:
                phrasing += f" SSIM change after registration was negligible ({imp:+.3f}). [OBSERVED]"

        summary_lines.append(phrasing)
        cross_sensor.append({
            "sensor": sensor,
            "observation": phrasing,
            "confidence": conf,
            "ssim_before": cmp.get("ssim_before"),
            "ssim_after": cmp.get("ssim_after"),
            "ssim_improvement": cmp.get("improvement"),
        })
        quality_section[sensor] = {
            "status": "SUCCESS",
            "transformation_model": res["metrics"]["transformation_model"],
            "inlier_ratio": inlier_ratio,
            "checkpoint_rmse_final": rmse_final,
            "confidence": conf,
            "ssim_improvement": cmp.get("improvement"),
        }
        confidence_overall.append(conf)

    if confidence_overall:
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        overall = min(confidence_overall, key=lambda c: rank[c])
    else:
        overall = "LOW"

    return {
        "scientific_summary": summary_lines,
        "detected_features": detected_features,
        "regions_of_interest": [],  # populated interactively via explain_region
        "sensor_contributions": sensor_contributions,
        "cross_sensor_observations": cross_sensor,
        "registration_quality": quality_section,
        "evidence": "All figures above are taken directly from the registration engine's computed metrics for this run; none are estimated or hard-coded.",
        "confidence_overall": overall,
        "limitations": limitations,
        "recommended_further_analysis": [
            "Acquire or attach real Chandrayaan-2 OHRC/TMC/IIRS/SAR geolocated products for this "
            "region to enable absolute coordinate reporting.",
            "If IIRS spectral calibration data becomes available, extend Lunar AI with a spectral "
            "unmixing module before attempting any compositional statements.",
            "Increase keypoint budget / relax ratio-test threshold for any sensor that reached only "
            "LOW confidence, then re-run.",
        ],
    }


def explain_region(roi: dict, sensor_results: dict) -> dict:
    """Section 40 - EXPLAIN THIS REGION. roi: {x, y, radius} in OHRC pixel coords."""
    observations = []
    for sensor, res in sensor_results.items():
        if res.get("status") != "SUCCESS":
            observations.append({"sensor": sensor, "evidence": "Not available - registration failed for this sensor.", "classification": "UNCERTAIN"})
            continue
        conf = res["confidence"]["level"]
        observations.append({
            "sensor": sensor,
            "evidence": res.get("sensor_evidence_description", "Structural correspondence present at measured confidence."),
            "registration_confidence": conf,
            "classification": "OBSERVED" if conf in ("HIGH", "MEDIUM") else "UNCERTAIN",
        })
    return {
        "roi": roi,
        "what_is_observed": "A local image neighbourhood centred at the requested pixel location, as rendered by each successfully registered sensor layer. [OBSERVED]",
        "sensor_contributions": observations,
        "morphological_characteristics": "Not automatically classified in this prototype; requires a trained morphological classifier that was out of scope for this run. [UNCERTAIN]",
        "cross_sensor_evidence": [o for o in observations if o.get("classification") == "OBSERVED"],
        "possible_interpretation": "Structural feature consistent across the sensors marked OBSERVED above; no semantic (compositional/geological) interpretation is offered without calibrated spectral/geological reference data. [UNCERTAIN beyond structural correspondence]",
        "confidence": min((o.get("registration_confidence", "LOW") for o in observations), key=lambda c: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(c, 0), default="LOW"),
        "limitations": "No geolocation, spectral calibration, or independent ground truth is available for this ROI.",
    }


def compare_sensors(sensor_results: dict, comparison_metrics: dict = None) -> dict:
    """Section 41 - COMPARE SENSORS, ordered pipeline of registered layers."""
    comparison_metrics = comparison_metrics or {}
    chain = []
    for sensor in ["OHRC", "TMC-Azimuth", "TMC-Slope", "IIRS", "SAR"]:
        res = sensor_results.get(sensor)
        if sensor == "OHRC":
            chain.append({"sensor": "OHRC", "role": "REFERENCE", "status": "LOCKED"})
            continue
        if res is None:
            chain.append({"sensor": sensor, "status": "NOT RUN"})
            continue
        cmp = comparison_metrics.get(sensor, {})
        chain.append({
            "sensor": sensor,
            "status": res.get("status"),
            "confidence": res.get("confidence", {}).get("level") if res.get("status") == "SUCCESS" else None,
            "checkpoint_rmse": res.get("metrics", {}).get("checkpoint_rmse_final") if res.get("status") == "SUCCESS" else None,
            "ssim_improvement": cmp.get("improvement"),
        })
    interpretation = "Cross-sensor spatial comparison is only meaningful for sensors marked SUCCESS above, weighted by their listed confidence."
    successful = [c for c in chain if c.get("status") == "SUCCESS" and c.get("checkpoint_rmse") is not None]
    if successful:
        best = min(successful, key=lambda c: c["checkpoint_rmse"])
        worst = max(successful, key=lambda c: c["checkpoint_rmse"])
        if best["sensor"] != worst["sensor"]:
            interpretation += (f" {best['sensor']} achieved the lowest (best) checkpoint RMSE "
                                f"({best['checkpoint_rmse']:.2f}px); {worst['sensor']} the highest "
                                f"({worst['checkpoint_rmse']:.2f}px).")
    return {"chain": chain, "cross_sensor_interpretation": interpretation}


def ask_lunar_ai(question: str, context: dict) -> dict:
    """
    Interactive free-form Q&A grounded strictly in `context` (the real
    computed dataset/registration/report data for the current run).
    """
    context_json = json.dumps(context, indent=2, default=str)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if _ANTHROPIC_AVAILABLE and api_key:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"CURRENT RUN DATA (JSON):\n{context_json}\n\nQUESTION: {question}"
                }],
            )
            text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
            return {"answer": text, "engine": "claude-sonnet-4-6", "grounded_in": "actual computed run data"}
        except Exception as e:
            fallback = _rule_based_answer(question, context)
            fallback["engine_error"] = f"Claude API call failed ({e}); used rule-based fallback."
            return fallback

    return _rule_based_answer(question, context)


def _rule_based_answer(question: str, context: dict) -> dict:
    """
    A genuinely broad, intent-matching answer engine used whenever no
    ANTHROPIC_API_KEY is configured (or the API call fails). It is not a
    single keyword-to-canned-line lookup: it recognises greetings,
    definitions of every technical term this project actually uses,
    per-sensor detail requests, sensor-vs-sensor comparisons, "which
    sensor is best/worst", methodology/algorithm questions, and the
    registered-vs-original visual-comparison metrics, all answered from
    the real computed data in `context` - never fabricated.
    """
    q = question.lower().strip()
    report = context.get("lunar_ai_report", {})
    metrics = context.get("sensor_results", {})
    comparison = context.get("comparison_metrics", {})
    ALL_SENSORS = ["OHRC", "TMC-Azimuth", "TMC-Slope", "IIRS", "SAR"]

    def engine(ans: str) -> dict:
        return {"answer": ans, "engine": "rule-based-fallback"}

    def sensor_line(s: str) -> str:
        r = metrics.get(s)
        if r is None:
            return f"{s}: not part of this run."
        if r.get("status") != "SUCCESS":
            return f"{s}: FAILED at stage '{r.get('failed_stage')}' ({r.get('failure_reason')})."
        m = r["metrics"]
        cmp = comparison.get(s, {})
        line = (f"{s}: SUCCESS - {m['transformation_model']} model, "
                f"{m['n_inliers']}/{m['n_total_fit']} inliers ({m['inlier_ratio']:.1%}), "
                f"checkpoint RMSE {m['checkpoint_rmse_final']:.2f}px, confidence {r['confidence']['level']}.")
        if cmp.get("improvement") is not None:
            line += f" SSIM change vs. naive resize: {cmp['improvement']:+.3f}."
        return line

    # ---- greetings / small talk ----
    if re.fullmatch(r"(hi|hello|hey|yo|hola)[!. ]*", q):
        return engine("Hello. I'm Lunar AI, the scientific interpretation layer for this INVINCIBLES run. "
                       "Ask me about confidence, RMSE, a specific sensor, how registration compares to the "
                       "original unregistered images, or the algorithms used.")
    if any(w in q for w in ["thank", "thanks", "bye", "goodbye"]):
        return engine("You're welcome. Let me know if you'd like another breakdown of this run's results.")

    # ---- glossary / definitions (checked before generic sensor matching) ----
    GLOSSARY = {
        "rmse": "RMSE (Root Mean Square Error) here is measured in pixels, on INDEPENDENT checkpoint "
                "correspondences that were never used to fit the transformation - it is the honest, "
                "held-out estimate of geometric registration error.",
        "inlier ratio": "Inlier ratio is the fraction of candidate correspondences that remained "
                         "geometrically consistent with the FSC-estimated transformation. A low ratio "
                         "means most raw matches were rejected as outliers before the model was fit.",
        "checkpoint": "Checkpoints are correspondence points deliberately withheld from model fitting "
                      "(Section 22) specifically so RMSE can be measured without the optimistic bias of "
                      "testing a model on the same points used to build it.",
        "phase congruency": "Phase congruency is an illumination-invariant structural feature "
                             "representation (Kovesi-style, log-Gabor based) used here instead of raw "
                             "image gradients, because it stays stable under the sun-angle/radiometric "
                             "differences between OHRC, TMC, IIRS and SAR.",
        "rift": "RIFT (Radiation-variation Insensitive Feature Transform) is the primary multi-modal "
                "descriptor: a log-Gabor Maximum Index Map built to be repeatable across sensors with "
                "very different radiometry. CFOG is used as an automatic fallback if RIFT's response is "
                "too flat to be discriminative on a given image.",
        "cfog": "CFOG (Channel Features of Oriented Gradients) is the fallback descriptor used when "
                "RIFT's log-Gabor orientation response is measured to be too flat/degenerate on the "
                "actual image to reliably discriminate keypoints.",
        "fsc": "FSC (Fast Sample Consensus) is this project's robust outlier-rejection method: guided "
               "sampling weighted by descriptor match quality, plus a spatial-grid fast-consensus "
               "pre-check, which converges faster than plain RANSAC on the low-inlier-ratio "
               "correspondence sets typical of cross-sensor matching.",
        "anms": "ANMS (Adaptive Non-Maximal Suppression) spreads keypoints spatially instead of letting "
                "them cluster on the single strongest edge, which is important for a stable, well-"
                "conditioned transformation fit.",
        "ssim": "SSIM (Structural Similarity Index) is used here to independently check whether "
                "registration actually improved pixel-level alignment: it's measured between OHRC and "
                "the source image both BEFORE (naive resize) and AFTER (actual estimated warp), over "
                "the same registered footprint, so the only thing that changes between the two numbers "
                "is geometric alignment.",
        "ecc": "ECC (Enhanced Correlation Coefficient) is used in the dense area-based fallback "
               "algorithm to align gradient-magnitude maps directly by intensity optimisation, for "
               "sensors (TMC-Azimuth, TMC-Slope) where sparse keypoint descriptors don't find enough "
               "stable correspondences.",
        "mutual information": "Mutual Information similarity search is the second-tier fallback used "
                               "when ECC itself doesn't converge - it directly maximises statistical "
                               "dependence between the two images over a similarity-transform search, "
                               "which works even when the intensity relationship between sensors is "
                               "nonlinear.",
        "homography": "Homography is the most complex transformation model this pipeline can select "
                      "(projective, 8 degrees of freedom) - only chosen if it demonstrably reduces "
                      "residual error over an affine fit by a meaningful margin (Section 19).",
        "affine": "An affine transform (6 degrees of freedom: rotation, scale, shear, translation) is "
                  "chosen over a homography unless the extra projective terms measurably reduce "
                  "residual error - simpler models are preferred unless the evidence justifies more "
                  "complexity.",
        "confidence": "Confidence (HIGH/MEDIUM/LOW) is a rule-based aggregation of inlier ratio, inlier "
                      "count, checkpoint RMSE, spatial coverage, and whether refinement stages actually "
                      "reduced error - never an arbitrary label.",
    }
    for term, definition in GLOSSARY.items():
        if term in q:
            return engine(definition)

    # ---- methodology / algorithm questions ----
    if any(p in q for p in ["how does registration work", "how does this work", "what algorithm",
                             "explain the pipeline", "how is this done", "methodology"]):
        return engine(
            "The pipeline: sensor-aware preprocessing -> multi-scale phase-congruency features -> "
            "keypoint detection + ANMS -> coarse FFT-based alignment -> RIFT (or CFOG fallback) "
            "descriptor matching -> FSC robust outlier rejection -> similarity/affine/homography model "
            "selection -> local terrain-relief refinement -> sub-pixel refinement -> independent "
            "checkpoint RMSE validation. For TMC-Azimuth/TMC-Slope, when sparse matching doesn't find "
            "enough correspondences, a separate dense area-based algorithm (ECC or Mutual-Information "
            "global alignment + Shi-Tomasi/NCC dense correspondences) is used instead - ask me about "
            "'ecc' or 'mutual information' for details."
        )

    # ---- registered-vs-original visual comparison ----
    if any(p in q for p in ["registered image", "original image", "unregistered", "did registration help",
                             "improve", "before and after", "before/after", "visual comparison",
                             "compare the output", "compare the registered"]):
        lines = []
        any_data = False
        for s, cmp in comparison.items():
            if cmp.get("improvement") is None:
                continue
            any_data = True
            verdict = "IMPROVED alignment" if cmp["improvement"] > 0.02 else (
                "did NOT measurably improve alignment" if cmp["improvement"] < -0.02 else "changed alignment negligibly")
            lines.append(f"{s}: SSIM {cmp['ssim_before']:.3f} -> {cmp['ssim_after']:.3f} "
                         f"({cmp['improvement']:+.3f}) - registration {verdict}.")
        if not any_data:
            return engine("No comparison metrics are available for this run yet.")
        return engine("Measured structural-similarity (SSIM) comparison between the registered output and "
                       "the original unregistered image, same footprint, before vs after the estimated "
                       "transform:\n" + "\n".join(lines))

    # ---- failures (checked before "which sensor ..." pattern below, since
    # "which sensors failed" would otherwise match the best/worst intent) ----
    if "fail" in q:
        failed = [s for s, r in metrics.items() if r.get("status") != "SUCCESS"]
        if failed:
            return engine("The following sensors failed registration:\n" +
                           "\n".join(sensor_line(s) for s in failed))
        return engine("No sensor failed registration in this run.")

    # ---- "which sensor is best/worst" ----
    if any(p in q for p in ["best sensor", "worst sensor", "most accurate", "least accurate",
                             "which sensor performed", "which sensor is best", "which sensor is worst"]):
        successful = {s: r["metrics"]["checkpoint_rmse_final"] for s, r in metrics.items()
                      if r.get("status") == "SUCCESS" and r.get("metrics", {}).get("checkpoint_rmse_final") is not None}
        if not successful:
            return engine("No sensor completed registration successfully in this run, so there's no basis for a comparison.")
        best = min(successful, key=successful.get)
        worst = max(successful, key=successful.get)
        return engine(f"By checkpoint RMSE, {best} performed best ({successful[best]:.2f}px) and "
                       f"{worst} performed worst ({successful[worst]:.2f}px) among sensors that "
                       f"completed registration in this run.")

    # ---- compare two named sensors ----
    mentioned = [s for s in ALL_SENSORS if s.lower() in q or s.lower().replace("-", " ") in q or s.lower().replace("-", "") in q]
    if len(mentioned) >= 2:
        return engine("Comparing " + " vs. ".join(mentioned) + ":\n" + "\n".join(sensor_line(s) for s in mentioned))

    # ---- single sensor detail ----
    if len(mentioned) == 1:
        return engine(sensor_line(mentioned[0]))

    # ---- confidence ----
    if "confidence" in q:
        overall = report.get("confidence_overall", "UNKNOWN")
        return engine(f"Overall registration confidence for this run is {overall}, based on the "
                      f"per-sensor inlier ratios and checkpoint RMSE values computed during this run.")

    # ---- rmse / accuracy / error ----
    if "rmse" in q or "accuracy" in q or "error" in q or "precise" in q:
        return engine("Measured checkpoint RMSE per sensor:\n" + "\n".join(sensor_line(s) for s in ALL_SENSORS if s != "OHRC" and s in metrics))

    # ---- overall summary ----
    if any(p in q for p in ["summary", "summarize", "overview", "tell me about this run", "what happened"]):
        lines = [sensor_line(s) for s in ALL_SENSORS if s != "OHRC" and s in metrics]
        return engine(f"Run summary (overall confidence: {report.get('confidence_overall', 'UNKNOWN')}):\n" + "\n".join(lines))

    # ---- limitations ----
    if "limitation" in q or "caveat" in q or "cannot" in q or "can't" in q:
        lims = report.get("limitations", [])
        if lims:
            return engine("Known limitations of this run:\n" + "\n".join(f"- {l}" for l in lims))

    return engine(
        "No ANTHROPIC_API_KEY is configured, so I'm answering with a rule-based engine grounded in this "
        "run's actual computed data (not a language model). I can answer questions about: overall "
        "confidence, RMSE/accuracy, a specific sensor (OHRC, TMC-Azimuth, TMC-Slope, IIRS, SAR), "
        "comparisons between two sensors, which sensor performed best/worst, whether registration "
        "actually improved alignment vs. the original image (SSIM comparison), failures, the "
        "pipeline/algorithms used, or definitions of terms like RMSE, RIFT, FSC, SSIM, ECC, confidence. "
        "Try rephrasing with one of those topics."
    )
