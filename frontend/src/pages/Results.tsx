import { useEffect, useState } from "react";
import { useRun } from "../context/RunContext";
import { registrationApi } from "../api/registration";
import { resolveAssetUrl } from "../api/client";
import { MetricCard, ConfidencePill, StatusPill, ImageFrame, EmptyState, Loader, ErrorState } from "../components/common/Primitives";
import { CompareSlider } from "../components/common/CompareSlider";
import type { RegistrationImagesResponse } from "../types";
import { formatNumber, formatPercent, formatPixels } from "../utils/format";

export default function Results() {
  const { runId, summary } = useRun();
  const [images, setImages] = useState<RegistrationImagesResponse | null>(null);
  const [sensor, setSensor] = useState("");

  useEffect(() => {
    if (!runId) return;
    registrationApi
      .images(runId)
      .then((imgs) => {
        setImages(imgs);
        const first = Object.keys(imgs).find((k) => k !== "OHRC");
        if (first) setSensor(first);
      })
      .catch(() => {});
  }, [runId]);

  if (!runId) {
    return (
      <div className="page">
        <Header />
        <EmptyState title="No results yet" desc="Run a full analysis from Mission Control to see registration results here." />
      </div>
    );
  }
  if (!summary) {
    return (
      <div className="page">
        <Header />
        <Loader label="Loading registration results…" />
      </div>
    );
  }

  const sensorKeys = Object.keys(summary.sensor_results);
  const res = summary.sensor_results[sensor];
  const set = images?.[sensor];

  return (
    <div className="page">
      <Header />

      <div className="tabs">
        {sensorKeys.map((s) => {
          const r = summary.sensor_results[s];
          return (
            <div key={s} className={`tab${sensor === s ? " active" : ""}`} onClick={() => setSensor(s)}>
              {s} {r.status === "FAILED" ? "⚠" : ""}
            </div>
          );
        })}
      </div>

      {!res && <ErrorState title="Sensor not found" message="Select a sensor tab above." />}

      {res?.status === "FAILED" && (
        <div className="panel panel-padded">
          <div className="panel-header">
            <div className="panel-title">{sensor} → OHRC</div>
            <StatusPill tone="error">FAILED</StatusPill>
          </div>
          <p style={{ fontSize: 13.5 }}>
            Registration was not accepted for this sensor. The pipeline stopped rather than
            produce an unreliable result.
          </p>
          <div className="state-code mt-16">
            Stage: {(res as { failed_stage?: string }).failed_stage}
            {"\n"}Reason: {(res as { failure_reason?: string }).failure_reason}
          </div>
        </div>
      )}

      {res?.status === "SUCCESS" && (
        <>
          <div className="flex justify-between items-center mt-8" style={{ marginBottom: 16 }}>
            <div className="section-title" style={{ marginBottom: 0 }}>{sensor} → OHRC</div>
            <ConfidencePill level={res.confidence.level} />
          </div>

          <div className="grid grid-4">
            <MetricCard label="Detected Features" value={formatNumber(res.metrics.n_keypoints)} tooltip="Total keypoints detected across reference and source (phase-congruency based, multiscale)." />
            <MetricCard label="Accepted Correspondences" value={formatNumber(res.metrics.n_accepted_matches)} tooltip="Correspondences that passed ratio test + mutual cross-check (or the area-based NCC fallback)." />
            <MetricCard label="Inliers" value={`${formatNumber(res.metrics.n_inliers)} / ${formatNumber(res.metrics.n_total_fit)}`} tooltip="Correspondences consistent with the FSC-estimated transformation." />
            <MetricCard label="Inlier Ratio" value={formatPercent(res.metrics.inlier_ratio)} />
          </div>

          <div className="grid grid-4 mt-16">
            <MetricCard label="Checkpoint RMSE (before)" value={formatPixels(res.metrics.checkpoint_rmse_before_refinement)} />
            <MetricCard label="Checkpoint RMSE (local refined)" value={formatPixels(res.metrics.checkpoint_rmse_after_local_refinement)} />
            <MetricCard
              label="Final RMSE"
              value={formatPixels(res.metrics.checkpoint_rmse_final)}
              tone={res.confidence.level === "HIGH" ? "success" : res.confidence.level === "MEDIUM" ? "warning" : "error"}
              tooltip="Root Mean Square Error measured on independent checkpoint correspondences never used to fit the transform."
            />
            <MetricCard label="Transformation Model" value={res.metrics.transformation_model.toUpperCase()} sub={res.metrics.model_selection_justification.slice(0, 60) + "…"} />
          </div>

          <div className="grid grid-2 mt-24">
            <div className="panel panel-padded">
              <div className="panel-title">Confidence Factors</div>
              <div className="factor-list mt-12">
                {res.confidence.factors.map((f, i) => (
                  <div key={i} className="factor-item">
                    <span className={`flag ${f[0] === "PASS" ? "pass" : "weak"}`}>{f[0]}</span>
                    <span>{f[1]}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="panel panel-padded">
              <div className="panel-title">Correspondence Method</div>
              <p className="mt-12" style={{ fontSize: 13 }}>
                <strong style={{ color: "var(--text-primary)" }}>Descriptor: </strong>
                {res.descriptor_method}
              </p>
              {res.descriptor_deviation_reason && (
                <p className="mt-8" style={{ fontSize: 12.5 }}>{res.descriptor_deviation_reason}</p>
              )}
              {res.area_based_fallback && (
                <p className="mt-8" style={{ fontSize: 12.5 }}>
                  <strong style={{ color: "var(--text-primary)" }}>Area-based fallback: </strong>
                  {res.area_based_fallback.notes.split("|")[0]}
                </p>
              )}
              <p className="mt-8" style={{ fontSize: 12.5 }}>
                <strong style={{ color: "var(--text-primary)" }}>Local refinement applied: </strong>
                {res.local_refinement.used ? "Yes" : "No"} (residual CoV {res.local_refinement.residual_coefficient_of_variation.toFixed(3)})
              </p>
            </div>
          </div>

          <div className="panel panel-padded mt-24">
            <div className="panel-title">Checkerboard Validation</div>
            <div className="panel-title-sub mt-4" style={{ marginBottom: 14 }}>
              A continuous, unbroken boundary across tile edges is the visual signature of correct
              spatial alignment.
            </div>
            <ImageFrame src={resolveAssetUrl(set?.checkerboard)} label={`OHRC vs ${sensor} (REGISTERED)`} empty="Checkerboard not available." aspect="0.85 / 1" />
          </div>

          <div className="panel panel-padded mt-24">
            <div className="panel-title">Before / After Registration</div>
            {set?.source_analysis_view && set?.warped_registered ? (
              <div className="mt-12">
                <CompareSlider
                  beforeSrc={resolveAssetUrl(set.source_analysis_view) || ""}
                  afterSrc={resolveAssetUrl(set.warped_registered) || ""}
                  beforeLabel="BEFORE (SOURCE FRAME)"
                  afterLabel="AFTER (OHRC FRAME)"
                />
              </div>
            ) : (
              <ImageFrame src={null} empty="Before/after comparison not available." />
            )}
          </div>
        </>
      )}
    </div>
  );
}

function Header() {
  return (
    <div className="page-header">
      <div>
        <span className="page-eyebrow">Results</span>
        <h1 className="page-title">Registration Result</h1>
        <p className="page-subtitle">Quantitative, independently validated accuracy for every source sensor.</p>
      </div>
    </div>
  );
}
