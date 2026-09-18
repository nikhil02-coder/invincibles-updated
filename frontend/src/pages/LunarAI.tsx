import { useEffect, useRef, useState } from "react";
import { useRun } from "../context/RunContext";
import { lunarAIApi } from "../api/lunarAI";
import { registrationApi } from "../api/registration";
import { resolveAssetUrl } from "../api/client";
import { EmptyState, Loader, ErrorState } from "../components/common/Primitives";
import { IconSparkle } from "../components/common/Icons";
import type { LunarAIReport, CompareSensorsResponse, ExplainRegionResponse, RegistrationImagesResponse } from "../types";

interface ChatTurn {
  question: string;
  answer: string;
  engine: string;
}

function classify(text: string): "observed" | "derived" | "uncertain" | "insufficient" {
  const t = text.toUpperCase();
  if (t.includes("UNCERTAIN")) return "uncertain";
  if (t.includes("DERIVED")) return "derived";
  if (t.includes("OBSERVED")) return "observed";
  return "insufficient";
}

export default function LunarAI() {
  const { runId } = useRun();
  const [report, setReport] = useState<LunarAIReport | null>(null);
  const [compare, setCompare] = useState<CompareSensorsResponse | null>(null);
  const [region, setRegion] = useState<ExplainRegionResponse | null>(null);
  const [images, setImages] = useState<RegistrationImagesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState<ChatTurn[]>([]);
  const [asking, setAsking] = useState(false);
  const [marker, setMarker] = useState<{ x: number; y: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    lunarAIApi
      .analyze(runId)
      .then(setReport)
      .catch((e) => setError(e instanceof Error ? e.message : "Lunar AI analysis failed."))
      .finally(() => setLoading(false));
    registrationApi.images(runId).then(setImages).catch(() => {});
  }, [runId]);

  async function handleCompare() {
    if (!runId) return;
    const res = await lunarAIApi.compareSensors(runId);
    setCompare(res);
  }

  async function handleImageClick(e: React.MouseEvent<HTMLImageElement>) {
    if (!runId || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;
    const scaleX = imgRef.current.naturalWidth / rect.width;
    const scaleY = imgRef.current.naturalHeight / rect.height;
    const x = displayX * scaleX;
    const y = displayY * scaleY;
    setMarker({ x: displayX, y: displayY });
    const res = await lunarAIApi.explainRegion(runId, x, y, 26);
    setRegion(res);
  }

  async function handleAsk() {
    if (!runId || !question.trim()) return;
    setAsking(true);
    try {
      const res = await lunarAIApi.ask(runId, question.trim());
      setChat((c) => [...c, { question: question.trim(), answer: res.answer, engine: res.engine }]);
      setQuestion("");
    } catch (e) {
      setChat((c) => [...c, { question: question.trim(), answer: `Error: ${e instanceof Error ? e.message : "request failed"}`, engine: "error" }]);
    } finally {
      setAsking(false);
    }
  }

  if (!runId) {
    return (
      <div className="page">
        <Header />
        <EmptyState title="No registered data to analyze" desc="Run a full analysis first — Lunar AI interprets computed registration results, it does not run independently of them." />
      </div>
    );
  }

  const ohrcImg = images?.["OHRC"]?.multiband_layer_preview || images?.[Object.keys(images || {})[0]]?.reference_analysis_view;

  return (
    <div className="page">
      <Header />

      <div className="quick-action-row">
        <button className="btn btn-secondary btn-sm" onClick={handleCompare}>COMPARE SENSORS</button>
        <button className="btn btn-secondary btn-sm" onClick={() => setQuestion("What is the overall registration confidence and why?")}>
          ASK ABOUT THIS REGION
        </button>
      </div>

      {loading && <Loader label="Lunar AI is interpreting the registered data…" />}
      {error && <ErrorState title="Lunar AI unavailable" message={error} />}

      {report && (
        <div className="grid" style={{ gridTemplateColumns: "1.3fr 1fr", gap: 24 }}>
          <div>
            <div className="ai-card">
              <div className="ai-card-label">Scientific Summary</div>
              {report.scientific_summary.map((line, i) => (
                <div key={i} className="ai-card-body mt-8" style={{ display: "flex", gap: 10 }}>
                  <span className={`evidence-tag ${classify(line)}`}>{classify(line)}</span>
                  <span>{line.replace(/\[.*?\]/g, "").trim()}</span>
                </div>
              ))}
            </div>

            <div className="ai-card">
              <div className="ai-card-label">Sensor Contributions</div>
              {Object.entries(report.sensor_contributions).map(([sensor, c]) => (
                <div key={sensor} className="flex justify-between items-center mt-8" style={{ borderBottom: "1px solid var(--border-hairline)", paddingBottom: 8 }}>
                  <div>
                    <div className="mono" style={{ fontSize: 12.5, fontWeight: 600 }}>{sensor}</div>
                    <div className="text-tertiary" style={{ fontSize: 12 }}>{c.evidence}</div>
                  </div>
                  <span className={`evidence-tag ${classify(c.classification)}`}>{c.classification}</span>
                </div>
              ))}
            </div>

            <div className="ai-card">
              <div className="ai-card-label">Limitations</div>
              {report.limitations.map((l, i) => (
                <div key={i} className="ai-card-body mt-8" style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>• {l}</div>
              ))}
            </div>

            {compare && (
              <div className="ai-card">
                <div className="ai-card-label">Cross-Sensor Chain</div>
                <div className="flex-col gap-8 mt-8">
                  {compare.chain.map((c, i) => (
                    <div key={i} className="flex items-center gap-12">
                      <span className="mono" style={{ fontSize: 12, width: 90 }}>{c.sensor}</span>
                      <span className="text-tertiary" style={{ fontSize: 12 }}>
                        {c.status} {c.confidence ? `· ${c.confidence}` : ""} {c.checkpoint_rmse ? `· ${c.checkpoint_rmse.toFixed(2)}px` : ""}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-12" style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>{compare.cross_sensor_interpretation}</div>
              </div>
            )}
          </div>

          <div>
            <div className="panel panel-padded">
              <div className="panel-title">Region Explorer</div>
              <div className="panel-title-sub mt-4" style={{ marginBottom: 12 }}>Click anywhere on the reference image to interpret that region.</div>
              <div style={{ position: "relative" }}>
                {ohrcImg ? (
                  <img
                    ref={imgRef}
                    src={resolveAssetUrl(ohrcImg) || ""}
                    alt="OHRC reference"
                    onClick={handleImageClick}
                    style={{ width: "100%", borderRadius: "var(--radius-md)", cursor: "crosshair", border: "1px solid var(--border-hairline)" }}
                  />
                ) : (
                  <div className="image-empty">Reference image unavailable</div>
                )}
                {marker && (
                  <div
                    style={{
                      position: "absolute",
                      left: marker.x - 8,
                      top: marker.y - 8,
                      width: 16,
                      height: 16,
                      borderRadius: "50%",
                      border: "2px solid var(--accent-cyan)",
                      boxShadow: "0 0 0 4px rgba(79,209,232,0.15)",
                      pointerEvents: "none",
                    }}
                  />
                )}
              </div>

              {region && (
                <div className="mt-16">
                  <div className="ai-card-label">Observation</div>
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>{region.what_is_observed}</div>
                  <div className="ai-card-label mt-12">Sensor Evidence</div>
                  {region.sensor_contributions.map((c, i) => (
                    <div key={i} className="flex justify-between items-center mt-8">
                      <span className="mono" style={{ fontSize: 11.5 }}>{c.sensor}</span>
                      <span className={`evidence-tag ${classify(c.classification)}`}>{c.classification}</span>
                    </div>
                  ))}
                  <div className="ai-card-label mt-12">Interpretation</div>
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>{region.possible_interpretation}</div>
                  <div className="ai-card-label mt-12">Confidence</div>
                  <div style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>{region.confidence}</div>
                </div>
              )}
            </div>

            <div className="panel panel-padded mt-24">
              <div className="panel-title flex items-center gap-8">
                <IconSparkle style={{ width: 15, height: 15, color: "var(--accent-cyan)" }} /> Ask Lunar AI
              </div>
              <div className="flex-col gap-12 mt-12" style={{ maxHeight: 320, overflowY: "auto" }}>
                {chat.map((turn, i) => (
                  <div key={i}>
                    <div style={{ fontSize: 12.5, color: "var(--text-primary)", fontWeight: 600 }}>{turn.question}</div>
                    <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 4, whiteSpace: "pre-wrap" }}>{turn.answer}</div>
                    <div className="mono text-tertiary" style={{ fontSize: 10, marginTop: 4 }}>via {turn.engine}</div>
                  </div>
                ))}
                {chat.length === 0 && <div className="text-tertiary" style={{ fontSize: 12.5 }}>Ask about confidence, RMSE, sensor failures, or the pipeline itself.</div>}
              </div>
              <div className="chat-input-row">
                <input
                  className="chat-input"
                  placeholder="Ask about this region or run…"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                />
                <button className="btn btn-primary btn-sm" onClick={handleAsk} disabled={asking || !question.trim()}>
                  {asking ? "…" : "ASK"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Header() {
  return (
    <div className="page-header">
      <div>
        <span className="page-eyebrow">Lunar AI</span>
        <h1 className="page-title">Scientific Analysis Assistant</h1>
        <p className="page-subtitle">
          Interprets already-registered, already-validated data. It does not perform geometry, and
          it never claims more than the registration metrics support.
        </p>
      </div>
    </div>
  );
}
