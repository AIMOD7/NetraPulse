"use client";

import { useCallback, useRef, useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Eye,
  Upload,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ImageIcon,
  Activity,
  Microscope,
  Info,
  Moon,
  Sun,
} from "lucide-react";
import { useTheme } from "next-themes";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface PredictResponse {
  grade: number;
  label: string;
  confidence: number[];
}

interface ExplainResponse {
  grade: number;
  label: string;
  heatmap_png: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const API_BASE = "http://localhost:8000";

const GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

const GRADE_COLORS = [
  "#10b981", // green  — No DR
  "#f59e0b", // amber  — Mild
  "#f97316", // orange — Moderate
  "#ef4444", // red    — Severe
  "#a78bfa", // purple — Proliferative
];

const GRADE_DESCRIPTIONS = [
  "No signs of diabetic retinopathy detected.",
  "Microaneurysms only. Monitor regularly.",
  "More than just microaneurysms. Referral may be needed.",
  "Severe NPDR — high risk of progression. Urgent referral.",
  "New vessels or vitreous/pre-retinal hemorrhage. Urgent treatment.",
];

// ---------------------------------------------------------------------------
// Custom tooltip for bar chart
// ---------------------------------------------------------------------------
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="google-card p-3 text-sm">
        <p className="font-medium text-[var(--text-primary)]">{label}</p>
        <p className="text-[var(--accent)]">{(payload[0].value * 100).toFixed(1)}%</p>
      </div>
    );
  }
  return null;
};

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------
export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // ---- File handling -------------------------------------------------------
  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPEG).");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction(null);
    setHeatmap(null);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  // ---- /predict ------------------------------------------------------------
  const runPredict = async () => {
    if (!selectedFile) return;
    setLoadingPredict(true);
    setError(null);
    setPrediction(null);
    setHeatmap(null);

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error ${res.status}`);
      }
      const data: PredictResponse = await res.json();
      setPrediction(data);
    } catch (err: any) {
      setError(
        err.message?.includes("fetch")
          ? "Could not connect to FastAPI at localhost:8000. Is the server running?"
          : err.message ?? "Unknown error"
      );
    } finally {
      setLoadingPredict(false);
    }
  };

  // ---- /explain (Grad-CAM) -------------------------------------------------
  const runExplain = async () => {
    if (!selectedFile) return;
    setLoadingExplain(true);

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/explain`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error ${res.status}`);
      }
      const data: ExplainResponse = await res.json();
      setHeatmap(data.heatmap_png);
    } catch (err: any) {
      setError(err.message ?? "Grad-CAM failed.");
    } finally {
      setLoadingExplain(false);
    }
  };

  // ---- Chart data ----------------------------------------------------------
  const chartData = prediction
    ? GRADE_LABELS.map((label, i) => ({
        label,
        value: prediction.confidence[i],
        color: GRADE_COLORS[i],
      }))
    : [];

  // ---- Render --------------------------------------------------------------
  return (
    <main className="relative min-h-screen z-10">
      {/* Header */}
      <header className="border-b border-[var(--border)] bg-[var(--bg-card)] px-6 py-3">
        <div className="max-w-6xl mx-auto flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-[var(--accent)]/10 flex items-center justify-center">
            <Eye className="w-5 h-5 text-[var(--accent)]" />
          </div>
          <div>
            <h1 className="text-xl font-medium text-[var(--text-primary)] tracking-tight">NetraPulse</h1>
            <p className="text-xs text-[var(--text-muted)]">Diabetic Retinopathy Screening</p>
          </div>
          <div className="ml-auto flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <Activity className="w-4 h-4" />
            <span className="hidden sm:inline">ResNet50 · ONNX · Grad-CAM</span>
            {mounted && (
              <button
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="ml-4 p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                aria-label="Toggle dark mode"
              >
                {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        {/* Top row: Upload + Preview */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload panel */}
          <div className="google-card p-6 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Upload className="w-5 h-5 text-[var(--text-primary)] opacity-70" />
              <h2 className="font-medium text-[var(--text-primary)] text-lg">Upload Fundus Image</h2>
            </div>

            <div
              id="drop-zone"
              className={`drop-zone p-10 flex flex-col items-center justify-center gap-3 text-center transition-all ${
                dragActive ? "active" : ""
              }`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
            >
              <ImageIcon className="w-10 h-10 text-[var(--text-muted)] opacity-50" />
              <div>
                <p className="text-[var(--text-primary)] opacity-90 font-medium">
                  Drag & drop or click to select
                </p>
                <p className="text-[var(--text-muted)] text-sm mt-1">PNG, JPEG · Fundus photograph</p>
              </div>
              <input
                id="file-input"
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onInputChange}
              />
            </div>

            {selectedFile && (
              <p className="text-sm text-[var(--text-muted)] truncate">
                📎 {selectedFile.name} ({(selectedFile.size / 1024).toFixed(0)} KB)
              </p>
            )}

            <button
              id="analyze-btn"
              onClick={runPredict}
              disabled={!selectedFile || loadingPredict}
              className="w-full py-2.5 rounded-full bg-[var(--accent)] hover:opacity-90 shadow-sm disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium transition-all flex items-center justify-center gap-2"
            >
              {loadingPredict ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  <Microscope className="w-4 h-4" />
                  Analyze Image
                </>
              )}
            </button>

            {error && (
              <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm fade-in">
                <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Preview panel */}
          <div className="google-card p-6">
            <h2 className="font-medium text-[var(--text-primary)] text-lg mb-4 flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-[var(--text-primary)] opacity-70" />
              Image Preview
            </h2>
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Uploaded fundus"
                className="w-full rounded-lg object-cover max-h-64 border border-[var(--border)]"
              />
            ) : (
              <div className="h-48 rounded-lg border border-dashed border-[var(--border)] flex items-center justify-center bg-[var(--bg-primary)]">
                <p className="text-[var(--text-muted)] text-sm">No image selected</p>
              </div>
            )}
          </div>
        </div>

        {/* Results section */}
        {prediction && (
          <div className="space-y-6 fade-in">
            {/* Grade card */}
            <div className="google-card p-6">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-5 h-5 text-green-500" />
                <h2 className="font-medium text-[var(--text-primary)] text-lg">Classification Result</h2>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                <div>
                  <p className="text-[var(--text-muted)] text-xs uppercase tracking-wider mb-2 font-semibold">DR Grade</p>
                  <span className={`grade-badge grade-${prediction.grade}`}>
                    Grade {prediction.grade} — {prediction.label}
                  </span>
                </div>
                <div className="sm:ml-8">
                  <p className="text-[var(--text-muted)] text-xs uppercase tracking-wider mb-1 font-semibold">Clinical note</p>
                  <p className="text-[var(--text-primary)] opacity-90 text-sm">
                    {GRADE_DESCRIPTIONS[prediction.grade]}
                  </p>
                </div>
              </div>
            </div>

            {/* Confidence bar chart */}
            <div className="google-card p-6">
              <h2 className="font-medium text-[var(--text-primary)] text-lg mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-[var(--text-primary)] opacity-70" />
                Confidence Scores
              </h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                  <XAxis
                    dataKey="label"
                    tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(128,128,128,0.1)" }} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {chartData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Grad-CAM section */}
            <div className="google-card p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-medium text-[var(--text-primary)] text-lg flex items-center gap-2">
                  <Eye className="w-5 h-5 text-[var(--text-primary)] opacity-70" />
                  Grad-CAM Explanation
                </h2>
                {!heatmap && (
                  <button
                    id="explain-btn"
                    onClick={runExplain}
                    disabled={loadingExplain}
                    className="px-4 py-2 rounded-full border border-[var(--border)] hover:bg-[var(--bg-primary)] disabled:opacity-40 text-[var(--text-primary)] opacity-90 text-sm font-medium transition-all flex items-center gap-2"
                  >
                    {loadingExplain ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        Generating…
                      </>
                    ) : (
                      "Generate Heatmap"
                    )}
                  </button>
                )}
              </div>

              {heatmap ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 fade-in">
                  <div>
                    <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wider font-semibold">Original</p>
                    <img src={previewUrl!} alt="Original" className="w-full rounded-lg border border-[var(--border)] object-cover" />
                  </div>
                  <div>
                    <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wider font-semibold">Grad-CAM Overlay</p>
                    <img
                      src={`data:image/png;base64,${heatmap}`}
                      alt="Grad-CAM heatmap"
                      className="w-full rounded-lg border border-[var(--border)] object-cover"
                    />
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-2 p-4 rounded-lg bg-[var(--accent)]/10 border border-[var(--accent)]/20 text-[var(--text-primary)] opacity-90 text-sm">
                  <Info className="w-4 h-4 mt-0.5 flex-shrink-0 text-[var(--accent)]" />
                  <span>
                    Click <strong className="text-[var(--text-primary)] font-semibold">Generate Heatmap</strong> to visualize which retinal regions
                    the model focused on. Uses Grad-CAM on ResNet50&apos;s last conv block.
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer disclaimer */}
        <p className="text-center text-xs text-[var(--text-muted)] pb-6 mt-4">
          NetraPulse is a research prototype — not a certified medical device.
          Always confirm findings with a qualified ophthalmologist.
        </p>
      </div>
    </main>
  );
}
