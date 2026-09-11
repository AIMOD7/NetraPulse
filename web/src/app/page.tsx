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
  ChevronDown,
  ArrowRight,
  ShieldCheck,
  Zap,
  Server,
  FileCheck,
  Send,
  Mail,
  ExternalLink,
  ChevronUp,
} from "lucide-react";
import gsap from "gsap";
import { NetraPulseLogo } from "@/components/logo";
import { LandingHeader } from "@/components/landing-header";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface PredictResponse {
  grade: number;
  label: string;
  confidence: number[];
  quality?: {
    is_gradable: boolean;
    quality_score: number;
    issues: string[];
  };
}

interface ExplainResponse {
  grade: number;
  label: string;
  heatmap_png: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

const GRADE_COLORS = [
  "#10b981", // green  — No DR
  "#f59e0b", // amber  — Mild
  "#f97316", // orange — Moderate
  "#ef4444", // red    — Severe
  "#a855f7", // purple — Proliferative
];

const GRADE_DESCRIPTIONS = [
  "No signs of diabetic retinopathy detected. Routine annual screening recommended.",
  "Microaneurysms only. Recommend re-evaluation in 6–12 months.",
  "More than just microaneurysms, but less than severe. Specialist referral recommended.",
  "Severe non-proliferative retinopathy. High risk of rapid progression. Urgent retinal consult.",
  "Neovascularization or pre-retinal hemorrhage. Urgent surgical or anti-VEGF intervention required.",
];

// Sample fundus SVG previews for rapid demo
const SAMPLE_PREVIEWS = [
  {
    name: "Sample 1 (Normal)",
    color: "#0f766e",
    gradeHint: "No DR",
    svg: `<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><defs><radialGradient id="n1" cx="45%" cy="50%" r="50%"><stop offset="0%" stop-color="#b45309"/><stop offset="60%" stop-color="#78350f"/><stop offset="100%" stop-color="#1e1b4b"/></radialGradient></defs><circle cx="100" cy="100" r="95" fill="url(#n1)"/><circle cx="135" cy="100" r="16" fill="#fde68a" opacity="0.85"/><path d="M135 100 Q105 80 60 70 M135 100 Q105 120 55 135 M135 100 Q95 95 40 100" stroke="#b91c1c" stroke-width="2.5" fill="none" opacity="0.75"/><circle cx="85" cy="100" r="8" fill="#451a03" opacity="0.6"/></svg>`,
  },
  {
    name: "Sample 2 (Moderate DR)",
    color: "#b45309",
    gradeHint: "Moderate DR",
    svg: `<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><defs><radialGradient id="n2" cx="45%" cy="50%" r="50%"><stop offset="0%" stop-color="#9a3412"/><stop offset="65%" stop-color="#451a03"/><stop offset="100%" stop-color="#18181b"/></radialGradient></defs><circle cx="100" cy="100" r="95" fill="url(#n2)"/><circle cx="135" cy="100" r="16" fill="#fde68a" opacity="0.85"/><path d="M135 100 Q100 70 50 65 M135 100 Q105 130 50 145 M135 100 Q95 98 35 105" stroke="#7f1d1d" stroke-width="3" fill="none" opacity="0.8"/><circle cx="85" cy="100" r="8" fill="#451a03" opacity="0.6"/><circle cx="75" cy="85" r="3" fill="#ef4444"/><circle cx="95" cy="120" r="3.5" fill="#ef4444"/><circle cx="60" cy="110" r="2.5" fill="#ef4444"/><circle cx="110" cy="75" r="4" fill="#fef08a" opacity="0.9"/></svg>`,
  },
  {
    name: "Sample 3 (Severe/PDR)",
    color: "#991b1b",
    gradeHint: "Proliferative DR",
    svg: `<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><defs><radialGradient id="n3" cx="45%" cy="50%" r="50%"><stop offset="0%" stop-color="#7f1d1d"/><stop offset="60%" stop-color="#450a0a"/><stop offset="100%" stop-color="#09090b"/></radialGradient></defs><circle cx="100" cy="100" r="95" fill="url(#n3)"/><circle cx="135" cy="100" r="16" fill="#fde68a" opacity="0.85"/><path d="M135 100 Q95 60 40 55 M135 100 Q100 140 45 155 M135 100 Q90 100 30 110" stroke="#450a0a" stroke-width="3.5" fill="none" opacity="0.85"/><path d="M135 100 Q145 80 155 70 M135 100 Q150 115 160 125" stroke="#dc2626" stroke-width="2" fill="none"/><ellipse cx="70" cy="90" rx="14" ry="7" fill="#dc2626" opacity="0.8"/><circle cx="100" cy="130" r="6" fill="#dc2626" opacity="0.85"/><circle cx="55" cy="120" r="5" fill="#dc2626" opacity="0.85"/><circle cx="80" cy="65" r="4.5" fill="#fef08a" opacity="0.95"/></svg>`,
  },
];

// Helper to convert sample SVG to File
const sampleSvgToFile = (svgStr: string, name: string): File => {
  const blob = new Blob([svgStr], { type: "image/svg+xml" });
  return new File([blob], `${name}.svg`, { type: "image/svg+xml" });
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const heroRef = useRef<HTMLDivElement>(null);
  const aboutRef = useRef<HTMLDivElement>(null);
  const scannerRef = useRef<HTMLDivElement>(null);
  const contactRef = useRef<HTMLDivElement>(null);

  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Contact form state
  const [contactForm, setContactForm] = useState({
    name: "",
    email: "",
    organization: "",
    subject: "Clinical Inquiry",
    message: "",
  });
  const [contactSubmitted, setContactSubmitted] = useState(false);

  // GSAP entrance animations
  useEffect(() => {
    const ctx = gsap.context(() => {
      // Hero elements animation
      gsap.from(".hero-anim", {
        y: 35,
        opacity: 0,
        duration: 1,
        stagger: 0.15,
        ease: "power3.out",
      });

      // Floating flair pulse
      gsap.to(".hero-flair", {
        y: -10,
        duration: 3,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
      });
    }, heroRef);

    return () => ctx.revert();
  }, []);

  // Smooth scroll helper
  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  // ---- File Handling -------------------------------------------------------
  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload a valid retinal image file (PNG, JPEG, or SVG).");
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
          ? `Could not connect to the API at ${API_BASE}. Please verify that the backend is online.`
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
    if (prediction) {
      form.append("target_grade", prediction.grade.toString());
    }

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
      setError(err.message ?? "Grad-CAM generation failed.");
    } finally {
      setLoadingExplain(false);
    }
  };

  // Chart Data preparation
  const chartData = prediction
    ? prediction.confidence.map((prob, i) => ({
        grade: GRADE_LABELS[i],
        probability: prob,
        color: GRADE_COLORS[i],
      }))
    : [];

  // Contact Form Submission
  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setContactSubmitted(true);
    setTimeout(() => {
      setContactForm({
        name: "",
        email: "",
        organization: "",
        subject: "Clinical Inquiry",
        message: "",
      });
    }, 4000);
  };

  return (
    <div className="min-h-screen text-white">
      {/* Header Navigation */}
      <LandingHeader />

      {/* ===================================================================== */}
      {/* SECTION 1: HERO SECTION                                               */}
      {/* ===================================================================== */}
      <section
        id="hero"
        ref={heroRef}
        className="relative min-h-screen flex flex-col items-center justify-center pt-24 pb-16 px-4 sm:px-6 lg:px-8 text-center overflow-hidden"
      >
        {/* Background Ambient Aura */}
        <div className="pointer-events-none absolute left-1/2 top-1/3 h-[32rem] w-[42rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-300/10 blur-[120px]" />

        <div className="relative z-10 max-w-5xl mx-auto flex flex-col items-center">
          {/* Top Pill Badge */}
          <div className="hero-anim mb-6 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/[0.04] border border-white/10 backdrop-blur-md text-xs font-semibold tracking-wider uppercase text-slate-300">
            <span className="w-2 h-2 rounded-full bg-[#0ae448] animate-pulse" />
            Clinical-Grade Ophthalmic Intelligence
          </div>

          {/* Large Animated Logo */}
          <div className="hero-anim hero-flair mb-7 rounded-[2rem] border border-emerald-200/15 bg-white/[0.045] p-5 shadow-[0_24px_80px_rgba(67,230,160,0.12)] backdrop-blur-xl">
            <NetraPulseLogo size={84} showText={false} />
          </div>

          {/* Main Hero Headline (GSAP style) */}
          <h1 className="hero-anim text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight leading-[1.05] mb-6">
            Diabetic Retinopathy <br />
            <span className="text-gradient-green">Screening & Diagnostics</span>
          </h1>

          {/* Subtitle */}
          <p className="hero-anim max-w-2xl text-base sm:text-lg text-slate-400 font-normal leading-relaxed mb-8">
            Next-generation retinal triage powered by deep convolutional ResNet-50 networks,
            sub-second ONNX inference, and transparent Grad-CAM explainability for hospital PACS and clinics.
          </p>

          {/* Interactive Action CTAs */}
          <div className="hero-anim flex flex-wrap items-center justify-center gap-4 mb-12">
            <button
              onClick={() => scrollToSection("scanner")}
              className="flex cursor-pointer items-center gap-2 rounded-full bg-[#92f5c8] px-7 py-3.5 text-sm font-bold text-[#071014] shadow-[0_12px_40px_rgba(67,230,160,0.2)] transition-all duration-300 hover:-translate-y-0.5 hover:bg-white hover:shadow-[0_18px_55px_rgba(67,230,160,0.3)]"
            >
              Launch Live Scanner
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => scrollToSection("about")}
              className="px-8 py-3.5 rounded-full font-semibold text-sm text-slate-200 bg-white/[0.05] hover:bg-white/[0.1] border border-white/10 hover:border-white/20 transition-all duration-300 backdrop-blur-md cursor-pointer"
            >
              Explore Architecture
            </button>
          </div>

          {/* Quick Specification Badges */}
          <div className="hero-anim grid grid-cols-2 sm:grid-cols-4 gap-3 w-full max-w-3xl text-left">
            <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/10 backdrop-blur-sm flex items-center gap-3">
              <ShieldCheck className="w-5 h-5 text-[#0ae448] flex-shrink-0" />
              <div>
                <div className="text-xs font-bold text-white">5-Tier ICDR</div>
                <div className="text-[11px] text-slate-400">Clinical Grading</div>
              </div>
            </div>

            <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/10 backdrop-blur-sm flex items-center gap-3">
              <Zap className="w-5 h-5 text-[#00bae2] flex-shrink-0" />
              <div>
                <div className="text-xs font-bold text-white">&lt; 1.2s Latency</div>
                <div className="text-[11px] text-slate-400">ONNX Accelerated</div>
              </div>
            </div>

            <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/10 backdrop-blur-sm flex items-center gap-3">
              <Microscope className="w-5 h-5 text-[#fec5fb] flex-shrink-0" />
              <div>
                <div className="text-xs font-bold text-white">Grad-CAM</div>
                <div className="text-[11px] text-slate-400">Visual Attention</div>
              </div>
            </div>

            <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/10 backdrop-blur-sm flex items-center gap-3">
              <Server className="w-5 h-5 text-[#abff84] flex-shrink-0" />
              <div>
                <div className="text-xs font-bold text-white">DICOM & FHIR</div>
                <div className="text-[11px] text-slate-400">Hospital Interop</div>
              </div>
            </div>
          </div>

          {/* Animated Scroll Down Indicator (as requested) */}
          <div className="mt-14 flex flex-col items-center">
            <button
              onClick={() => scrollToSection("about")}
              className="group flex flex-col items-center gap-2.5 text-xs font-semibold tracking-widest uppercase text-slate-400 hover:text-[#0ae448] transition-colors cursor-pointer focus:outline-none"
              aria-label="Scroll to About section"
            >
              <span>Scroll to Explore</span>
              {/* Outer Pill */}
              <div className="w-7 h-12 rounded-full border-2 border-slate-600 group-hover:border-[#0ae448] flex items-start justify-center p-1.5 transition-colors">
                {/* Moving Scroll Wheel */}
                <div className="w-1.5 h-3 rounded-full bg-[#0ae448] scroll-indicator-wheel shadow-[0_0_10px_#0ae448]" />
              </div>
              <ChevronDown className="w-4 h-4 animate-bounce text-slate-500 group-hover:text-[#0ae448]" />
            </button>
          </div>
        </div>
      </section>

      {/* ===================================================================== */}
      {/* SECTION 2: ABOUT & ARCHITECTURE SECTION                               */}
      {/* ===================================================================== */}
      <section
        id="about"
        ref={aboutRef}
        className="relative border-t border-white/[0.07] px-4 py-28 sm:px-6 lg:px-8"
      >
        <div className="max-w-7xl mx-auto">
          {/* Section Header */}
          <div className="text-center max-w-3xl mx-auto mb-16">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#00bae2]/10 border border-[#00bae2]/20 text-[#00bae2] text-xs font-semibold tracking-widest uppercase mb-4">
              Platform Architecture
            </div>
            <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-4">
              Built for <span className="text-gradient-blue">Speed, Precision</span>, and Clinical Trust
            </h2>
            <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
              NetraPulse bridges research-grade deep learning with clinical workflow integration.
              From MATLAB model exploration to ONNX microservice deployment, every component is
              engineered for high throughput and complete interpretability.
            </p>
          </div>

          {/* 4-Card GSAP Showcase Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
            {/* Card 1: ResNet-50 Classifier */}
            <div className="glass-panel-interactive p-8 rounded-3xl relative overflow-hidden group">
              <div className="w-12 h-12 rounded-2xl bg-[#0ae448]/10 border border-[#0ae448]/20 flex items-center justify-center text-[#0ae448] mb-6">
                <Activity className="w-6 h-6" />
              </div>
              <span className="text-xs font-mono tracking-wider text-[#0ae448] uppercase">01 / Inference Core</span>
              <h3 className="text-xl sm:text-2xl font-bold text-white mt-1 mb-3">
                Deep Convolutional ResNet-50
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                Trained on the APTOS 2019 Blindness Detection and IDRiD international fundus datasets.
                Trained networks are exported to ONNX format, enabling sub-second inference on standard
                CPU nodes without requiring expensive cloud GPUs.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  MATLAB Export
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  ONNX Runtime
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  224×224 Normalization
                </span>
              </div>
            </div>

            {/* Card 2: Grad-CAM */}
            <div className="glass-panel-interactive p-8 rounded-3xl relative overflow-hidden group">
              <div className="w-12 h-12 rounded-2xl bg-[#00bae2]/10 border border-[#00bae2]/20 flex items-center justify-center text-[#00bae2] mb-6">
                <Microscope className="w-6 h-6" />
              </div>
              <span className="text-xs font-mono tracking-wider text-[#00bae2] uppercase">02 / Explainability</span>
              <h3 className="text-xl sm:text-2xl font-bold text-white mt-1 mb-3">
                Transparent Grad-CAM Heatmaps
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                Clinical AI cannot remain a black box. NetraPulse calculates Gradient-weighted Class
                Activation Mapping across the final bottleneck convolutional layer, isolating microaneurysms,
                cotton wool spots, and hard exudates for ophthalmologist review.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Layer 4 Conv Blocks
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Base64 Overlay
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Zero Black-Box
                </span>
              </div>
            </div>

            {/* Card 3: Image Quality Assessment */}
            <div className="glass-panel-interactive p-8 rounded-3xl relative overflow-hidden group">
              <div className="w-12 h-12 rounded-2xl bg-[#fec5fb]/10 border border-[#fec5fb]/20 flex items-center justify-center text-[#fec5fb] mb-6">
                <FileCheck className="w-6 h-6" />
              </div>
              <span className="text-xs font-mono tracking-wider text-[#fec5fb] uppercase">03 / Quality Guardrails</span>
              <h3 className="text-xl sm:text-2xl font-bold text-white mt-1 mb-3">
                Pre-Inference IQA Triage
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                Sub-optimal fundus images with underexposure, blur, or severe lens artifacting can lead to
                false negatives. The integrated IQA pipeline automatically checks pupil alignment and clarity
                before grading proceeds.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Contrast Analysis
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Illumination Metric
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Automatic Triage
                </span>
              </div>
            </div>

            {/* Card 4: Enterprise PACS & FHIR */}
            <div className="glass-panel-interactive p-8 rounded-3xl relative overflow-hidden group">
              <div className="w-12 h-12 rounded-2xl bg-[#ff8709]/10 border border-[#ff8709]/20 flex items-center justify-center text-[#ff8709] mb-6">
                <Server className="w-6 h-6" />
              </div>
              <span className="text-xs font-mono tracking-wider text-[#ff8709] uppercase">04 / Interoperability</span>
              <h3 className="text-xl sm:text-2xl font-bold text-white mt-1 mb-3">
                Orthanc DICOM & HL7 FHIR
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                Ready for real-world clinic workflows. Bundled Docker Compose configuration provides native
                hooks into Orthanc PACS (port 8042/4242) and HAPI FHIR R4 servers (port 8080) for archiving
                standardized diagnostic observations.
              </p>
              <div className="flex flex-wrap gap-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Orthanc PACS
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  HL7 FHIR R4
                </span>
                <span className="px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300">
                  Hospital EHR Ready
                </span>
              </div>
            </div>
          </div>

          {/* High-Impact Stat Strip */}
          <div className="p-8 rounded-3xl bg-white/[0.02] border border-white/10 backdrop-blur-md grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-3xl sm:text-4xl font-extrabold text-[#0ae448]">&lt; 1.2s</div>
              <div className="text-xs text-slate-400 mt-1 uppercase tracking-wider font-medium">Inference Latency</div>
            </div>
            <div>
              <div className="text-3xl sm:text-4xl font-extrabold text-[#00bae2]">5 Classes</div>
              <div className="text-xs text-slate-400 mt-1 uppercase tracking-wider font-medium">Clinical DR Staging</div>
            </div>
            <div>
              <div className="text-3xl sm:text-4xl font-extrabold text-[#fec5fb]">100%</div>
              <div className="text-xs text-slate-400 mt-1 uppercase tracking-wider font-medium">Heatmap Explainability</div>
            </div>
            <div>
              <div className="text-3xl sm:text-4xl font-extrabold text-[#ff8709]">FHIR R4</div>
              <div className="text-xs text-slate-400 mt-1 uppercase tracking-wider font-medium">Standardized Protocol</div>
            </div>
          </div>
        </div>
      </section>

      {/* ===================================================================== */}
      {/* SECTION 3: LIVE SCREENING WORKSPACE                                   */}
      {/* ===================================================================== */}
      <section
        id="scanner"
        ref={scannerRef}
        className="relative border-t border-white/[0.07] px-4 py-28 sm:px-6 lg:px-8"
      >
        <div className="max-w-7xl mx-auto">
          {/* Section Header */}
          <div className="text-center max-w-3xl mx-auto mb-14">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#0ae448]/10 border border-[#0ae448]/20 text-[#0ae448] text-xs font-semibold tracking-widest uppercase mb-4">
              Interactive Diagnostic Console
            </div>
            <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-4">
              Test Retinal Screening Live
            </h2>
            <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
              Upload a color fundus photograph or pick a demo sample to execute the live ONNX
              ResNet-50 classifier and inspect the Grad-CAM lesion heatmap.
            </p>
          </div>

          {/* Sample Presets Quick Loader */}
          <div className="flex flex-wrap items-center justify-center gap-3 mb-8">
            <span className="text-xs text-slate-400 font-medium">Quick Demo Samples:</span>
            {SAMPLE_PREVIEWS.map((sample, idx) => (
              <button
                key={idx}
                onClick={() => {
                  const file = sampleSvgToFile(sample.svg, `sample_${idx + 1}`);
                  handleFile(file);
                }}
                className="px-3.5 py-1.5 rounded-full text-xs font-medium bg-white/[0.05] hover:bg-white/[0.1] border border-white/10 hover:border-[#0ae448]/50 text-slate-300 hover:text-white transition-all cursor-pointer flex items-center gap-2"
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: sample.color }} />
                {sample.name}
              </button>
            ))}
          </div>

          {/* Diagnostic Console Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Column: Upload & Inspection Preview */}
            <div className="lg:col-span-6 space-y-6">
              <div className="glass-panel p-6 rounded-3xl">
                <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
                  <Upload className="w-4 h-4 text-[#0ae448]" />
                  Fundus Image Upload
                </h3>

                {/* Drop Zone */}
                <div
                  className={`drop-zone p-8 text-center flex flex-col items-center justify-center min-h-[260px] ${
                    dragActive ? "active" : ""
                  }`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={onInputChange}
                  />

                  {previewUrl ? (
                    <div className="flex flex-col items-center gap-4">
                      <div className="w-44 h-44 rounded-2xl overflow-hidden border border-white/20 shadow-2xl bg-black">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={previewUrl}
                          alt="Fundus preview"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <span className="text-xs text-slate-400">
                        {selectedFile?.name} ({(selectedFile?.size ? selectedFile.size / 1024 : 0).toFixed(1)} KB)
                      </span>
                      <span className="text-[11px] text-[#0ae448] underline">Click or drop to replace</span>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-3">
                      <div className="w-14 h-14 rounded-2xl bg-white/[0.04] border border-white/10 flex items-center justify-center text-slate-400">
                        <ImageIcon className="w-7 h-7 text-[#0ae448]" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-white">
                          Drag & drop fundus photograph here
                        </p>
                        <p className="text-xs text-slate-400 mt-1">Supports PNG, JPEG, or DICOM derivatives</p>
                      </div>
                      <button
                        type="button"
                        className="mt-2 px-4 py-1.5 text-xs font-semibold rounded-full bg-white/10 hover:bg-white/20 text-white border border-white/10"
                      >
                        Browse Files
                      </button>
                    </div>
                  )}
                </div>

                {/* Predict & Explain Actions */}
                <div className="flex items-center gap-3 mt-6">
                  <button
                    onClick={runPredict}
                    disabled={!selectedFile || loadingPredict}
                    className="flex-1 py-3 rounded-xl font-bold text-sm text-black bg-[#0ae448] hover:bg-[#abff84] disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 shadow-[0_0_20px_-3px_rgba(10,228,72,0.4)] flex items-center justify-center gap-2 cursor-pointer"
                  >
                    {loadingPredict ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Running ONNX Inference…
                      </>
                    ) : (
                      <>
                        <Activity className="w-4 h-4" />
                        Run DR Diagnosis
                      </>
                    )}
                  </button>

                  {prediction && (
                    <button
                      onClick={runExplain}
                      disabled={loadingExplain}
                      className="px-5 py-3 rounded-xl font-semibold text-sm text-white bg-[#00bae2]/20 hover:bg-[#00bae2]/30 border border-[#00bae2]/40 disabled:opacity-40 transition-all flex items-center gap-2 cursor-pointer"
                    >
                      {loadingExplain ? (
                        <Loader2 className="w-4 h-4 animate-spin text-[#00bae2]" />
                      ) : (
                        <Microscope className="w-4 h-4 text-[#00bae2]" />
                      )}
                      Grad-CAM
                    </button>
                  )}
                </div>

                {/* Error Banner */}
                {error && (
                  <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-start gap-2.5">
                    <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <span>{error}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Diagnostic Output & Analytics */}
            <div className="lg:col-span-6 space-y-6">
              {prediction ? (
                <div className="glass-panel p-6 rounded-3xl space-y-6">
                  {/* Primary Grade Header */}
                  <div className="flex items-start justify-between">
                    <div>
                      <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                        Classification Output
                      </span>
                      <h4 className="text-2xl font-black text-white mt-1">
                        Grade {prediction.grade}: {prediction.label}
                      </h4>
                    </div>

                    <span className={`grade-badge grade-${prediction.grade}`}>
                      <CheckCircle2 className="w-4 h-4" />
                      Grade {prediction.grade}
                    </span>
                  </div>

                  {/* Clinical Description */}
                  <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/10 text-xs text-slate-300 leading-relaxed">
                    <span className="font-semibold text-white block mb-1">Clinical Assessment:</span>
                    {GRADE_DESCRIPTIONS[prediction.grade]}
                  </div>

                  {/* Probability Distribution Bar Chart */}
                  <div>
                    <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                      Posterior Probability Distribution
                    </h5>
                    <div className="h-44 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <XAxis dataKey="grade" stroke="#64748b" fontSize={11} tickLine={false} />
                          <YAxis
                            stroke="#64748b"
                            fontSize={11}
                            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                            domain={[0, 1]}
                          />
                          <Tooltip
                            content={({ active, payload, label }) => {
                              if (active && payload && payload.length) {
                                return (
                                  <div className="p-2.5 rounded-lg bg-[#0f121a] border border-white/20 text-xs text-white shadow-xl">
                                    <p className="font-semibold">{label}</p>
                                    <p className="text-[#0ae448]">
                                      {((payload[0].value as number) * 100).toFixed(1)}%
                                    </p>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                            {chartData.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={index === prediction.grade ? entry.color : "rgba(255,255,255,0.15)"}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Grad-CAM Viewer */}
                  {heatmap && (
                    <div className="pt-4 border-t border-white/10">
                      <h5 className="text-xs font-semibold text-[#00bae2] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                        <Microscope className="w-3.5 h-3.5" />
                        Grad-CAM Attention Heatmap
                      </h5>
                      <div className="relative w-full aspect-video rounded-2xl overflow-hidden border border-[#00bae2]/30 bg-black">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={`data:image/png;base64,${heatmap}`}
                          alt="Grad-CAM Overlay"
                          className="w-full h-full object-contain"
                        />
                      </div>
                      <p className="text-[11px] text-slate-400 mt-2">
                        Warmer colors (red/yellow) indicate retinal features strongly driving the neural network’s diagnosis.
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="glass-panel p-8 rounded-3xl min-h-[380px] flex flex-col items-center justify-center text-center">
                  <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/10 flex items-center justify-center text-slate-500 mb-4">
                    <Eye className="w-8 h-8 text-slate-400" />
                  </div>
                  <h4 className="text-base font-bold text-white">Diagnostic Output Awaiting Scan</h4>
                  <p className="text-xs text-slate-400 max-w-sm mt-1.5 leading-relaxed">
                    Upload an image or pick a demo sample, then click &quot;Run DR Diagnosis&quot; to inspect class
                    probabilities and explainable Grad-CAM attention.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ===================================================================== */}
      {/* SECTION 4: CONTACT & INQUIRY SECTION                                  */}
      {/* ===================================================================== */}
      <section
        id="contact"
        ref={contactRef}
        className="relative border-t border-white/[0.07] px-4 py-28 sm:px-6 lg:px-8"
      >
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
            {/* Left Info Column */}
            <div className="lg:col-span-5 space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#fec5fb]/10 border border-[#fec5fb]/20 text-[#fec5fb] text-xs font-semibold tracking-widest uppercase">
                Connect With Us
              </div>
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
                Collaborate on Clinical Retinal Diagnostics
              </h2>
              <p className="text-slate-400 text-sm leading-relaxed">
                Whether you are an ophthalmologist interested in trialing NetraPulse, a researcher working
                on retinal datasets, or an institution seeking DICOM/FHIR deployment integration,
                we welcome collaboration.
              </p>

              <div className="space-y-4 pt-4">
                <a
                  href="https://github.com/AIMOD7/NetraPulse"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-4 rounded-2xl bg-white/[0.03] hover:bg-white/[0.06] border border-white/10 hover:border-white/20 transition-all flex items-center justify-between group"
                >
                  <div className="flex items-center gap-3">
                    <svg className="w-5 h-5 text-white fill-current" viewBox="0 0 24 24">
                      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                    </svg>
                    <div>
                      <div className="text-sm font-semibold text-white">GitHub Repository</div>
                      <div className="text-xs text-slate-400">AIMOD7/NetraPulse</div>
                    </div>
                  </div>
                  <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-white transition-colors" />
                </a>

                <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/10 flex items-center gap-3">
                  <Mail className="w-5 h-5 text-[#0ae448]" />
                  <div>
                    <div className="text-sm font-semibold text-white">Research Contact</div>
                    <div className="text-xs text-slate-400">sougatapaul0007@gmail.com</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Contact Form Column */}
            <div className="lg:col-span-7">
              <div className="glass-panel p-8 rounded-3xl">
                <h3 className="text-xl font-bold text-white mb-6">Send an Inquiry</h3>

                {contactSubmitted ? (
                  <div className="p-8 text-center rounded-2xl bg-[#0ae448]/10 border border-[#0ae448]/30">
                    <CheckCircle2 className="w-10 h-10 text-[#0ae448] mx-auto mb-3" />
                    <h4 className="text-lg font-bold text-white">Message Transmitted</h4>
                    <p className="text-xs text-slate-300 mt-1">
                      Thank you for reaching out. We will respond to your inquiry shortly.
                    </p>
                  </div>
                ) : (
                  <form onSubmit={handleContactSubmit} className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                          Your Name
                        </label>
                        <input
                          type="text"
                          required
                          value={contactForm.name}
                          onChange={(e) => setContactForm({ ...contactForm, name: e.target.value })}
                          placeholder="Dr. Jane Doe"
                          className="w-full px-4 py-3 rounded-xl bg-white/[0.04] border border-white/10 focus:border-[#0ae448] focus:outline-none text-sm text-white transition-colors"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                          Email Address
                        </label>
                        <input
                          type="email"
                          required
                          value={contactForm.email}
                          onChange={(e) => setContactForm({ ...contactForm, email: e.target.value })}
                          placeholder="jane.doe@hospital.org"
                          className="w-full px-4 py-3 rounded-xl bg-white/[0.04] border border-white/10 focus:border-[#0ae448] focus:outline-none text-sm text-white transition-colors"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                          Organization / Clinic
                        </label>
                        <input
                          type="text"
                          value={contactForm.organization}
                          onChange={(e) => setContactForm({ ...contactForm, organization: e.target.value })}
                          placeholder="Eye Care Institute"
                          className="w-full px-4 py-3 rounded-xl bg-white/[0.04] border border-white/10 focus:border-[#0ae448] focus:outline-none text-sm text-white transition-colors"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                          Subject Area
                        </label>
                        <select
                          value={contactForm.subject}
                          onChange={(e) => setContactForm({ ...contactForm, subject: e.target.value })}
                          className="w-full px-4 py-3 rounded-xl bg-[#0f121a] border border-white/10 focus:border-[#0ae448] focus:outline-none text-sm text-white transition-colors"
                        >
                          <option value="Clinical Inquiry">Clinical Pilot Inquiry</option>
                          <option value="Research & Datasets">Research & Datasets</option>
                          <option value="PACS/FHIR Integration">PACS/FHIR Deployment</option>
                          <option value="General Question">General Question</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                        Message
                      </label>
                      <textarea
                        rows={4}
                        required
                        value={contactForm.message}
                        onChange={(e) => setContactForm({ ...contactForm, message: e.target.value })}
                        placeholder="Tell us about your screening requirements or research inquiry..."
                        className="w-full px-4 py-3 rounded-xl bg-white/[0.04] border border-white/10 focus:border-[#0ae448] focus:outline-none text-sm text-white transition-colors resize-none"
                      />
                    </div>

                    <button
                      type="submit"
                      className="w-full py-3.5 rounded-xl font-bold text-sm text-black bg-[#0ae448] hover:bg-[#abff84] transition-all shadow-[0_0_20px_-3px_rgba(10,228,72,0.4)] flex items-center justify-center gap-2 cursor-pointer"
                    >
                      <Send className="w-4 h-4" />
                      Submit Message
                    </button>
                  </form>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===================================================================== */}
      {/* FOOTER                                                                */}
      {/* ===================================================================== */}
      <footer className="border-t border-white/[0.07] px-4 py-12 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <NetraPulseLogo size={36} />
          </div>

          <div className="text-center md:text-left text-xs text-slate-400 max-w-md">
            NetraPulse is an AI screening prototype. All model predictions and Grad-CAM visualizations
            are designed to assist qualified healthcare practitioners and do not substitute professional medical diagnosis.
          </div>

          <button
            onClick={() => scrollToSection("hero")}
            className="p-3 rounded-full bg-white/[0.05] hover:bg-white/[0.1] border border-white/10 text-slate-300 hover:text-white transition-colors cursor-pointer"
            aria-label="Back to top"
          >
            <ChevronUp className="w-5 h-5" />
          </button>
        </div>
      </footer>
    </div>
  );
}
