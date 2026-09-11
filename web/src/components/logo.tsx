import React from "react";

interface LogoProps {
  className?: string;
  size?: number;
  showText?: boolean;
}

export function NetraPulseLogo({ className = "", size = 44, showText = true }: LogoProps) {
  return (
    <div className={`inline-flex items-center gap-3 select-none ${className}`}>
      <div
        className="relative flex items-center justify-center flex-shrink-0"
        style={{ width: size, height: size }}
      >
        {/* Ambient background glow */}
        <div
          className="absolute inset-0 rounded-full blur-md opacity-60 animate-pulse"
          style={{
            background: "radial-gradient(circle, rgba(10,228,72,0.45) 0%, rgba(0,186,226,0.3) 60%, transparent 80%)",
          }}
        />

        {/* Vector SVG Emblem */}
        <svg
          width={size}
          height={size}
          viewBox="0 0 100 100"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="relative z-10 transition-transform duration-500 hover:scale-105"
        >
          <defs>
            <linearGradient id="npPulseGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#0ae448" />
              <stop offset="50%" stopColor="#00bae2" />
              <stop offset="100%" stopColor="#9d95ff" />
            </linearGradient>

            <linearGradient id="npEyeBorder" x1="0%" y1="50%" x2="100%" y2="50%">
              <stop offset="0%" stopColor="#00bae2" stopOpacity="0.2" />
              <stop offset="50%" stopColor="#0ae448" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#00bae2" stopOpacity="0.2" />
            </linearGradient>

            <radialGradient id="npIrisGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#0ae448" stopOpacity="0.8" />
              <stop offset="60%" stopColor="#00bae2" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#0e1015" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Outer Ophthalmic Eye Contour */}
          <path
            d="M 6 50 C 24 22, 76 22, 94 50 C 76 78, 24 78, 6 50 Z"
            stroke="url(#npEyeBorder)"
            strokeWidth="3.5"
            strokeLinecap="round"
            fill="#0b0d13"
          />

          {/* Retinal Fundus Vascular Rings */}
          <circle
            cx="50"
            cy="50"
            r="26"
            stroke="#00bae2"
            strokeWidth="1.2"
            strokeDasharray="3 3"
            strokeOpacity="0.5"
          />
          <circle
            cx="50"
            cy="50"
            r="19"
            stroke="#0ae448"
            strokeWidth="1.5"
            strokeOpacity="0.7"
          />

          {/* Glowing Iris Core */}
          <circle cx="50" cy="50" r="14" fill="url(#npIrisGlow)" />

          {/* Optic Center Pupil */}
          <circle cx="50" cy="50" r="7.5" fill="#040608" stroke="#0ae448" strokeWidth="2" />

          {/* Electrocardiogram / Optical Pulse Line crossing the pupil */}
          <path
            d="M 12 50 L 32 50 L 38 42 L 44 58 L 50 36 L 56 64 L 62 44 L 68 50 L 88 50"
            stroke="url(#npPulseGrad)"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Light reflection sparkles */}
          <circle cx="46" cy="46" r="2.2" fill="#ffffff" />
          <circle cx="55" cy="54" r="1.2" fill="#abff84" />
        </svg>
      </div>

      {showText && (
        <div className="flex flex-col">
          <div className="flex items-center gap-1.5">
            <span className="font-extrabold tracking-tight text-white text-xl leading-none">
              NETRA<span className="text-[#0ae448]">PULSE</span>
            </span>
            <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-[#0ae448]/10 text-[#0ae448] border border-[#0ae448]/20">
              AI
            </span>
          </div>
          <span className="text-[10px] tracking-wider text-slate-400 font-medium">
            Retinal Diagnostics Platform
          </span>
        </div>
      )}
    </div>
  );
}
export default NetraPulseLogo;
