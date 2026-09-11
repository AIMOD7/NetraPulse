"use client";

import React, { useState, useEffect } from "react";
import NetraPulseLogo from "./logo";
import { ArrowUpRight, Activity, Menu, X, ShieldCheck } from "lucide-react";

export function LandingHeader() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 40);
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollTo = (id: string) => {
    setMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-[#08090c]/85 backdrop-blur-md border-b border-white/10 py-3.5 shadow-2xl"
          : "bg-transparent py-5"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Brand Logo */}
        <button
          onClick={() => scrollTo("hero")}
          className="cursor-pointer focus:outline-none flex items-center"
        >
          <NetraPulseLogo size={38} />
        </button>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-1.5 p-1 rounded-full bg-white/[0.04] border border-white/10 backdrop-blur-md">
          <button
            onClick={() => scrollTo("hero")}
            className="px-4 py-1.5 text-xs font-medium text-slate-300 hover:text-white hover:bg-white/10 rounded-full transition-colors"
          >
            Home
          </button>
          <button
            onClick={() => scrollTo("about")}
            className="px-4 py-1.5 text-xs font-medium text-slate-300 hover:text-white hover:bg-white/10 rounded-full transition-colors"
          >
            About & Architecture
          </button>
          <button
            onClick={() => scrollTo("scanner")}
            className="px-4 py-1.5 text-xs font-medium text-[#0ae448] hover:bg-[#0ae448]/10 rounded-full transition-colors flex items-center gap-1.5"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[#0ae448] animate-ping" />
            Live Scanner
          </button>
          <button
            onClick={() => scrollTo("contact")}
            className="px-4 py-1.5 text-xs font-medium text-slate-300 hover:text-white hover:bg-white/10 rounded-full transition-colors"
          >
            Contact
          </button>
        </nav>

        {/* Right CTA / Telemetry */}
        <div className="hidden sm:flex items-center gap-3">
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            ONNX Engine Live
          </div>

          <button
            onClick={() => scrollTo("scanner")}
            className="px-4 py-2 text-xs font-semibold text-black bg-[#0ae448] hover:bg-[#abff84] rounded-full transition-all duration-200 shadow-[0_0_20px_-3px_rgba(10,228,72,0.4)] hover:shadow-[0_0_25px_0_rgba(10,228,72,0.6)] flex items-center gap-1.5"
          >
            Launch Scanner
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Mobile menu toggle */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 text-slate-300 hover:text-white focus:outline-none"
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Mobile dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#0e1017]/95 border-b border-white/10 px-6 py-4 space-y-3 backdrop-blur-xl">
          <button
            onClick={() => scrollTo("hero")}
            className="block w-full text-left py-2 text-sm font-medium text-slate-200 hover:text-[#0ae448]"
          >
            Home
          </button>
          <button
            onClick={() => scrollTo("about")}
            className="block w-full text-left py-2 text-sm font-medium text-slate-200 hover:text-[#0ae448]"
          >
            About & Architecture
          </button>
          <button
            onClick={() => scrollTo("scanner")}
            className="block w-full text-left py-2 text-sm font-medium text-[#0ae448]"
          >
            Live Scanner
          </button>
          <button
            onClick={() => scrollTo("contact")}
            className="block w-full text-left py-2 text-sm font-medium text-slate-200 hover:text-[#0ae448]"
          >
            Contact
          </button>
          <div className="pt-2">
            <button
              onClick={() => scrollTo("scanner")}
              className="w-full py-2.5 text-center text-xs font-semibold text-black bg-[#0ae448] rounded-full"
            >
              Launch Scanner
            </button>
          </div>
        </div>
      )}
    </header>
  );
}
export default LandingHeader;
