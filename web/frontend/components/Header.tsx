"use client";

import { ScanFace } from "lucide-react";

interface HeaderProps {
  processingTimeSec?: number | null;
}

export function Header({ processingTimeSec }: HeaderProps) {
  return (
    <header className="flex min-w-0 items-center gap-3">
      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-sky-400/10 ring-1 ring-sky-300/25 shadow-[0_0_24px_-8px_rgba(56,189,248,0.7)]">
        <ScanFace className="h-5 w-5 text-sky-300" strokeWidth={1.6} />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] uppercase tracking-[0.28em] text-slate-500">
          Micro-expression lab
        </p>
        <h1 className="truncate text-lg font-medium tracking-tight text-slate-50">
          FER Compare
        </h1>
      </div>
      {processingTimeSec != null ? (
        <span className="hidden rounded-full bg-emerald-400/10 px-3 py-1 text-[11px] text-emerald-200 ring-1 ring-emerald-300/20 sm:inline-flex">
          Processed in {processingTimeSec.toFixed(1)}s
        </span>
      ) : null}
    </header>
  );
}
