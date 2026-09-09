"use client";

import type { Ref } from "react";
import { LoaderCircle, ScanFace } from "lucide-react";
import type { PanelStatus, VideoPanelMeta } from "@/lib/types";

interface VideoPanelProps {
  meta: VideoPanelMeta;
  status: PanelStatus;
  src?: string;
  videoRef: Ref<HTMLVideoElement>;
  isPlaying: boolean;
  elapsedLabel?: string | null;
  onLoadedMetadata: () => void;
  onEnded: () => void;
}

export function VideoPanel({
  meta,
  status,
  src,
  videoRef,
  isPlaying,
  elapsedLabel,
  onLoadedMetadata,
  onEnded,
}: VideoPanelProps) {
  const active = status === "processing" || (status === "ready" && isPlaying);

  return (
    <article
      className={`group relative overflow-hidden rounded-[28px] bg-slate-900/70 ring-1 ring-white/5 backdrop-blur-sm transition duration-500 ${
        active ? `${meta.glow} ring-white/15` : "shadow-none"
      }`}
    >
      <div
        className={`pointer-events-none absolute inset-x-0 top-0 z-10 h-px bg-gradient-to-r ${meta.accent} opacity-80`}
      />

      <div className="relative aspect-video overflow-hidden bg-slate-950">
        {src && status === "ready" ? (
          <video
            ref={videoRef}
            src={src}
            muted
            playsInline
            preload="auto"
            controls={false}
            className="h-full w-full object-cover"
            onLoadedMetadata={onLoadedMetadata}
            onEnded={onEnded}
          />
        ) : (
          <div className="relative flex h-full w-full items-center justify-center">
            <div className="absolute inset-0 animate-[shimmer_2.4s_linear_infinite] bg-[linear-gradient(110deg,transparent_40%,rgba(255,255,255,0.05)_50%,transparent_60%)] bg-[length:200%_100%]" />
            {status === "processing" ? (
              <div className="relative flex flex-col items-center gap-3 text-slate-400">
                <span className="relative flex h-14 w-14 items-center justify-center">
                  <span className="absolute inset-0 rounded-full border border-slate-700" />
                  <span className="absolute inset-1 rounded-full border border-sky-400/20" />
                  <LoaderCircle className="h-10 w-10 animate-spin text-sky-300/90" />
                </span>
                <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">
                  Inferencing
                </p>
                {elapsedLabel ? (
                  <p className="font-mono text-[11px] text-sky-200">Elapsed: {elapsedLabel}</p>
                ) : null}
              </div>
            ) : (
              <div className="relative flex flex-col items-center gap-2 text-slate-600">
                <ScanFace className="h-7 w-7" strokeWidth={1.4} />
                <p className="text-xs tracking-wide">Awaiting clip</p>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between px-4 py-3">
        <div>
          <h2 className="text-sm font-medium text-slate-100">{meta.title}</h2>
          <p className="text-[11px] text-slate-500">{meta.subtitle}</p>
        </div>
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            status === "ready"
              ? "bg-emerald-400 shadow-[0_0_10px_#34d399]"
              : status === "processing"
                ? "animate-pulse bg-sky-400 shadow-[0_0_10px_#38bdf8]"
                : "bg-slate-600"
          }`}
        />
      </div>
    </article>
  );
}
