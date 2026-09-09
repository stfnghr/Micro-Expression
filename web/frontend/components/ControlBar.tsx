"use client";

import { Pause, Play } from "lucide-react";

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];

interface ControlBarProps {
  disabled?: boolean;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  speed: number;
  processingTimeSec?: number | null;
  onToggle: () => void;
  onSeek: (time: number) => void;
  onSpeed: (rate: number) => void;
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function ControlBar({
  disabled,
  isPlaying,
  currentTime,
  duration,
  speed,
  processingTimeSec,
  onToggle,
  onSeek,
  onSpeed,
}: ControlBarProps) {
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
  const nextSpeed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length] ?? 1;

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-30 flex justify-center px-4">
      <div
        className={`pointer-events-auto glass flex w-full max-w-3xl items-center gap-3 rounded-full px-3 py-2.5 shadow-glow sm:gap-4 ${
          disabled ? "opacity-40" : ""
        }`}
      >
        <button
          type="button"
          disabled={disabled}
          onClick={onToggle}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-white text-slate-950 transition hover:bg-sky-100 disabled:cursor-not-allowed"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <Pause className="h-4 w-4 fill-current" />
          ) : (
            <Play className="h-4 w-4 fill-current pl-0.5" />
          )}
        </button>

        <span className="w-10 shrink-0 text-right font-mono text-[11px] text-slate-400">
          {formatTime(currentTime)}
        </span>

        <div className="relative min-w-0 flex-1">
          <div className="pointer-events-none absolute inset-y-[7px] left-0 right-0 rounded-full bg-slate-800" />
          <div
            className="pointer-events-none absolute inset-y-[7px] left-0 rounded-full bg-gradient-to-r from-sky-400 to-violet-400"
            style={{ width: `${progress}%` }}
          />
          <input
            type="range"
            min={0}
            max={duration || 1}
            step={0.01}
            value={Number.isFinite(currentTime) ? currentTime : 0}
            disabled={disabled}
            onChange={(event) => onSeek(Number(event.target.value))}
            className="relative w-full bg-transparent"
            aria-label="Seek"
          />
        </div>

        <span className="w-10 shrink-0 font-mono text-[11px] text-slate-400">
          {formatTime(duration)}
        </span>

        {processingTimeSec != null ? (
          <span className="hidden shrink-0 rounded-full bg-emerald-400/10 px-2.5 py-1 text-[10px] text-emerald-200 ring-1 ring-emerald-300/20 md:inline">
            Processed in {processingTimeSec.toFixed(1)}s
          </span>
        ) : null}

        <button
          type="button"
          disabled={disabled}
          onClick={() => onSpeed(nextSpeed)}
          className="rounded-full px-2.5 py-1 text-[10px] tracking-wide text-slate-300 ring-1 ring-white/10 sm:hidden"
        >
          {speed}x
        </button>

        <div className="hidden items-center gap-1 sm:flex">
          {SPEEDS.map((rate) => (
            <button
              key={rate}
              type="button"
              disabled={disabled}
              onClick={() => onSpeed(rate)}
              className={`rounded-full px-2 py-1 text-[10px] tracking-wide transition ${
                speed === rate
                  ? "bg-slate-100 text-slate-950"
                  : "text-slate-500 hover:text-slate-200"
              }`}
            >
              {rate}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
