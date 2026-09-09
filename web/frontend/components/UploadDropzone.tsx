"use client";

import { useCallback, useState } from "react";
import { CloudUpload, FileVideo, Timer } from "lucide-react";

const ACCEPT = ".mp4,.avi,.webm,video/mp4,video/x-msvideo,video/webm";
const ALLOWED = new Set(["video/mp4", "video/x-msvideo", "video/webm", "video/avi"]);
const ALLOWED_EXT = new Set([".mp4", ".avi", ".webm"]);

interface UploadDropzoneProps {
  disabled?: boolean;
  filename?: string | null;
  processing?: boolean;
  elapsedSec?: number;
  onFile: (file: File) => void;
  onInvalid?: (message: string) => void;
}

function isAllowed(file: File) {
  const ext = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
  return ALLOWED.has(file.type) || ALLOWED_EXT.has(ext);
}

function formatElapsed(seconds: number) {
  const safe = Math.max(0, seconds);
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  const tenth = Math.floor((safe % 1) * 10);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${tenth}`;
}

export function UploadDropzone({
  disabled,
  filename,
  processing,
  elapsedSec = 0,
  onFile,
  onInvalid,
}: UploadDropzoneProps) {
  const [isOver, setIsOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      if (!isAllowed(file)) {
        onInvalid?.("Please drop an .mp4, .avi, or .webm clip.");
        return;
      }
      onFile(file);
    },
    [onFile, onInvalid],
  );

  return (
    <label
      className={`group relative flex w-full cursor-pointer items-center justify-between gap-4 overflow-hidden rounded-3xl border border-dashed px-5 py-4 transition duration-300 ${
        isOver
          ? "border-sky-300/70 bg-sky-400/10 shadow-[0_0_40px_-16px_rgba(56,189,248,0.8)]"
          : "border-slate-700/80 bg-slate-900/50 hover:border-slate-500"
      } ${disabled ? "pointer-events-none opacity-60" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsOver(true);
      }}
      onDragLeave={() => setIsOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsOver(false);
        handleFiles(event.dataTransfer.files);
      }}
    >
      <div className="pointer-events-none absolute inset-0 opacity-0 transition group-hover:opacity-100 bg-[radial-gradient(ellipse_at_top_right,rgba(56,189,248,0.08),transparent_55%)]" />
      <div className="relative flex min-w-0 items-center gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-slate-800/80 text-sky-300 ring-1 ring-white/5">
          <CloudUpload className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium text-slate-100">
            {processing ? "Processing models" : "Upload video"}
          </p>
          <p className="truncate text-xs text-slate-500">
            {processing
              ? "Rendering 3D-CNN, ViT, and ST-GCN overlays"
              : "Drop an .mp4, .avi, or .webm clip — or click to browse"}
          </p>
        </div>
      </div>
      {processing ? (
        <div className="relative flex items-center gap-2 rounded-full bg-sky-400/10 px-3 py-1.5 font-mono text-[11px] text-sky-200 ring-1 ring-sky-300/20">
          <Timer className="h-3.5 w-3.5" />
          Elapsed: {formatElapsed(elapsedSec)}
        </div>
      ) : (
        <div className="relative hidden max-w-[40%] items-center gap-2 truncate rounded-full bg-slate-800/80 px-3 py-1.5 text-[11px] text-slate-400 ring-1 ring-white/5 sm:flex">
          <FileVideo className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">{filename ?? "No file selected"}</span>
        </div>
      )}
      <input
        type="file"
        accept={ACCEPT}
        className="hidden"
        disabled={disabled}
        onChange={(event) => {
          handleFiles(event.target.files);
          event.currentTarget.value = "";
        }}
      />
    </label>
  );
}
