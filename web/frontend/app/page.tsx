"use client";

import { useCallback, useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { UploadDropzone } from "@/components/UploadDropzone";
import { VideoGrid } from "@/components/VideoGrid";
import { ControlBar } from "@/components/ControlBar";
import { DownloadButton } from "@/components/DownloadButton";
import { useSyncVideo } from "@/hooks/useSyncVideo";
import type { PanelStatus, VideoSources } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function fileIdFromSources(payload: VideoSources): string | null {
  if (payload.file_id && /^[0-9a-f]{32}$/i.test(payload.file_id)) {
    return payload.file_id;
  }
  const path = payload.original.split("/").pop() ?? "";
  const stem = path.replace(/\.[^.]+$/, "").replace(/_(r3d18|vit|gcn)$/, "");
  return /^[0-9a-f]{32}$/i.test(stem) ? stem : null;
}

function formatElapsed(seconds: number) {
  const safe = Math.max(0, seconds);
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  const tenth = Math.floor((safe % 1) * 10);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${tenth}`;
}

export default function HomePage() {
  const {
    setRef,
    isPlaying,
    currentTime,
    duration,
    speed,
    readyCount,
    toggle,
    seek,
    setPlaybackSpeed,
    reset,
    onLoadedMetadata,
    onEnded,
  } = useSyncVideo(4);
  const [status, setStatus] = useState<PanelStatus>("idle");
  const [sources, setSources] = useState<VideoSources | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [processingTimeSec, setProcessingTimeSec] = useState<number | null>(null);

  useEffect(() => {
    if (status !== "processing") return;
    const started = performance.now();
    setElapsedSec(0);
    const id = window.setInterval(() => {
      setElapsedSec((performance.now() - started) / 1000);
    }, 100);
    return () => window.clearInterval(id);
  }, [status]);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setFilename(file.name);
      setSources(null);
      setProcessingTimeSec(null);
      setPreviewUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return URL.createObjectURL(file);
      });
      reset();
      setStatus("processing");

      try {
        const body = new FormData();
        body.append("file", file);
        const response = await fetch(`${API_URL}/upload`, {
          method: "POST",
          body,
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `Upload failed (${response.status})`);
        }
        const payload = (await response.json()) as VideoSources;
        const toAbsolute = (url: string) =>
          url.startsWith("http") ? url : `${API_URL}${url}`;
        const rawTime = payload.processing_time_sec;
        const parsedTime =
          typeof rawTime === "number"
            ? rawTime
            : rawTime != null
              ? Number(rawTime)
              : null;
        setSources({
          original: toAbsolute(payload.original),
          r3d18: toAbsolute(payload.r3d18),
          vit: toAbsolute(payload.vit),
          gcn: toAbsolute(payload.gcn),
          filename: payload.filename,
          file_id: fileIdFromSources(payload) ?? undefined,
          processing_time_sec: parsedTime ?? undefined,
        });
        setProcessingTimeSec(
          parsedTime != null && Number.isFinite(parsedTime) ? parsedTime : null,
        );
        setStatus("ready");
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : "Upload failed");
      }
    },
    [reset],
  );

  const controlsEnabled =
    status === "ready" && Boolean(sources) && duration > 0 && readyCount >= 4;
  const fileId = sources ? fileIdFromSources(sources) : null;
  const downloadHref = fileId ? `${API_URL}/download/${fileId}` : null;
  const downloadEnabled = status === "ready" && Boolean(downloadHref);

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-900 bg-grain">
      <div className="pointer-events-none absolute -top-24 left-1/2 h-72 w-[42rem] -translate-x-1/2 rounded-full bg-sky-500/10 blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 right-0 h-80 w-80 rounded-full bg-violet-500/10 blur-3xl" />

      <main className="relative mx-auto flex max-w-6xl flex-col gap-8 px-5 pb-32 pt-8 sm:px-8">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <Header processingTimeSec={status === "ready" ? processingTimeSec : null} />
          <div className="w-full lg:max-w-xl">
            <UploadDropzone
              disabled={status === "processing"}
              processing={status === "processing"}
              elapsedSec={elapsedSec}
              filename={filename}
              onFile={handleFile}
              onInvalid={setError}
            />
          </div>
        </div>

        {error ? (
          <p className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-slate-200">
            {error}
          </p>
        ) : null}

        {status === "ready" && (processingTimeSec != null || downloadEnabled) ? (
          <div className="-mb-4 flex flex-wrap items-center justify-center gap-2">
            {processingTimeSec != null ? (
              <span className="inline-flex rounded-full bg-emerald-400/10 px-3 py-1 text-[11px] text-emerald-200 ring-1 ring-emerald-300/20">
                Processed in {processingTimeSec.toFixed(1)}s
              </span>
            ) : null}
            <DownloadButton href={downloadHref} enabled={downloadEnabled} />
          </div>
        ) : null}

        <VideoGrid
          sources={sources}
          previewUrl={previewUrl}
          status={status === "error" ? "idle" : status}
          isPlaying={isPlaying}
          elapsedLabel={status === "processing" ? formatElapsed(elapsedSec) : null}
          setRef={setRef}
          onLoadedMetadata={onLoadedMetadata}
          onEnded={onEnded}
        />

        {status === "ready" && readyCount < 4 ? (
          <p className="text-center text-[11px] uppercase tracking-[0.2em] text-slate-500">
            Syncing {readyCount}/4 streams
          </p>
        ) : null}

        <ControlBar
          disabled={!controlsEnabled}
          isPlaying={isPlaying}
          currentTime={currentTime}
          duration={duration}
          speed={speed}
          processingTimeSec={status === "ready" ? processingTimeSec : null}
          downloadHref={downloadHref}
          downloadEnabled={downloadEnabled}
          onToggle={toggle}
          onSeek={seek}
          onSpeed={setPlaybackSpeed}
        />
      </main>
    </div>
  );
}
