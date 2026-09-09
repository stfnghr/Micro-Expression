"use client";

import type { Ref } from "react";
import { VIDEO_PANELS, type PanelStatus, type VideoSources } from "@/lib/types";
import { VideoPanel } from "@/components/VideoPanel";

interface VideoGridProps {
  sources: VideoSources | null;
  previewUrl?: string | null;
  status: PanelStatus;
  isPlaying: boolean;
  elapsedLabel?: string | null;
  setRef: (index: number) => Ref<HTMLVideoElement>;
  onLoadedMetadata: () => void;
  onEnded: () => void;
}

export function VideoGrid({
  sources,
  previewUrl,
  status,
  isPlaying,
  elapsedLabel,
  setRef,
  onLoadedMetadata,
  onEnded,
}: VideoGridProps) {
  return (
    <section className="grid gap-4 md:grid-cols-2">
      {VIDEO_PANELS.map((meta, index) => {
        const isOriginal = meta.key === "original";
        const src = isOriginal
          ? (previewUrl ?? sources?.original ?? undefined)
          : sources?.[meta.key];
        const panelStatus: PanelStatus =
          isOriginal && (previewUrl || sources?.original)
            ? "ready"
            : status === "error"
              ? "idle"
              : status;

        return (
          <VideoPanel
            key={meta.key}
            meta={meta}
            status={panelStatus}
            src={src || undefined}
            videoRef={setRef(index)}
            isPlaying={isPlaying}
            elapsedLabel={!isOriginal && status === "processing" ? elapsedLabel : null}
            onLoadedMetadata={onLoadedMetadata}
            onEnded={onEnded}
          />
        );
      })}
    </section>
  );
}
