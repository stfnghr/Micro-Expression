"use client";

import { Download } from "lucide-react";

interface DownloadButtonProps {
  href?: string | null;
  enabled?: boolean;
  compact?: boolean;
}

export function DownloadButton({
  href,
  enabled = false,
  compact = false,
}: DownloadButtonProps) {
  const canDownload = Boolean(enabled && href);

  return (
    <a
      href={canDownload ? href! : undefined}
      onClick={(event) => {
        if (!canDownload) {
          event.preventDefault();
          return;
        }
        event.preventDefault();
        const link = document.createElement("a");
        link.href = href!;
        link.download = "FER_Results.zip";
        link.rel = "noopener";
        document.body.appendChild(link);
        link.click();
        link.remove();
      }}
      aria-disabled={!canDownload}
      className={
        compact
          ? `inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] ring-1 transition ${
              canDownload
                ? "bg-sky-400/10 text-sky-200 ring-sky-300/25 hover:bg-sky-400/20"
                : "pointer-events-none cursor-not-allowed text-slate-500 ring-white/10"
            }`
          : `inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[11px] font-medium ring-1 transition ${
              canDownload
                ? "bg-sky-400/10 text-sky-100 ring-sky-300/25 hover:bg-sky-400/20"
                : "pointer-events-none cursor-not-allowed text-slate-500 ring-white/10"
            }`
      }
    >
      <Download className={compact ? "h-3 w-3" : "h-3.5 w-3.5"} />
      {compact ? "ZIP" : "Download Results (ZIP)"}
    </a>
  );
}
