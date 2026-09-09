export type ModelKey = "original" | "r3d18" | "vit" | "gcn";

export type PanelStatus = "idle" | "processing" | "ready" | "error";

export interface VideoSources {
  original: string;
  r3d18: string;
  vit: string;
  gcn: string;
  filename?: string;
  file_id?: string;
  processing_time_sec?: number;
}

export interface VideoPanelMeta {
  key: ModelKey;
  title: string;
  subtitle: string;
  accent: string;
  glow: string;
}

export const VIDEO_PANELS: VideoPanelMeta[] = [
  {
    key: "original",
    title: "Original",
    subtitle: "Source clip",
    accent: "from-slate-400/80 to-slate-200/40",
    glow: "shadow-glow",
  },
  {
    key: "r3d18",
    title: "3D-CNN",
    subtitle: "R3D-18 · Kinetics",
    accent: "from-sky-400 to-cyan-300",
    glow: "shadow-glow",
  },
  {
    key: "vit",
    title: "ViT",
    subtitle: "Optical Flow · ImageNet",
    accent: "from-violet-400 to-fuchsia-300",
    glow: "shadow-glow-violet",
  },
  {
    key: "gcn",
    title: "GCN",
    subtitle: "ST-GCN · Face Mesh",
    accent: "from-emerald-400 to-teal-300",
    glow: "shadow-glow-emerald",
  },
];
