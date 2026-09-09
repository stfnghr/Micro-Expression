"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const DRIFT_SECONDS = 0.08;
const SEEK_EPSILON = 0.02;

export function useSyncVideo(count = 4) {
  const refs = useRef<Array<HTMLVideoElement | null>>(
    Array.from({ length: count }, () => null),
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [readyCount, setReadyCount] = useState(0);
  const isPlayingRef = useRef(false);
  const seekingRef = useRef(false);
  const speedRef = useRef(1);
  const rafRef = useRef<number | null>(null);

  const videos = useCallback(
    () => refs.current.filter((video): video is HTMLVideoElement => video !== null),
    [],
  );

  const master = useCallback(() => videos()[0] ?? null, [videos]);

  const applyAll = useCallback(
    (fn: (video: HTMLVideoElement) => void) => {
      for (const video of videos()) fn(video);
    },
    [videos],
  );

  const alignTo = useCallback(
    (time: number) => {
      applyAll((video) => {
        if (Math.abs(video.currentTime - time) > SEEK_EPSILON) {
          video.currentTime = time;
        }
        video.playbackRate = speedRef.current;
      });
    },
    [applyAll],
  );

  const setRef = useCallback(
    (index: number) => (element: HTMLVideoElement | null) => {
      refs.current[index] = element;
    },
    [],
  );

  const pause = useCallback(() => {
    applyAll((video) => video.pause());
    isPlayingRef.current = false;
    setIsPlaying(false);
  }, [applyAll]);

  const play = useCallback(async () => {
    const lead = master();
    if (!lead) return;
    alignTo(lead.currentTime);
    isPlayingRef.current = true;
    const results = await Promise.allSettled(videos().map((video) => video.play()));
    const started = results.some((result) => result.status === "fulfilled");
    if (!started) {
      isPlayingRef.current = false;
      setIsPlaying(false);
      return;
    }
    setIsPlaying(true);
  }, [alignTo, master, videos]);

  const toggle = useCallback(() => {
    if (isPlayingRef.current) pause();
    else void play();
  }, [pause, play]);

  const seek = useCallback(
    (time: number) => {
      const lead = master();
      const max = Number.isFinite(duration) && duration > 0 ? duration : time;
      const clamped = Math.max(0, Math.min(time, max));
      seekingRef.current = true;
      alignTo(clamped);
      setCurrentTime(clamped);
      if (isPlayingRef.current) {
        void Promise.allSettled(videos().map((video) => video.play()));
      }
      window.setTimeout(() => {
        seekingRef.current = false;
      }, 80);
      if (lead && clamped >= (lead.duration || 0) - 0.05) {
        pause();
      }
    },
    [alignTo, duration, master, pause, videos],
  );

  const setPlaybackSpeed = useCallback(
    (rate: number) => {
      speedRef.current = rate;
      setSpeed(rate);
      applyAll((video) => {
        video.playbackRate = rate;
      });
    },
    [applyAll],
  );

  const reset = useCallback(() => {
    pause();
    setCurrentTime(0);
    setDuration(0);
    setReadyCount(0);
    applyAll((video) => {
      video.pause();
      video.currentTime = 0;
    });
  }, [applyAll, pause]);

  const refreshReady = useCallback(() => {
    const loaded = videos().filter((video) => video.readyState >= 1);
    setReadyCount(loaded.length);
    const lead = master();
    if (lead && Number.isFinite(lead.duration) && lead.duration > 0) {
      setDuration(lead.duration);
    }
  }, [master, videos]);

  const onLoadedMetadata = useCallback(() => {
    refreshReady();
  }, [refreshReady]);

  const onEnded = useCallback(() => {
    pause();
    const lead = master();
    if (lead && Number.isFinite(lead.duration)) {
      setCurrentTime(lead.duration);
    }
  }, [master, pause]);

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }

    const tick = () => {
      const lead = master();
      if (lead && !seekingRef.current) {
        setCurrentTime(lead.currentTime);
        applyAll((video) => {
          video.playbackRate = speedRef.current;
          if (video === lead) return;
          if (Math.abs(video.currentTime - lead.currentTime) > DRIFT_SECONDS) {
            video.currentTime = lead.currentTime;
          }
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [applyAll, isPlaying, master]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      if (event.target instanceof HTMLTextAreaElement) return;
      if (event.code !== "Space") return;
      const loaded = videos();
      if (loaded.length < count || loaded.some((video) => video.readyState < 1)) {
        return;
      }
      event.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, videos]);

  return {
    setRef,
    isPlaying,
    currentTime,
    duration,
    speed,
    readyCount,
    play,
    pause,
    toggle,
    seek,
    setPlaybackSpeed,
    reset,
    onLoadedMetadata,
    onEnded,
  };
}
