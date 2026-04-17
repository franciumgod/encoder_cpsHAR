from __future__ import annotations

from typing import Dict

import numpy as np


def normalize_sampling_mode(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"raw", "origin", "original"}:
        return "raw"
    if text in {"downsampled", "downsample", "ds"}:
        return "downsampled"
    if text in {"both", "raw_downsampled", "hybrid"}:
        return "both"
    return "raw"


def normalize_downsample_method(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"interval", "stride", "decimate"}:
        return "interval"
    if text in {"mean_pool", "avg_pool", "average", "pool"}:
        return "mean_pool"
    if text in {
        "sliding_window",
        "sliding_mean",
        "moving_average",
        "sliding_window_mean",
        "rolling_mean",
        "sliding",
    }:
        return "sliding_window"
    return "interval"


def _compute_factor(source_hz: int, target_hz: int) -> int:
    src = int(source_hz)
    tgt = int(target_hz)
    if src <= 0 or tgt <= 0:
        return 1
    if tgt >= src:
        return 1
    ratio = float(src) / float(tgt)
    factor = int(round(ratio))
    return max(1, factor)


def downsample_windows(
    X: np.ndarray,
    source_hz: int,
    target_hz: int,
    method: str = "interval",
    window_size: int = 40,
    window_step: int = 20,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 3:
        return X_arr

    method = normalize_downsample_method(method)
    n, t, c = X_arr.shape
    if method == "sliding_window":
        win = max(1, int(window_size))
        step = max(1, int(window_step))
        if t < win:
            return np.mean(X_arr, axis=1, keepdims=True, dtype=np.float32).astype(np.float32, copy=False)
        starts = np.arange(0, t - win + 1, step, dtype=np.int32)
        if starts.size == 0:
            return np.mean(X_arr, axis=1, keepdims=True, dtype=np.float32).astype(np.float32, copy=False)

        # Prefix-sum based sliding-window mean:
        # mean[t0:t1] = (prefix[t1] - prefix[t0]) / win
        prefix = np.concatenate(
            [np.zeros((n, 1, c), dtype=np.float32), np.cumsum(X_arr, axis=1, dtype=np.float32)],
            axis=1,
        )
        sums = prefix[:, starts + win, :] - prefix[:, starts, :]
        return (sums / float(win)).astype(np.float32, copy=False)

    factor = _compute_factor(source_hz=source_hz, target_hz=target_hz)
    if factor <= 1:
        return X_arr

    if method == "interval":
        return X_arr[:, ::factor, :].astype(np.float32, copy=False)

    if method == "mean_pool":
        # Non-overlapping average pooling with stride == window == factor.
        usable = (t // factor) * factor
        if usable <= 0:
            return X_arr[:, :1, :].astype(np.float32, copy=False)
        trimmed = X_arr[:, :usable, :]
        pooled = trimmed.reshape(n, usable // factor, factor, c).mean(axis=2, dtype=np.float32)
        return pooled.astype(np.float32, copy=False)

    # Should not happen due normalization, keep interval as safe fallback.
    return X_arr[:, ::factor, :].astype(np.float32, copy=False)


def build_sampling_views(
    X: np.ndarray,
    mode: str,
    source_hz: int,
    target_hz: int,
    method: str,
    window_size: int = 40,
    window_step: int = 20,
) -> Dict[str, dict]:
    mode = normalize_sampling_mode(mode)
    out: Dict[str, dict] = {}
    if mode in {"raw", "both"}:
        out["raw"] = {
            "X": np.asarray(X, dtype=np.float32),
            "sample_hz": int(source_hz),
            "downsample_factor": 1,
            "downsample_method": "none",
            "downsample_window_size": None,
            "downsample_window_step": None,
        }

    if mode in {"downsampled", "both"}:
        norm_method = normalize_downsample_method(method)
        ds = downsample_windows(
            X=np.asarray(X, dtype=np.float32),
            source_hz=source_hz,
            target_hz=target_hz,
            method=norm_method,
            window_size=window_size,
            window_step=window_step,
        )
        if norm_method == "sliding_window":
            factor = max(1, int(window_step))
            eff_hz = int(round(float(source_hz) / float(factor)))
            win = max(1, int(window_size))
            step = max(1, int(window_step))
        else:
            factor = max(1, _compute_factor(source_hz=source_hz, target_hz=target_hz))
            eff_hz = int(round(float(source_hz) / float(factor)))
            win = None
            step = None
        out["downsampled"] = {
            "X": ds,
            "sample_hz": eff_hz,
            "downsample_factor": factor,
            "downsample_method": norm_method,
            "downsample_window_size": win,
            "downsample_window_step": step,
        }
    return out
