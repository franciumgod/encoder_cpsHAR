from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import pywt
except Exception:  # pragma: no cover
    pywt = None


def normalize_feature_domain(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"time", "freq", "time_freq"}:
        return text
    return "time_freq"


def normalize_spectrum_method(value: str) -> str:
    text = str(value).strip().lower()
    compact = text.replace(" ", "").replace("-", "").replace("_", "")
    if compact in {"rfft", "fft"}:
        return "rfft"
    if compact in {"welch", "welchpsd", "psd"}:
        return "welch_psd"
    if compact in {"stft"}:
        return "stft"
    if compact in {"dwt", "wavelet"}:
        return "dwt"
    return "welch_psd"


def _build_rfft_spectrum(X: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(X, axis=1)
    return np.abs(spectrum).astype(np.float32, copy=False)


def _build_welch_psd_spectrum(X: np.ndarray) -> np.ndarray:
    n, t, _ = X.shape
    if t < 4:
        return _build_rfft_spectrum(X)

    nperseg = min(256, t)
    step = max(1, nperseg // 2)
    starts = list(range(0, t - nperseg + 1, step))
    if not starts:
        starts = [0]

    window = np.hanning(nperseg).astype(np.float32)
    win_norm = float(np.sum(window * window) + 1e-8)
    psd_acc = None

    for s in starts:
        seg = X[:, s:s + nperseg, :] * window[None, :, None]
        fft = np.fft.rfft(seg, axis=1)
        psd = (np.abs(fft) ** 2).astype(np.float32, copy=False) / win_norm
        psd_acc = psd if psd_acc is None else (psd_acc + psd)

    return (psd_acc / float(len(starts))).astype(np.float32, copy=False)


def _build_stft_spectrum(X: np.ndarray) -> np.ndarray:
    n, t, _ = X.shape
    if t < 4:
        return _build_rfft_spectrum(X)

    nperseg = min(256, t)
    step = max(1, nperseg // 2)
    starts = list(range(0, t - nperseg + 1, step))
    if not starts:
        starts = [0]

    window = np.hanning(nperseg).astype(np.float32)
    mags = []
    for s in starts:
        seg = X[:, s:s + nperseg, :] * window[None, :, None]
        fft = np.fft.rfft(seg, axis=1)
        mags.append(np.abs(fft).astype(np.float32, copy=False))

    stacked = np.stack(mags, axis=1)  # (n, frames, bins, c)
    return np.mean(stacked, axis=1, dtype=np.float32)


def _build_dwt_spectrum(X: np.ndarray) -> np.ndarray:
    if pywt is None:
        return _build_rfft_spectrum(X)

    n, _, c = X.shape
    dwt_blocks = []
    for ch in range(c):
        ch_windows = X[:, :, ch]
        approx_list = []
        detail_list = []
        for i in range(n):
            approx, detail = pywt.dwt(ch_windows[i], "db1", mode="symmetric")
            approx_list.append(approx)
            detail_list.append(detail)
        approx_arr = np.asarray(approx_list, dtype=np.float32)
        detail_arr = np.asarray(detail_list, dtype=np.float32)
        dwt_blocks.append(np.stack([approx_arr, detail_arr], axis=-1))

    # (n, bins, channels*2)
    return np.concatenate(dwt_blocks, axis=-1).astype(np.float32, copy=False)


def build_spectrum(X: np.ndarray, method: str) -> np.ndarray:
    method = normalize_spectrum_method(method)
    if method == "rfft":
        return _build_rfft_spectrum(X)
    if method == "welch_psd":
        return _build_welch_psd_spectrum(X)
    if method == "stft":
        return _build_stft_spectrum(X)
    if method == "dwt":
        return _build_dwt_spectrum(X)
    return _build_welch_psd_spectrum(X)


def extract_time_features(X: np.ndarray, use_feature_engineering: bool = True) -> np.ndarray:
    mean = np.mean(X, axis=1, dtype=np.float32)
    std = np.std(X, axis=1, dtype=np.float32)
    max_val = np.max(X, axis=1)
    min_val = np.min(X, axis=1)

    feat_blocks = [mean, std, max_val, min_val]
    if use_feature_engineering:
        rms = np.sqrt(np.mean(X * X, axis=1, dtype=np.float32))
        centered = X - mean[:, None, :]
        rmse = np.sqrt(np.mean(centered * centered, axis=1, dtype=np.float32))

        if X.shape[1] > 1:
            diff_1 = np.diff(X, n=1, axis=1)
            diff_1_mean = np.mean(diff_1, axis=1, dtype=np.float32)
        else:
            diff_1_mean = np.zeros_like(mean, dtype=np.float32)

        lag_blocks = []
        last_idx = X.shape[1] - 1
        for lag in (1, 3, 5, 10):
            lag_idx = max(0, last_idx - lag)
            lag_blocks.append(X[:, lag_idx, :])

        feat_blocks.extend([rms, rmse, diff_1_mean] + lag_blocks)
    return np.hstack(feat_blocks).astype(np.float32, copy=False)


def extract_freq_features(X: np.ndarray, method: str, use_feature_engineering: bool = True) -> np.ndarray:
    spec = build_spectrum(X, method=method)
    mean = np.mean(spec, axis=1, dtype=np.float32)
    std = np.std(spec, axis=1, dtype=np.float32)
    max_val = np.max(spec, axis=1)
    min_val = np.min(spec, axis=1)
    feat_blocks = [mean, std, max_val, min_val]

    if use_feature_engineering:
        power = np.mean(spec * spec, axis=1, dtype=np.float32)
        feat_blocks.append(power)

    return np.hstack(feat_blocks).astype(np.float32, copy=False)


def build_handcrafted_features(
    X: np.ndarray,
    feature_domain: str,
    spectrum_method: str,
    use_feature_engineering: bool = True,
) -> np.ndarray:
    feature_domain = normalize_feature_domain(feature_domain)
    blocks = []
    if feature_domain in {"time", "time_freq"}:
        blocks.append(extract_time_features(X, use_feature_engineering=use_feature_engineering))
    if feature_domain in {"freq", "time_freq"}:
        blocks.append(
            extract_freq_features(
                X,
                method=spectrum_method,
                use_feature_engineering=use_feature_engineering,
            )
        )
    if not blocks:
        return np.empty((X.shape[0], 0), dtype=np.float32)
    return np.hstack(blocks).astype(np.float32, copy=False)


def build_encoder_tensor(
    X: np.ndarray,
    feature_domain: str,
    spectrum_method: str,
) -> np.ndarray:
    feature_domain = normalize_feature_domain(feature_domain)
    if feature_domain == "time":
        return np.asarray(X, dtype=np.float32)
    freq = build_spectrum(X, method=spectrum_method)
    if feature_domain == "freq":
        return np.asarray(freq, dtype=np.float32)
    return np.concatenate([X, freq], axis=1).astype(np.float32, copy=False)


def flatten_tensor(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim <= 2:
        return X
    return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)


def build_feature_matrices(
    X: np.ndarray,
    feature_domain: str,
    spectrum_method: str,
    use_feature_engineering: bool = True,
    include_handcrafted_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - handcrafted_features: 2D feature matrix (n, f)
    - encoder_tensor: 3D/2D tensor before encoder projection
    """
    handcrafted = (
        build_handcrafted_features(
            X,
            feature_domain=feature_domain,
            spectrum_method=spectrum_method,
            use_feature_engineering=use_feature_engineering,
        )
        if include_handcrafted_features
        else np.empty((X.shape[0], 0), dtype=np.float32)
    )
    encoder_tensor = build_encoder_tensor(
        X,
        feature_domain=feature_domain,
        spectrum_method=spectrum_method,
    )
    return handcrafted, encoder_tensor

