from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def parse_target_tokens(value: str) -> List[str]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return []
    normalized = text.replace(";", ",").replace("|", ",")
    out = []
    seen = set()
    for item in normalized.split(","):
        tok = item.strip().strip("\"'")
        if not tok:
            continue
        key = tok.lower()
        if key in seen:
            continue
        out.append(tok)
        seen.add(key)
    return out


def build_target_mask(y: np.ndarray, label_cols: List[str], target_spec: str) -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    tokens = parse_target_tokens(target_spec)
    if not tokens:
        return np.zeros((y_arr.shape[0],), dtype=bool)

    label_map = {name.strip().lower(): idx for idx, name in enumerate(label_cols)}
    mask = np.zeros((y_arr.shape[0],), dtype=bool)
    for tok in tokens:
        key = tok.strip().lower()
        if key in {"all", "any"}:
            mask |= np.ones_like(mask, dtype=bool)
            continue
        if key == "multilabel":
            mask |= np.sum(y_arr, axis=1) > 1
            continue
        idx = label_map.get(key)
        if idx is None:
            continue
        mask |= y_arr[:, idx].astype(bool)
    return mask


def _axis_pairs_for_plane(plane: str) -> Tuple[Tuple[str, str], ...]:
    p = str(plane).strip().lower()
    if p == "xy":
        return (("x", "y"),)
    if p == "xz":
        return (("x", "z"),)
    if p == "yz":
        return (("y", "z"),)
    return (("x", "y"), ("x", "z"), ("y", "z"))


def _collect_rotation_pairs(sensor_cols: List[str], plane: str) -> List[Tuple[int, int]]:
    name_to_idx = {name: i for i, name in enumerate(sensor_cols)}
    pairs = []
    for axis_a, axis_b in _axis_pairs_for_plane(plane):
        for prefix in ("Acc", "Gyro"):
            a = name_to_idx.get(f"{prefix}.{axis_a}")
            b = name_to_idx.get(f"{prefix}.{axis_b}")
            if a is not None and b is not None and a != b:
                pairs.append((a, b))
    # keep insertion order
    dedup = []
    seen = set()
    for item in pairs:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup


def _apply_rotation_sample(
    sample: np.ndarray,
    pairs: List[Tuple[int, int]],
    angle_deg: float,
) -> np.ndarray:
    if not pairs:
        return sample.copy()

    out = sample.copy()
    rad = np.deg2rad(float(angle_deg))
    c = np.cos(rad)
    s = np.sin(rad)
    for idx_a, idx_b in pairs:
        x = out[:, idx_a].copy()
        y = out[:, idx_b].copy()
        out[:, idx_a] = c * x - s * y
        out[:, idx_b] = s * x + c * y
    return out


def apply_rotation_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    sensor_cols: List[str],
    label_cols: List[str],
    augment_count: int,
    augment_target: str,
    rotation_plane: str = "xyz",
    rotation_max_degrees: float = 15.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)
    if X_arr.ndim != 3 or y_arr.ndim != 2:
        raise ValueError("Expected X shape (n, time, channels) and y shape (n, classes).")

    n = X_arr.shape[0]
    if n == 0 or int(augment_count) <= 0:
        return X_arr, y_arr, {"selected": 0, "augmented": 0, "ratio": 0.0}

    target_mask = build_target_mask(y_arr, label_cols=label_cols, target_spec=augment_target)
    selected_idx = np.where(target_mask)[0]
    if selected_idx.size == 0:
        return X_arr, y_arr, {"selected": 0, "augmented": 0, "ratio": 0.0}

    rng = np.random.default_rng(int(random_seed))
    base_plane = str(rotation_plane).strip().lower()
    plane_choices = ["xy", "xz", "yz"] if base_plane == "xyz" else [base_plane]

    X_new = [X_arr]
    y_new = [y_arr]
    aug_total = 0

    for idx in selected_idx:
        sample = X_arr[idx]
        label = y_arr[idx]
        for _ in range(int(augment_count)):
            chosen_plane = plane_choices[int(rng.integers(0, len(plane_choices)))]
            pairs = _collect_rotation_pairs(sensor_cols=sensor_cols, plane=chosen_plane)
            angle = rng.uniform(-abs(float(rotation_max_degrees)), abs(float(rotation_max_degrees)))
            aug_x = _apply_rotation_sample(sample, pairs=pairs, angle_deg=angle)
            X_new.append(aug_x[None, :, :].astype(np.float32, copy=False))
            y_new.append(label[None, :].astype(y_arr.dtype, copy=False))
            aug_total += 1

    X_out = np.concatenate(X_new, axis=0).astype(np.float32, copy=False)
    y_out = np.concatenate(y_new, axis=0).astype(y_arr.dtype, copy=False)

    return X_out, y_out, {
        "selected": int(selected_idx.size),
        "augmented": int(aug_total),
        "ratio": float(aug_total / max(1, n)),
    }

