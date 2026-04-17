from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from build_windowed_sample_manifests import build_materialized_windows_for_step
from utils.config import Config as LegacyConfig


SPECIAL_SIGNAL_COMBO_BY_CLASS: Dict[str, List[str]] = {
    "Driving(curve)": ["Acc.x", "Acc.z", "Gyro.norm"],
    "Driving(straight)": ["Acc.y", "Acc.z", "Gyro.y", "Gyro.z", "Acc.norm", "Gyro.norm"],
    "Lifting(lowering)": ["Acc.z", "Gyro.x", "Baro.x"],
    "Lifting(raising)": ["Acc.x", "Acc.z", "Gyro.x", "Baro.x", "Acc.norm", "Gyro.norm"],
    "Stationary processes": ["Acc.y", "Gyro.x", "Gyro.z", "Baro.x", "Acc.norm", "Gyro.norm"],
    "Turntable wrapping": ["Acc.z", "Gyro.z", "Gyro.norm"],
}


@dataclass
class WindowSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    sensor_cols: List[str]
    label_cols: List[str]


def _window_file(data_dir: Path, step: int) -> Path:
    return data_dir / f"cps_windows_2s_2000hz_step_{int(step)}.pkl"


def ensure_window_dataset(data_dir: Path, raw_dataset_file: str, step: int, window_size: int) -> Path:
    out_path = _window_file(data_dir, step)
    if out_path.exists():
        return out_path

    raw_path = data_dir / Path(raw_dataset_file).name
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    legacy = LegacyConfig()
    legacy.data.raw_dataset_file = raw_dataset_file
    raw_meta = pd.read_pickle(raw_path)
    payload = build_materialized_windows_for_step(
        raw_meta=raw_meta,
        config=legacy,
        window_size=window_size,
        step=int(step),
    )
    pd.to_pickle(payload, out_path)
    return out_path


def load_window_payload(data_dir: Path, raw_dataset_file: str, step: int, window_size: int) -> dict:
    path = ensure_window_dataset(data_dir=data_dir, raw_dataset_file=raw_dataset_file, step=step, window_size=window_size)
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict) or payload.get("kind") != "window_samples":
        raise ValueError(f"Expected window_samples payload, got: {type(payload)}")
    return payload


def _select_sensor_indices(
    sensor_cols: List[str],
    use_special_signal_combo: bool,
    use_synth_signals: bool,
) -> Tuple[List[int], List[str]]:
    cols = list(sensor_cols)
    if not use_synth_signals:
        cols = [c for c in cols if c not in {"Acc.norm", "Gyro.norm"}]

    if use_special_signal_combo:
        keep = set()
        for _, names in SPECIAL_SIGNAL_COMBO_BY_CLASS.items():
            keep.update(names)
        cols = [c for c in cols if c in keep]

    if not cols:
        raise ValueError("No sensor channels left after switches.")

    idx_map = {name: i for i, name in enumerate(sensor_cols)}
    idxs = [idx_map[name] for name in cols]
    return idxs, cols


def split_window_payload(
    payload: dict,
    test_experiment_id: int,
    val_experiment_id: int,
    use_special_signal_combo: bool,
    use_synth_signals: bool,
) -> WindowSplit:
    X = np.asarray(payload["X"], dtype=np.float32)
    y = np.asarray(payload["y"], dtype=np.int8)
    exp = np.asarray(payload["experiment"])
    sensor_cols = list(payload["sensor_cols"])
    label_cols = list(payload["label_cols"])

    sensor_idxs, kept_sensor_cols = _select_sensor_indices(
        sensor_cols=sensor_cols,
        use_special_signal_combo=use_special_signal_combo,
        use_synth_signals=use_synth_signals,
    )
    X = X[:, :, sensor_idxs]

    test_mask = exp == int(test_experiment_id)
    val_mask = exp == int(val_experiment_id)
    train_mask = (~test_mask) & (~val_mask)

    return WindowSplit(
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_val=X[val_mask],
        y_val=y[val_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
        sensor_cols=kept_sensor_cols,
        label_cols=label_cols,
    )

