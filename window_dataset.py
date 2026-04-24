from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_MODULE_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _MODULE_DIR.parent
DEFAULT_WINDOW_DATA_DIR = _MODULE_DIR / "data"
LEGACY_DATA_DIR = _PARENT_DIR / "cpsHAR" / "data"

DEFAULT_SENSOR_COLS: List[str] = [
    "Acc.x",
    "Acc.y",
    "Acc.z",
    "Gyro.x",
    "Gyro.y",
    "Gyro.z",
    "Baro.x",
    "Acc.norm",
    "Gyro.norm",
]

DEFAULT_SUPERCLASS_MAPPING: Dict[str, str] = {
    "Driving(curve)": "Driving(curve)",
    "Driving(straight)": "Driving(straight)",
    "Lifting(lowering)": "Lifting(lowering)",
    "Lifting(raising)": "Lifting(raising)",
    "Wrapping": "Turntable wrapping",
    "Wrapping(preparation)": "Stationary processes",
    "Docking": "Stationary processes",
    "Forks(entering or leaving front)": "Stationary processes",
    "Forks(entering or leaving side)": "Stationary processes",
    "Standing": "Stationary processes",
}

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


def canonical_window_dataset_file(step: int) -> str:
    return f"cps_windows_2s_2000hz_step_{int(step)}.pkl"


WINDOW_DATASET_RE = re.compile(r"^cps_windows_2s_2000hz_step_(\d+)\.pkl$", re.IGNORECASE)


def _parse_step_from_window_file(path: Path) -> int | None:
    match = WINDOW_DATASET_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _unique_paths(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def get_data_search_dirs(data_dir: Path | None) -> List[Path]:
    candidates: List[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    candidates.append(DEFAULT_WINDOW_DATA_DIR)
    if LEGACY_DATA_DIR.exists():
        candidates.append(LEGACY_DATA_DIR)
    return _unique_paths(candidates)


def _get_primary_data_dirs(data_dir: Path | None) -> List[Path]:
    candidates: List[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    candidates.append(DEFAULT_WINDOW_DATA_DIR)
    return _unique_paths(candidates)


def _discover_window_dataset_files(data_dir: Path | None, include_legacy: bool = True) -> List[Path]:
    paths: List[Path] = []
    search_dirs = get_data_search_dirs(data_dir) if include_legacy else _get_primary_data_dirs(data_dir)
    for base in search_dirs:
        if not base.exists() or not base.is_dir():
            continue
        for candidate in base.glob("cps_windows_2s_2000hz_step_*.pkl"):
            if _parse_step_from_window_file(candidate) is None:
                continue
            paths.append(candidate)
    return _unique_paths(paths)


def _resolve_existing_data_file(file_name_or_path: str | Path | None, data_dir: Path | None) -> Path | None:
    if not file_name_or_path:
        return None

    candidate = Path(file_name_or_path)
    locations: List[Path] = []
    if candidate.is_absolute():
        locations.append(candidate)
    else:
        locations.extend(
            [
                candidate,
                _MODULE_DIR / candidate,
                _PARENT_DIR / candidate,
            ]
        )
        for base_dir in get_data_search_dirs(data_dir):
            if candidate.parent == Path("."):
                locations.append(base_dir / candidate.name)
            else:
                locations.append(base_dir / candidate)

    for location in _unique_paths(locations):
        if location.exists():
            return location
    return None


def _clean_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore").reset_index(drop=True)


def _add_norm_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Acc.norm"] = np.sqrt(
        np.asarray(out["Acc.x"], dtype=np.float32) ** 2
        + np.asarray(out["Acc.y"], dtype=np.float32) ** 2
        + np.asarray(out["Acc.z"], dtype=np.float32) ** 2
    )
    out["Gyro.norm"] = np.sqrt(
        np.asarray(out["Gyro.x"], dtype=np.float32) ** 2
        + np.asarray(out["Gyro.y"], dtype=np.float32) ** 2
        + np.asarray(out["Gyro.z"], dtype=np.float32) ** 2
    )
    return out


def _apply_superclass_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    target_superclasses = sorted(set(mapping.values()))
    for super_name in target_superclasses:
        children = [child for child, parent in mapping.items() if parent == super_name]
        existing_children = [col for col in children if col in out.columns]
        if existing_children:
            out[super_name] = out[existing_children].max(axis=1)
        else:
            out[super_name] = 0

    cols_to_drop = [key for key in mapping.keys() if key in out.columns and key not in target_superclasses]
    out.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return out


def _count_windows(length: int, window_size: int, step: int) -> int:
    if length < window_size:
        return 0
    return 1 + (length - window_size) // step


def _build_materialized_windows_for_step(
    raw_meta: pd.DataFrame,
    window_size: int,
    step: int,
    sensor_cols: List[str] | None = None,
    superclass_mapping: Dict[str, str] | None = None,
) -> dict:
    sensor_cols = list(sensor_cols or DEFAULT_SENSOR_COLS)
    superclass_mapping = dict(superclass_mapping or DEFAULT_SUPERCLASS_MAPPING)
    final_target_cols = sorted(set(superclass_mapping.values()))
    n_channels = len(sensor_cols)
    n_labels = len(final_target_cols)

    total_windows = 0
    source_cache = []
    for source_index, row in raw_meta.iterrows():
        sample_df = _clean_raw_frame(row["data"])
        sample_df = _add_norm_signals(sample_df)
        sample_df = _apply_superclass_mapping(sample_df, superclass_mapping)

        missing_sensor_cols = [col for col in sensor_cols if col not in sample_df.columns]
        if missing_sensor_cols:
            raise KeyError(
                f"Raw sample is missing sensor columns required by encoderblock: {missing_sensor_cols}"
            )

        sample_len = len(sample_df)
        n_windows = _count_windows(sample_len, window_size, step)
        if n_windows <= 0:
            continue

        source_cache.append((source_index, row["scenario"], row["experiment"], sample_df, n_windows))
        total_windows += n_windows

    if total_windows == 0:
        raise ValueError(f"No windows were generated for step={step}.")

    X = np.empty((total_windows, window_size, n_channels), dtype=np.float32)
    y = np.empty((total_windows, n_labels), dtype=np.int8)
    label_ratio_pre_ds = np.empty((total_windows, n_labels), dtype=np.float32)
    experiment = np.empty(total_windows, dtype=np.int16)
    scenario = np.empty(total_windows, dtype=np.int16)
    source_indices = np.empty(total_windows, dtype=np.int32)
    start_indices = np.empty(total_windows, dtype=np.int32)

    write_pos = 0
    for source_index, scenario_id, experiment_id, sample_df, n_windows in source_cache:
        sensor_values = sample_df[sensor_cols].to_numpy(dtype=np.float32, copy=False)
        label_values = sample_df[final_target_cols].to_numpy(dtype=np.int8, copy=False)
        prefix = np.zeros((len(sample_df) + 1, n_labels), dtype=np.int64)
        if len(sample_df) > 0:
            prefix[1:] = np.cumsum(label_values, axis=0, dtype=np.int64)
        starts = np.arange(0, len(sample_df) - window_size + 1, step, dtype=np.int32)

        batch_size = 256
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start: batch_start + batch_size]
            batch_len = len(batch_starts)

            idx_matrix = batch_starts[:, None] + np.arange(window_size, dtype=np.int32)[None, :]
            X[write_pos: write_pos + batch_len] = sensor_values[idx_matrix]
            y[write_pos: write_pos + batch_len] = label_values[batch_starts + window_size - 1]
            sums = prefix[batch_starts + window_size] - prefix[batch_starts]
            label_ratio_pre_ds[write_pos: write_pos + batch_len] = (
                sums.astype(np.float32) / float(window_size)
            ).astype(np.float32, copy=False)
            experiment[write_pos: write_pos + batch_len] = experiment_id
            scenario[write_pos: write_pos + batch_len] = scenario_id
            source_indices[write_pos: write_pos + batch_len] = source_index
            start_indices[write_pos: write_pos + batch_len] = batch_starts
            write_pos += batch_len

    return {
        "kind": "window_samples",
        "source_dataset_file": None,
        "window_seconds": None,
        "window_size": window_size,
        "sample_frequency": None,
        "step_size": step,
        "sensor_cols": sensor_cols,
        "label_cols": final_target_cols,
        "label_ratio_pre_ds": label_ratio_pre_ds,
        "X": X,
        "y": y,
        "experiment": experiment,
        "scenario": scenario,
        "source_index": source_indices,
        "start_idx": start_indices,
    }


def ensure_window_dataset(
    data_dir: Path,
    raw_dataset_file: str,
    step: int,
    window_size: int,
    window_dataset_file: str | None = None,
) -> Path:
    canonical_name = canonical_window_dataset_file(step)

    if window_dataset_file:
        resolved_window_path = _resolve_existing_data_file(window_dataset_file, data_dir=data_dir)
        if resolved_window_path is not None:
            return resolved_window_path
    else:
        # Prefer explicitly provided/local data dirs before falling back to legacy dirs.
        primary_windows = _discover_window_dataset_files(data_dir=data_dir, include_legacy=False)
        primary_exact = [p for p in primary_windows if p.name.lower() == canonical_name.lower()]
        if primary_exact:
            return primary_exact[0]
        if len(primary_windows) == 1:
            only_path = primary_windows[0]
            detected_step = _parse_step_from_window_file(only_path)
            if detected_step is not None and int(detected_step) != int(step):
                print(
                    f"[window_dataset] Requested step={int(step)} not found; "
                    f"auto-using existing dataset {only_path.name} (step={detected_step})."
                )
            return only_path

        resolved_window_path = _resolve_existing_data_file(canonical_name, data_dir=data_dir)
        if resolved_window_path is not None:
            return resolved_window_path

    if window_dataset_file:
        search_text = ", ".join(str(p) for p in get_data_search_dirs(data_dir))
        raise FileNotFoundError(
            f"Window dataset not found: {window_dataset_file}. "
            f"Searched current path and data dirs: {search_text}"
        )

    discovered_windows = _discover_window_dataset_files(data_dir=data_dir, include_legacy=True)
    if len(discovered_windows) == 1:
        only_path = discovered_windows[0]
        detected_step = _parse_step_from_window_file(only_path)
        if detected_step is not None and int(detected_step) != int(step):
            print(
                f"[window_dataset] Requested step={int(step)} not found; "
                f"auto-using existing dataset {only_path.name} (step={detected_step})."
            )
        return only_path

    raw_path = _resolve_existing_data_file(raw_dataset_file, data_dir=data_dir)
    if raw_path is None:
        search_text = ", ".join(str(p) for p in get_data_search_dirs(data_dir))
        if discovered_windows:
            found_text = ", ".join(str(p) for p in discovered_windows)
            raise FileNotFoundError(
                f"Requested step={int(step)} dataset was not found, and raw dataset was not found: {raw_dataset_file}. "
                f"Found existing window datasets: {found_text}. "
                f"Searched current path and data dirs: {search_text}"
            )
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_dataset_file}. "
            f"Searched current path and data dirs: {search_text}"
        )

    raw_meta = pd.read_pickle(raw_path)
    if not isinstance(raw_meta, pd.DataFrame):
        raise ValueError(
            f"Raw dataset at {raw_path} must be a pandas DataFrame with columns like "
            "'data', 'scenario', and 'experiment'."
        )

    target_dir = Path(data_dir).resolve() if data_dir is not None else DEFAULT_WINDOW_DATA_DIR.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / canonical_window_dataset_file(step)

    payload = _build_materialized_windows_for_step(
        raw_meta=raw_meta,
        window_size=window_size,
        step=int(step),
    )
    payload["source_dataset_file"] = raw_path.name
    pd.to_pickle(payload, out_path)
    return out_path


def load_window_payload(
    data_dir: Path,
    raw_dataset_file: str,
    step: int,
    window_size: int,
    window_dataset_file: str | None = None,
) -> Tuple[dict, Path]:
    path = ensure_window_dataset(
        data_dir=data_dir,
        raw_dataset_file=raw_dataset_file,
        step=step,
        window_size=window_size,
        window_dataset_file=window_dataset_file,
    )
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict) or payload.get("kind") != "window_samples":
        raise ValueError(
            f"Expected a materialized 'window_samples' payload at {path}, "
            f"but got kind={payload.get('kind')!r}. "
            "If your file only stores manifest metadata, encoderblock cannot extract signal windows from it directly."
        )
    return payload, path


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
