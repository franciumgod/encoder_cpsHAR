from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    PACKAGE_PARENT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))

    from encoderblock.config import EncoderBlockConfig, parse_bool, parse_csv_ints
    from encoderblock.domain import build_feature_matrices
    from encoderblock.encoder import EncoderProjector
    from encoderblock.metrics import aggregate_fold_metrics, evaluate_one_fold, plot_confusion_and_timeline, to_jsonable
    from encoderblock.sampling import downsample_windows
    from encoderblock.tree_models import MultiLabelTreeModel
    from encoderblock.window_dataset import load_window_payload, split_window_payload
else:
    from .config import EncoderBlockConfig, parse_bool, parse_csv_ints
    from .domain import build_feature_matrices
    from .encoder import EncoderProjector
    from .metrics import aggregate_fold_metrics, evaluate_one_fold, plot_confusion_and_timeline, to_jsonable
    from .sampling import downsample_windows
    from .tree_models import MultiLabelTreeModel
    from .window_dataset import load_window_payload, split_window_payload


REQUIRED_SENSOR_COLS_9 = [
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

SCALE_SPECS = [
    {"name": "raw2000", "hz": 2000, "method": "none", "window": None, "step": None},
    {"name": "sw500", "hz": 500, "method": "sliding_window", "window": 40, "step": 4},
    {"name": "sw100", "hz": 100, "method": "sliding_window", "window": 40, "step": 20},
]

HANDCRAFTED_INJECTION_CHOICES = ("inside", "outside", "both")


def _parse_folds(text: str) -> list[int]:
    vals = [int(x) for x in parse_csv_ints(text, default=[1, 2, 3, 4])]
    out = []
    for x in vals:
        if x < 1 or x > 4:
            raise ValueError(f"Fold id must be in [1, 4], got {x}")
        out.append(int(x))
    return out or [1, 2, 3, 4]


def _normalize_handcrafted_feature_injection(value: str) -> str:
    text = str(value).strip().lower()
    if text in HANDCRAFTED_INJECTION_CHOICES:
        return text
    return "outside"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EncoderBlock fixed 3-scale pipeline (2000/raw + 500/sw40-4 + 100/sw40-20) with native XGBoost."
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--raw_dataset_file", default="cps_data_multi_label.pkl")
    parser.add_argument("--window_dataset_file", default=None)
    parser.add_argument("--step", type=int, default=250)
    parser.add_argument("--window_seconds", type=float, default=2.0)
    parser.add_argument("--sample_rate_hz", type=int, default=2000)
    parser.add_argument("--folds", default="1,2,3,4")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show_images", default=False)

    parser.add_argument(
        "--encoder_backend",
        default="inceptiontime",
        choices=["none", "mlp", "cnn1d", "rescnn1d", "tcn", "inceptiontime"],
    )
    parser.add_argument("--encoder_axis_mode", default="joint", choices=["joint", "per_axis"])
    parser.add_argument("--encoder_output_dim", type=int, default=64)
    parser.add_argument("--encoder_hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_channels", default="64,128")
    parser.add_argument("--encoder_kernels", default="3,5,7")
    parser.add_argument("--encoder_use_se", default=False)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--encoder_epochs", type=int, default=15)
    parser.add_argument("--encoder_batch_size", type=int, default=128)
    parser.add_argument("--encoder_lr", type=float, default=1e-3)

    parser.add_argument("--include_handcrafted_features", default=False)
    parser.add_argument(
        "--handcrafted_feature_injection",
        default="outside",
        choices=list(HANDCRAFTED_INJECTION_CHOICES),
    )
    parser.add_argument("--drop_zero_label_before_feature_extraction", default=False)
    parser.add_argument("--export_encoder_table", default=True)
    parser.add_argument("--encoder_table_dim", type=int, default=64)

    parser.add_argument("--xgb_device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--xgb_n_estimators", type=int, default=2000)
    parser.add_argument("--xgb_early_stopping_rounds", type=int, default=100)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _update_cfg_from_args(cfg: EncoderBlockConfig, args) -> EncoderBlockConfig:
    cfg.raw_dataset_file = str(args.raw_dataset_file)
    cfg.window_dataset_file = str(args.window_dataset_file).strip() if args.window_dataset_file else None
    cfg.step = int(args.step)
    cfg.window_seconds = float(args.window_seconds)
    cfg.sample_rate_hz = int(args.sample_rate_hz)
    cfg.folds = _parse_folds(args.folds)
    cfg.random_seed = int(args.seed)

    # Fixed pipeline behavior
    cfg.use_special_signal_combo = False
    cfg.use_synth_signals = True
    cfg.feature_domain = "time_freq"
    cfg.spectrum_method = "welch_psd"
    cfg.use_feature_engineering = True
    cfg.include_handcrafted_features = parse_bool(args.include_handcrafted_features, default=False)
    cfg.handcrafted_feature_injection = _normalize_handcrafted_feature_injection(args.handcrafted_feature_injection)
    if cfg.include_handcrafted_features and cfg.handcrafted_feature_injection in {"outside", "both"}:
        cfg.feature_fusion_mode = "hybrid"
    else:
        cfg.feature_fusion_mode = "encoder_only"

    cfg.use_rotation_augment = False
    cfg.augment_count = 0
    cfg.augment_target = "none"

    cfg.use_encoder = True
    cfg.encoder_backend = str(args.encoder_backend).strip().lower()
    cfg.encoder_axis_mode = str(args.encoder_axis_mode).strip().lower()
    cfg.encoder_output_dim = int(args.encoder_output_dim)
    cfg.encoder_hidden_dim = int(args.encoder_hidden_dim)
    cfg.encoder_channels = parse_csv_ints(args.encoder_channels, default=[64, 128])
    cfg.encoder_kernels = parse_csv_ints(args.encoder_kernels, default=[3, 5, 7])
    cfg.encoder_use_se = parse_bool(args.encoder_use_se, default=False)
    cfg.encoder_dropout = float(args.encoder_dropout)
    cfg.encoder_epochs = int(args.encoder_epochs)
    cfg.encoder_batch_size = int(args.encoder_batch_size)
    cfg.encoder_lr = float(args.encoder_lr)

    cfg.drop_zero_label_before_feature_extraction = parse_bool(
        args.drop_zero_label_before_feature_extraction, default=False
    )
    cfg.export_encoder_table = parse_bool(args.export_encoder_table, default=True)
    cfg.encoder_table_dim = max(1, int(args.encoder_table_dim))
    cfg.encoder_table_branch = "raw"

    cfg.model_type = "xgboost"
    cfg.train_with_val = False

    cfg.xgb_device = str(args.xgb_device).strip().lower()
    cfg.xgb_n_estimators = int(args.xgb_n_estimators)
    cfg.xgb_early_stopping_rounds = int(args.xgb_early_stopping_rounds)
    cfg.xgb_learning_rate = float(args.xgb_learning_rate)
    cfg.xgb_max_depth = int(args.xgb_max_depth)
    cfg.xgb_subsample = float(args.xgb_subsample)
    cfg.xgb_colsample_bytree = float(args.xgb_colsample_bytree)
    cfg.xgb_reg_alpha = float(args.xgb_reg_alpha)
    cfg.xgb_reg_lambda = float(args.xgb_reg_lambda)

    cfg.threshold = float(args.threshold)

    if args.data_dir:
        cfg.data_dir = Path(args.data_dir).resolve()
    if args.output:
        cfg.output_dir = Path(args.output).resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _ensure_nine_axis_windows(X: np.ndarray, sensor_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    x_arr = np.asarray(X, dtype=np.float32)
    if x_arr.ndim != 3:
        raise ValueError(f"Expected X with shape (n,time,channels), got {x_arr.shape}")

    idx = {name: i for i, name in enumerate(sensor_cols)}
    missing_base = [c for c in REQUIRED_SENSOR_COLS_9[:7] if c not in idx]
    if missing_base:
        raise ValueError(f"Missing required base channels for 9-axis completion: {missing_base}")

    n, t, _ = x_arr.shape
    out = np.empty((n, t, 9), dtype=np.float32)
    # base axes
    out[:, :, 0] = x_arr[:, :, idx["Acc.x"]]
    out[:, :, 1] = x_arr[:, :, idx["Acc.y"]]
    out[:, :, 2] = x_arr[:, :, idx["Acc.z"]]
    out[:, :, 3] = x_arr[:, :, idx["Gyro.x"]]
    out[:, :, 4] = x_arr[:, :, idx["Gyro.y"]]
    out[:, :, 5] = x_arr[:, :, idx["Gyro.z"]]
    out[:, :, 6] = x_arr[:, :, idx["Baro.x"]]

    if "Acc.norm" in idx:
        out[:, :, 7] = x_arr[:, :, idx["Acc.norm"]]
    else:
        out[:, :, 7] = np.sqrt(out[:, :, 0] ** 2 + out[:, :, 1] ** 2 + out[:, :, 2] ** 2)

    if "Gyro.norm" in idx:
        out[:, :, 8] = x_arr[:, :, idx["Gyro.norm"]]
    else:
        out[:, :, 8] = np.sqrt(out[:, :, 3] ** 2 + out[:, :, 4] ** 2 + out[:, :, 5] ** 2)

    return out.astype(np.float32, copy=False), list(REQUIRED_SENSOR_COLS_9)


def _build_fixed_three_scales(X: np.ndarray, source_hz: int) -> Dict[str, dict]:
    x_arr = np.asarray(X, dtype=np.float32)
    views = {
        "raw2000": {
            "X": x_arr,
            "sample_hz": int(source_hz),
            "downsample_method": "none",
            "downsample_window_size": None,
            "downsample_window_step": None,
        }
    }
    x_500 = downsample_windows(
        X=x_arr,
        source_hz=source_hz,
        target_hz=500,
        method="sliding_window",
        window_size=40,
        window_step=4,
    )
    x_100 = downsample_windows(
        X=x_arr,
        source_hz=source_hz,
        target_hz=100,
        method="sliding_window",
        window_size=40,
        window_step=20,
    )
    views["sw500"] = {
        "X": x_500.astype(np.float32, copy=False),
        "sample_hz": 500,
        "downsample_method": "sliding_window",
        "downsample_window_size": 40,
        "downsample_window_step": 4,
    }
    views["sw100"] = {
        "X": x_100.astype(np.float32, copy=False),
        "sample_hz": 100,
        "downsample_method": "sliding_window",
        "downsample_window_size": 40,
        "downsample_window_step": 20,
    }
    return views


def _filter_zero_label_samples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    y_arr = np.asarray(y)
    if y_arr.ndim != 2:
        raise ValueError(f"Expected y with shape (n_samples, n_labels), got shape={y_arr.shape}")
    y_arr = (y_arr > 0).astype(np.int8, copy=False)
    mask = np.sum(y_arr, axis=1) > 0
    dropped = int(np.sum(~mask))
    return np.asarray(X, dtype=np.float32)[mask], y_arr[mask], dropped


def _reduce_feature_dim(
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    target_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = max(1, int(target_dim))
    tr_arr = np.asarray(tr, dtype=np.float32)
    va_arr = np.asarray(va, dtype=np.float32)
    te_arr = np.asarray(te, dtype=np.float32)

    if tr_arr.shape[1] == dim:
        return tr_arr, va_arr, te_arr
    if tr_arr.shape[1] < dim:
        pad_tr = np.zeros((tr_arr.shape[0], dim - tr_arr.shape[1]), dtype=np.float32)
        pad_va = np.zeros((va_arr.shape[0], dim - va_arr.shape[1]), dtype=np.float32)
        pad_te = np.zeros((te_arr.shape[0], dim - te_arr.shape[1]), dtype=np.float32)
        return (
            np.hstack([tr_arr, pad_tr]).astype(np.float32, copy=False),
            np.hstack([va_arr, pad_va]).astype(np.float32, copy=False),
            np.hstack([te_arr, pad_te]).astype(np.float32, copy=False),
        )

    pca = PCA(n_components=dim, random_state=int(seed))
    tr_z = pca.fit_transform(tr_arr).astype(np.float32, copy=False)
    va_z = pca.transform(va_arr).astype(np.float32, copy=False)
    te_z = pca.transform(te_arr).astype(np.float32, copy=False)
    return tr_z, va_z, te_z


def _build_table_df(X_feat: np.ndarray, y: np.ndarray, label_cols: List[str], prefix: str) -> pd.DataFrame:
    x_arr = np.asarray(X_feat, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int8)
    feat_cols = [f"{prefix}_{i:03d}" for i in range(x_arr.shape[1])]
    feat_df = pd.DataFrame(x_arr, columns=feat_cols)
    label_df = pd.DataFrame(y_arr, columns=label_cols)
    return pd.concat([feat_df, label_df], axis=1)


def _safe_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _inject_handcrafted_into_encoder_tensor(encoder_tensor: np.ndarray, handcrafted: np.ndarray) -> np.ndarray:
    enc = np.asarray(encoder_tensor, dtype=np.float32)
    hand = np.asarray(handcrafted, dtype=np.float32)
    if hand.size == 0:
        return enc
    if enc.shape[0] != hand.shape[0]:
        raise ValueError(
            f"Encoder tensor and handcrafted feature rows mismatch: "
            f"encoder={enc.shape[0]} vs handcrafted={hand.shape[0]}"
        )
    if enc.ndim == 3:
        repeated = np.repeat(hand[:, None, :], enc.shape[1], axis=1).astype(np.float32, copy=False)
        return np.concatenate([enc, repeated], axis=2).astype(np.float32, copy=False)
    if enc.ndim == 2:
        return np.hstack([enc, hand]).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported encoder tensor ndim={enc.ndim}, expected 2D or 3D.")


def run(cfg: EncoderBlockConfig, show_images: bool = False) -> Path:
    requested_step = int(cfg.step)
    requested_sample_rate_hz = int(cfg.sample_rate_hz)
    payload, payload_path = load_window_payload(
        data_dir=cfg.data_dir,
        raw_dataset_file=cfg.raw_dataset_file,
        window_dataset_file=cfg.window_dataset_file,
        step=cfg.step,
        window_size=cfg.window_size,
    )
    print(f"resolved_window_data : {payload_path}")

    payload_step = _safe_int(payload.get("step_size"))
    if payload_step is not None and payload_step != int(cfg.step):
        print(f"[warn] requested step={cfg.step}, but payload step_size={payload_step}; using payload step_size.")
        cfg.step = int(payload_step)

    payload_sample_hz = _safe_int(payload.get("sample_frequency"))
    if payload_sample_hz is not None and payload_sample_hz > 0 and payload_sample_hz != int(cfg.sample_rate_hz):
        print(
            f"[warn] requested sample_rate_hz={cfg.sample_rate_hz}, "
            f"but payload sample_frequency={payload_sample_hz}; using payload sample_frequency."
        )
        cfg.sample_rate_hz = int(payload_sample_hz)

    handcrafted_mode = _normalize_handcrafted_feature_injection(
        str(getattr(cfg, "handcrafted_feature_injection", "outside"))
    )
    cfg.handcrafted_feature_injection = handcrafted_mode
    use_handcrafted_any = bool(cfg.include_handcrafted_features)
    use_handcrafted_for_encoder = use_handcrafted_any and handcrafted_mode in {"inside", "both"}
    use_handcrafted_for_tree = use_handcrafted_any and handcrafted_mode in {"outside", "both"}

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "encoderblock_fixed_multiscale",
        "config": to_jsonable(copy.deepcopy(cfg.__dict__)),
        "fixed_scales": SCALE_SPECS,
        "data_loading": {
            "resolved_window_dataset_file": str(payload_path),
            "payload_kind": payload.get("kind"),
            "source_dataset_file": payload.get("source_dataset_file"),
            "requested_step": int(requested_step),
            "effective_step": int(cfg.step),
            "step_size": payload.get("step_size"),
            "requested_sample_rate_hz": int(requested_sample_rate_hz),
            "effective_sample_rate_hz": int(cfg.sample_rate_hz),
            "window_size": payload.get("window_size"),
            "sample_frequency": payload.get("sample_frequency"),
        },
        "handcrafted_feature_usage": {
            "enabled": bool(use_handcrafted_any),
            "mode": handcrafted_mode,
            "to_encoder": bool(use_handcrafted_for_encoder),
            "to_tree_model": bool(use_handcrafted_for_tree),
        },
        "folds": [],
        "aggregate": {},
    }

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_timeline: List[np.ndarray] = []

    for fold in cfg.folds:
        val_id = fold + 1 if fold < 4 else 1
        split = split_window_payload(
            payload=payload,
            test_experiment_id=int(fold),
            val_experiment_id=int(val_id),
            use_special_signal_combo=False,
            use_synth_signals=True,
        )

        X_train, _ = _ensure_nine_axis_windows(split.X_train, split.sensor_cols)
        X_val, _ = _ensure_nine_axis_windows(split.X_val, split.sensor_cols)
        X_test, _ = _ensure_nine_axis_windows(split.X_test, split.sensor_cols)
        y_train = (np.asarray(split.y_train) > 0).astype(np.int8, copy=False)
        y_val = (np.asarray(split.y_val) > 0).astype(np.int8, copy=False)
        y_test = (np.asarray(split.y_test) > 0).astype(np.int8, copy=False)

        zero_label_drop_stats = {"train": 0, "val": 0, "test": 0}
        if cfg.drop_zero_label_before_feature_extraction:
            X_train, y_train, zero_label_drop_stats["train"] = _filter_zero_label_samples(X_train, y_train)
            X_val, y_val, zero_label_drop_stats["val"] = _filter_zero_label_samples(X_val, y_val)
            X_test, y_test, zero_label_drop_stats["test"] = _filter_zero_label_samples(X_test, y_test)
            if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                raise ValueError(
                    f"After dropping zero-label samples, one split is empty: "
                    f"train/val/test={len(X_train)}/{len(X_val)}/{len(X_test)}"
                )

        train_views = _build_fixed_three_scales(X_train, source_hz=cfg.sample_rate_hz)
        val_views = _build_fixed_three_scales(X_val, source_hz=cfg.sample_rate_hz)
        test_views = _build_fixed_three_scales(X_test, source_hz=cfg.sample_rate_hz)

        train_enc_blocks = []
        val_enc_blocks = []
        test_enc_blocks = []
        train_hand_blocks = []
        val_hand_blocks = []
        test_hand_blocks = []
        branch_records = []

        for branch_name in ["raw2000", "sw500", "sw100"]:
            tr_info = train_views[branch_name]
            va_info = val_views[branch_name]
            te_info = test_views[branch_name]

            tr_hand, tr_enc_tensor = build_feature_matrices(
                tr_info["X"],
                feature_domain="time_freq",
                spectrum_method=cfg.spectrum_method,
                use_feature_engineering=True,
                include_handcrafted_features=use_handcrafted_any,
            )
            va_hand, va_enc_tensor = build_feature_matrices(
                va_info["X"],
                feature_domain="time_freq",
                spectrum_method=cfg.spectrum_method,
                use_feature_engineering=True,
                include_handcrafted_features=use_handcrafted_any,
            )
            te_hand, te_enc_tensor = build_feature_matrices(
                te_info["X"],
                feature_domain="time_freq",
                spectrum_method=cfg.spectrum_method,
                use_feature_engineering=True,
                include_handcrafted_features=use_handcrafted_any,
            )

            if use_handcrafted_for_encoder:
                tr_enc_tensor = _inject_handcrafted_into_encoder_tensor(tr_enc_tensor, tr_hand)
                va_enc_tensor = _inject_handcrafted_into_encoder_tensor(va_enc_tensor, va_hand)
                te_enc_tensor = _inject_handcrafted_into_encoder_tensor(te_enc_tensor, te_hand)

            projector = EncoderProjector(cfg)
            tr_enc = projector.fit_transform(tr_enc_tensor)
            va_enc = projector.transform(va_enc_tensor)
            te_enc = projector.transform(te_enc_tensor)

            train_enc_blocks.append(np.asarray(tr_enc, dtype=np.float32))
            val_enc_blocks.append(np.asarray(va_enc, dtype=np.float32))
            test_enc_blocks.append(np.asarray(te_enc, dtype=np.float32))

            if use_handcrafted_for_tree:
                train_hand_blocks.append(np.asarray(tr_hand, dtype=np.float32))
                val_hand_blocks.append(np.asarray(va_hand, dtype=np.float32))
                test_hand_blocks.append(np.asarray(te_hand, dtype=np.float32))

            branch_records.append(
                {
                    "name": branch_name,
                    "sample_hz": int(tr_info["sample_hz"]),
                    "downsample_method": str(tr_info["downsample_method"]),
                    "downsample_window_size": tr_info["downsample_window_size"],
                    "downsample_window_step": tr_info["downsample_window_step"],
                    "encoder": to_jsonable(projector.meta.__dict__),
                    "encoder_shape": {
                        "train": list(np.asarray(tr_enc).shape),
                        "val": list(np.asarray(va_enc).shape),
                        "test": list(np.asarray(te_enc).shape),
                    },
                    "encoder_input_shape": {
                        "train": list(np.asarray(tr_enc_tensor).shape),
                        "val": list(np.asarray(va_enc_tensor).shape),
                        "test": list(np.asarray(te_enc_tensor).shape),
                    },
                    "handcrafted_shape": {
                        "train": list(np.asarray(tr_hand).shape),
                        "val": list(np.asarray(va_hand).shape),
                        "test": list(np.asarray(te_hand).shape),
                    },
                }
            )

        if not train_enc_blocks:
            raise ValueError("No encoder features were produced.")

        enc_train_concat = np.hstack(train_enc_blocks).astype(np.float32, copy=False)
        enc_val_concat = np.hstack(val_enc_blocks).astype(np.float32, copy=False)
        enc_test_concat = np.hstack(test_enc_blocks).astype(np.float32, copy=False)
        enc_train_64, enc_val_64, enc_test_64 = _reduce_feature_dim(
            enc_train_concat, enc_val_concat, enc_test_concat, target_dim=cfg.encoder_table_dim, seed=cfg.random_seed
        )

        if use_handcrafted_for_tree and train_hand_blocks:
            hand_train_concat = np.hstack(train_hand_blocks).astype(np.float32, copy=False)
            hand_val_concat = np.hstack(val_hand_blocks).astype(np.float32, copy=False)
            hand_test_concat = np.hstack(test_hand_blocks).astype(np.float32, copy=False)
            X_train_model = np.hstack([enc_train_64, hand_train_concat]).astype(np.float32, copy=False)
            X_val_model = np.hstack([enc_val_64, hand_val_concat]).astype(np.float32, copy=False)
            X_test_model = np.hstack([enc_test_64, hand_test_concat]).astype(np.float32, copy=False)
        else:
            X_train_model = enc_train_64
            X_val_model = enc_val_64
            X_test_model = enc_test_64

        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train_model).astype(np.float32, copy=False)
        X_val_model = scaler.transform(X_val_model).astype(np.float32, copy=False)
        X_test_model = scaler.transform(X_test_model).astype(np.float32, copy=False)

        fold_dir = cfg.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Export encoder-only tables: first K dims + labels
        encoder_table_paths = {}
        if cfg.export_encoder_table:
            enc_table_dir = fold_dir / "encoder_tables"
            enc_table_dir.mkdir(parents=True, exist_ok=True)
            train_enc_df = _build_table_df(enc_train_64, y_train, split.label_cols, prefix="enc")
            val_enc_df = _build_table_df(enc_val_64, y_val, split.label_cols, prefix="enc")
            test_enc_df = _build_table_df(enc_test_64, y_test, split.label_cols, prefix="enc")
            all_enc_df = pd.concat([train_enc_df, val_enc_df, test_enc_df], axis=0, ignore_index=True)

            train_enc_path = enc_table_dir / "train_encoder_table.pkl"
            val_enc_path = enc_table_dir / "val_encoder_table.pkl"
            test_enc_path = enc_table_dir / "test_encoder_table.pkl"
            all_enc_path = enc_table_dir / "all_encoder_table.pkl"
            train_enc_df.to_pickle(train_enc_path)
            val_enc_df.to_pickle(val_enc_path)
            test_enc_df.to_pickle(test_enc_path)
            all_enc_df.to_pickle(all_enc_path)
            encoder_table_paths = {
                "train": str(train_enc_path),
                "val": str(val_enc_path),
                "test": str(test_enc_path),
                "all": str(all_enc_path),
            }

        # Model table (tree input) also persisted and then reloaded for training
        model_table_dir = fold_dir / "model_tables"
        model_table_dir.mkdir(parents=True, exist_ok=True)
        train_model_df = _build_table_df(X_train_model, y_train, split.label_cols, prefix="feat")
        val_model_df = _build_table_df(X_val_model, y_val, split.label_cols, prefix="feat")
        test_model_df = _build_table_df(X_test_model, y_test, split.label_cols, prefix="feat")
        train_model_path = model_table_dir / "train_model_table.pkl"
        val_model_path = model_table_dir / "val_model_table.pkl"
        test_model_path = model_table_dir / "test_model_table.pkl"
        train_model_df.to_pickle(train_model_path)
        val_model_df.to_pickle(val_model_path)
        test_model_df.to_pickle(test_model_path)

        # Read table back to satisfy "train from encoder extracted table data".
        train_tbl = pd.read_pickle(train_model_path)
        val_tbl = pd.read_pickle(val_model_path)
        test_tbl = pd.read_pickle(test_model_path)
        n_label = len(split.label_cols)
        X_train_fit = train_tbl.iloc[:, :-n_label].to_numpy(dtype=np.float32, copy=False)
        y_train_fit = train_tbl.iloc[:, -n_label:].to_numpy(dtype=np.int8, copy=False)
        X_val_fit = val_tbl.iloc[:, :-n_label].to_numpy(dtype=np.float32, copy=False)
        y_val_fit = val_tbl.iloc[:, -n_label:].to_numpy(dtype=np.int8, copy=False)
        X_test_fit = test_tbl.iloc[:, :-n_label].to_numpy(dtype=np.float32, copy=False)
        y_test_fit = test_tbl.iloc[:, -n_label:].to_numpy(dtype=np.int8, copy=False)

        model = MultiLabelTreeModel(cfg=cfg, label_cols=split.label_cols)
        model.fit(
            X_train=X_train_fit,
            y_train=y_train_fit,
            X_val=X_val_fit,
            y_val=y_val_fit,
            train_with_val=False,
        )
        y_prob = model.predict_proba(X_test_fit)
        y_pred = (y_prob >= float(cfg.threshold)).astype(np.int8, copy=False)

        eval_result = evaluate_one_fold(
            y_true=y_test_fit,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=split.label_cols,
            fold_label=str(fold),
            output_dir=cfg.output_dir,
            X_for_timeline=test_views["raw2000"]["X"],
            show_images=show_images,
        )

        fold_row = {
            "fold": int(fold),
            "validation_experiment_id": int(val_id),
            "train_samples": int(len(X_train_fit)),
            "val_samples": int(len(X_val_fit)),
            "test_samples": int(len(X_test_fit)),
            "drop_zero_label_before_feature_extraction": bool(cfg.drop_zero_label_before_feature_extraction),
            "zero_label_drop_stats": zero_label_drop_stats,
            "include_handcrafted_features": bool(cfg.include_handcrafted_features),
            "handcrafted_feature_injection_mode": handcrafted_mode,
            "handcrafted_used_for_encoder": bool(use_handcrafted_for_encoder),
            "handcrafted_used_for_tree_model": bool(use_handcrafted_for_tree),
            "encoder_table_dim": int(cfg.encoder_table_dim),
            "feature_shape": {
                "encoder_concat_train": list(enc_train_concat.shape),
                "encoder_64_train": list(enc_train_64.shape),
                "model_train": list(X_train_model.shape),
            },
            "feature_branches": branch_records,
            "encoder_tables": encoder_table_paths,
            "model_tables": {
                "train": str(train_model_path),
                "val": str(val_model_path),
                "test": str(test_model_path),
            },
            "metrics": eval_result["metrics"],
            "plots": eval_result["plots"],
            "xgb_best_iterations": [int(x) for x in model.best_iterations],
        }
        run_summary["folds"].append(fold_row)

        all_true.append(y_test_fit)
        all_pred.append(y_pred)
        all_timeline.append(test_views["raw2000"]["X"])

    run_summary["aggregate"] = aggregate_fold_metrics(run_summary["folds"])
    if all_true and all_pred:
        overall_true = np.concatenate(all_true, axis=0)
        overall_pred = np.concatenate(all_pred, axis=0)
        timeline = np.concatenate(all_timeline, axis=0) if all_timeline else None
        overall_plots = plot_confusion_and_timeline(
            y_true=overall_true,
            y_pred=overall_pred,
            class_names=split.label_cols,
            fold_label="overall",
            X_for_timeline=timeline,
            save_dir=cfg.output_dir,
            show_plots=show_images,
        )
        run_summary["aggregate"]["overall_plots"] = overall_plots

    summary_path = cfg.output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(run_summary), f, ensure_ascii=False, indent=2)
    return summary_path


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = EncoderBlockConfig()
    if not args.output:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(cfg.output_dir / f"run_{stamp}")
    cfg = _update_cfg_from_args(cfg, args)

    print("\n=== EncoderBlock Fixed Pipeline ===")
    print(f"data_dir             : {cfg.data_dir}")
    print(f"window_dataset_file  : {cfg.window_dataset_file or '(auto by step)'}")
    print(f"step                 : {cfg.step}")
    print(f"folds                : {cfg.folds}")
    print(f"encoder_backend      : {cfg.encoder_backend}")
    print(f"encoder_axis_mode    : {cfg.encoder_axis_mode}")
    print(f"encoder_output_dim   : {cfg.encoder_output_dim}")
    print(f"encoder_table_dim    : {cfg.encoder_table_dim}")
    print(f"include_handcrafted  : {cfg.include_handcrafted_features}")
    print(f"handcrafted_mode     : {cfg.handcrafted_feature_injection}")
    print(f"drop_zero_label      : {cfg.drop_zero_label_before_feature_extraction}")
    print(f"export_encoder_table : {cfg.export_encoder_table}")
    print(f"xgb_device           : {cfg.xgb_device}")
    print(f"xgb_rounds           : {cfg.xgb_n_estimators}")
    print(f"xgb_early_stop       : {cfg.xgb_early_stopping_rounds}")
    print(f"threshold            : {cfg.threshold}")
    print(f"output               : {cfg.output_dir}")
    print("fixed_scales         : raw2000 + sw500(win40,step4) + sw100(win40,step20)")
    print("===================================\n")

    summary_path = run(cfg=cfg, show_images=parse_bool(args.show_images, default=False))
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
