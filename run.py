from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .augment import apply_rotation_augmentation
from .config import EncoderBlockConfig, parse_bool, parse_csv_ints
from .data import load_window_payload, split_window_payload
from .domain import build_feature_matrices
from .encoder import EncoderProjector
from .metrics import aggregate_fold_metrics, evaluate_one_fold, plot_confusion_and_timeline, to_jsonable
from .sampling import build_sampling_views, normalize_downsample_method, normalize_sampling_mode
from .tree_models import MultiLabelTreeModel


def _parse_folds(text: str) -> list[int]:
    vals = parse_csv_ints(text, default=[1, 2, 3, 4])
    out = []
    for x in vals:
        if x < 1 or x > 4:
            raise ValueError(f"Fold id must be in [1, 4], got {x}")
        out.append(int(x))
    if not out:
        out = [1, 2, 3, 4]
    return out


def _update_cfg_from_args(cfg: EncoderBlockConfig, args) -> EncoderBlockConfig:
    cfg.raw_dataset_file = str(args.raw_dataset_file)
    cfg.step = int(args.step)
    cfg.window_seconds = float(args.window_seconds)
    cfg.sample_rate_hz = int(args.sample_rate_hz)
    cfg.folds = _parse_folds(args.folds)

    cfg.use_special_signal_combo = parse_bool(args.special_signal, default=False)
    cfg.use_synth_signals = parse_bool(args.use_synth_signals, default=True)

    cfg.feature_domain = str(args.feature_domain).strip().lower()
    cfg.spectrum_method = str(args.spectrum_method).strip().lower()
    cfg.use_feature_engineering = parse_bool(args.use_feature_engineering, default=True)
    cfg.include_handcrafted_features = parse_bool(args.include_handcrafted_features, default=True)
    cfg.feature_fusion_mode = str(args.feature_fusion_mode).strip().lower()
    cfg.sampling_feature_mode = normalize_sampling_mode(args.sampling_feature_mode)
    cfg.downsample_target_hz = int(args.downsample_target_hz)
    cfg.downsample_method = normalize_downsample_method(args.downsample_method)
    cfg.downsample_window_size = int(args.downsample_window_size)
    cfg.downsample_window_step = int(args.downsample_window_step)
    if cfg.downsample_target_hz <= 0:
        raise ValueError("downsample_target_hz must be > 0")
    if cfg.downsample_window_size <= 0:
        raise ValueError("downsample_window_size must be > 0")
    if cfg.downsample_window_step <= 0:
        raise ValueError("downsample_window_step must be > 0")

    cfg.use_rotation_augment = parse_bool(args.use_rotation_augment, default=True)
    cfg.augment_count = int(args.augment_count)
    cfg.augment_target = str(args.augment_target)
    cfg.rotation_plane = str(args.rotation_plane).strip().lower()
    cfg.rotation_max_degrees = float(args.rotation_max_degrees)

    cfg.use_encoder = parse_bool(args.use_encoder, default=True)
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

    cfg.model_type = str(args.model).strip().lower()
    cfg.train_with_val = parse_bool(args.train_with_val, default=True)
    cfg.random_seed = int(args.seed)

    cfg.lgbm_n_estimators = int(args.lgbm_n_estimators)
    cfg.lgbm_learning_rate = float(args.lgbm_learning_rate)
    cfg.lgbm_num_leaves = int(args.lgbm_num_leaves)
    cfg.lgbm_max_depth = int(args.lgbm_max_depth)
    cfg.lgbm_subsample = float(args.lgbm_subsample)
    cfg.lgbm_colsample_bytree = float(args.lgbm_colsample_bytree)
    cfg.lgbm_min_child_samples = int(args.lgbm_min_child_samples)

    cfg.xgb_n_estimators = int(args.xgb_n_estimators)
    cfg.xgb_learning_rate = float(args.xgb_learning_rate)
    cfg.xgb_max_depth = int(args.xgb_max_depth)
    cfg.xgb_subsample = float(args.xgb_subsample)
    cfg.xgb_colsample_bytree = float(args.xgb_colsample_bytree)

    if cfg.feature_fusion_mode == "handcrafted_only":
        cfg.include_handcrafted_features = True
        cfg.use_encoder = False
    elif cfg.feature_fusion_mode == "encoder_only":
        cfg.include_handcrafted_features = False
        cfg.use_encoder = True
    elif cfg.feature_fusion_mode == "hybrid":
        cfg.include_handcrafted_features = True
        cfg.use_encoder = True

    if cfg.feature_fusion_mode in {"encoder_only", "hybrid"} and cfg.encoder_backend == "none":
        raise ValueError("feature_fusion_mode requires encoder backend not equal to 'none'.")

    if args.data_dir:
        cfg.data_dir = Path(args.data_dir).resolve()
    if args.output:
        cfg.output_dir = Path(args.output).resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _hstack_nonempty(blocks: list[np.ndarray]) -> np.ndarray:
    valid = [b for b in blocks if b is not None and b.ndim == 2 and b.shape[1] > 0]
    if not valid:
        return np.empty((blocks[0].shape[0], 0), dtype=np.float32) if blocks else np.empty((0, 0), dtype=np.float32)
    return np.hstack(valid).astype(np.float32, copy=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal encoder + tree model pipeline for CPS HAR.")
    parser.add_argument("--data_dir", default=None, help="Dataset directory. Default: <repo>/encoderblock/data")
    parser.add_argument("--raw_dataset_file", default="cps_data_multi_label.pkl")
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--window_seconds", type=float, default=2.0)
    parser.add_argument("--sample_rate_hz", type=int, default=2000)
    parser.add_argument("--folds", default="1,2,3,4", help="Comma-separated folds in [1,2,3,4].")
    parser.add_argument("--output", default=None, help="Output directory.")
    parser.add_argument("--show_images", default=False)

    parser.add_argument("--special_signal", default=False, help="Use predefined special signal combination channels.")
    parser.add_argument("--use_synth_signals", default=True, help="Keep Acc.norm / Gyro.norm channels.")

    parser.add_argument("--feature_domain", default="time_freq", choices=["time", "freq", "time_freq"])
    parser.add_argument("--spectrum_method", default="welch_psd", choices=["rfft", "welch_psd", "stft", "dwt"])
    parser.add_argument(
        "--sampling_feature_mode",
        default="raw",
        choices=["raw", "downsampled", "both"],
        help="Feature extraction sampling view: raw / downsampled / both.",
    )
    parser.add_argument("--downsample_target_hz", type=int, default=100)
    parser.add_argument(
        "--downsample_method",
        default="interval",
        choices=["interval", "mean_pool", "sliding_window", "sliding_mean"],
        help="interval=抽点, mean_pool=不重叠均值池化, sliding_window=窗口均值(可设window/step)",
    )
    parser.add_argument("--downsample_window_size", type=int, default=40)
    parser.add_argument("--downsample_window_step", type=int, default=20)
    parser.add_argument("--use_feature_engineering", default=True)
    parser.add_argument("--include_handcrafted_features", default=True)
    parser.add_argument(
        "--feature_fusion_mode",
        default="auto",
        choices=["auto", "handcrafted_only", "encoder_only", "hybrid"],
        help="auto=respect booleans; handcrafted_only=only basic handcrafted; encoder_only=only encoder latent; hybrid=both.",
    )

    parser.add_argument("--use_rotation_augment", default=True)
    parser.add_argument("--augment_count", type=int, default=2)
    parser.add_argument("--augment_target", default="multilabel,Lifting(raising),Lifting(lowering)")
    parser.add_argument("--rotation_plane", default="xz", choices=["xy", "xz", "yz", "xyz"])
    parser.add_argument("--rotation_max_degrees", type=float, default=15.0)

    parser.add_argument("--use_encoder", default=True)
    parser.add_argument("--encoder_backend", default="mlp", choices=["none", "mlp", "cnn1d", "rescnn1d"])
    parser.add_argument("--encoder_axis_mode", default="joint", choices=["joint", "per_axis"])
    parser.add_argument("--encoder_output_dim", type=int, default=128)
    parser.add_argument("--encoder_hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_channels", default="64,128")
    parser.add_argument("--encoder_kernels", default="3,5,7")
    parser.add_argument("--encoder_use_se", default=False)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--encoder_epochs", type=int, default=15)
    parser.add_argument("--encoder_batch_size", type=int, default=128)
    parser.add_argument("--encoder_lr", type=float, default=1e-3)

    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--train_with_val", default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lgbm_n_estimators", type=int, default=500)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--lgbm_num_leaves", type=int, default=63)
    parser.add_argument("--lgbm_max_depth", type=int, default=6)
    parser.add_argument("--lgbm_subsample", type=float, default=0.9)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=20)

    parser.add_argument("--xgb_n_estimators", type=int, default=500)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    return parser


def run(cfg: EncoderBlockConfig, show_images: bool = False) -> Path:
    payload = load_window_payload(
        data_dir=cfg.data_dir,
        raw_dataset_file=cfg.raw_dataset_file,
        step=cfg.step,
        window_size=cfg.window_size,
    )

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "encoderblock",
        "config": to_jsonable(copy.deepcopy(cfg.__dict__)),
        "folds": [],
        "aggregate": {},
    }

    all_true = []
    all_pred = []
    all_timeline = []

    for fold in cfg.folds:
        val_id = fold + 1 if fold < 4 else 1
        split = split_window_payload(
            payload=payload,
            test_experiment_id=int(fold),
            val_experiment_id=int(val_id),
            use_special_signal_combo=cfg.use_special_signal_combo,
            use_synth_signals=cfg.use_synth_signals,
        )

        X_train, y_train = split.X_train, split.y_train
        X_val, y_val = split.X_val, split.y_val
        X_test, y_test = split.X_test, split.y_test

        if cfg.use_rotation_augment and cfg.augment_count > 0:
            X_train_aug, y_train_aug, aug_stats = apply_rotation_augmentation(
                X=X_train,
                y=y_train,
                sensor_cols=split.sensor_cols,
                label_cols=split.label_cols,
                augment_count=cfg.augment_count,
                augment_target=cfg.augment_target,
                rotation_plane=cfg.rotation_plane,
                rotation_max_degrees=cfg.rotation_max_degrees,
                random_seed=cfg.random_seed + int(fold),
            )
        else:
            X_train_aug, y_train_aug = X_train, y_train
            aug_stats = {"selected": 0, "augmented": 0, "ratio": 0.0}

        train_views = build_sampling_views(
            X=X_train_aug,
            mode=cfg.sampling_feature_mode,
            source_hz=cfg.sample_rate_hz,
            target_hz=cfg.downsample_target_hz,
            method=cfg.downsample_method,
            window_size=cfg.downsample_window_size,
            window_step=cfg.downsample_window_step,
        )
        val_views = build_sampling_views(
            X=X_val,
            mode=cfg.sampling_feature_mode,
            source_hz=cfg.sample_rate_hz,
            target_hz=cfg.downsample_target_hz,
            method=cfg.downsample_method,
            window_size=cfg.downsample_window_size,
            window_step=cfg.downsample_window_step,
        )
        test_views = build_sampling_views(
            X=X_test,
            mode=cfg.sampling_feature_mode,
            source_hz=cfg.sample_rate_hz,
            target_hz=cfg.downsample_target_hz,
            method=cfg.downsample_method,
            window_size=cfg.downsample_window_size,
            window_step=cfg.downsample_window_step,
        )

        train_blocks = []
        val_blocks = []
        test_blocks = []
        branch_records = []

        for branch_name, tr_info in train_views.items():
            va_info = val_views[branch_name]
            te_info = test_views[branch_name]
            X_train_branch = tr_info["X"]
            X_val_branch = va_info["X"]
            X_test_branch = te_info["X"]

            tr_hand, tr_enc_tensor = build_feature_matrices(
                X_train_branch,
                feature_domain=cfg.feature_domain,
                spectrum_method=cfg.spectrum_method,
                use_feature_engineering=cfg.use_feature_engineering,
                include_handcrafted_features=cfg.include_handcrafted_features,
            )
            va_hand, va_enc_tensor = build_feature_matrices(
                X_val_branch,
                feature_domain=cfg.feature_domain,
                spectrum_method=cfg.spectrum_method,
                use_feature_engineering=cfg.use_feature_engineering,
                include_handcrafted_features=cfg.include_handcrafted_features,
            )
            te_hand, te_enc_tensor = build_feature_matrices(
                X_test_branch,
                feature_domain=cfg.feature_domain,
                spectrum_method=cfg.spectrum_method,
                use_feature_engineering=cfg.use_feature_engineering,
                include_handcrafted_features=cfg.include_handcrafted_features,
            )

            projector = EncoderProjector(cfg)
            tr_enc = projector.fit_transform(tr_enc_tensor)
            va_enc = projector.transform(va_enc_tensor)
            te_enc = projector.transform(te_enc_tensor)

            tr_branch_feat = _hstack_nonempty([tr_hand, tr_enc])
            va_branch_feat = _hstack_nonempty([va_hand, va_enc])
            te_branch_feat = _hstack_nonempty([te_hand, te_enc])

            if tr_branch_feat.shape[1] > 0:
                train_blocks.append(tr_branch_feat)
                val_blocks.append(va_branch_feat)
                test_blocks.append(te_branch_feat)

            branch_records.append(
                {
                    "name": branch_name,
                    "sample_hz": int(tr_info["sample_hz"]),
                    "downsample_factor": int(tr_info["downsample_factor"]),
                    "downsample_method": str(tr_info["downsample_method"]),
                    "downsample_window_size": tr_info.get("downsample_window_size"),
                    "downsample_window_step": tr_info.get("downsample_window_step"),
                    "input_shape": {
                        "train": list(np.asarray(X_train_branch).shape),
                        "val": list(np.asarray(X_val_branch).shape),
                        "test": list(np.asarray(X_test_branch).shape),
                    },
                    "feature_shape": {
                        "train": list(tr_branch_feat.shape),
                        "val": list(va_branch_feat.shape),
                        "test": list(te_branch_feat.shape),
                    },
                    "encoder": to_jsonable(projector.meta.__dict__),
                }
            )

        if not train_blocks:
            raise ValueError("No features available. Enable handcrafted features or encoder.")

        X_train_feat = np.hstack(train_blocks).astype(np.float32, copy=False)
        X_val_feat = np.hstack(val_blocks).astype(np.float32, copy=False)
        X_test_feat = np.hstack(test_blocks).astype(np.float32, copy=False)

        scaler = StandardScaler()
        X_train_feat = scaler.fit_transform(X_train_feat).astype(np.float32, copy=False)
        X_val_feat = scaler.transform(X_val_feat).astype(np.float32, copy=False)
        X_test_feat = scaler.transform(X_test_feat).astype(np.float32, copy=False)

        model = MultiLabelTreeModel(cfg=cfg, label_cols=split.label_cols)
        model.fit(
            X_train=X_train_feat,
            y_train=y_train_aug,
            X_val=X_val_feat,
            y_val=y_val,
            train_with_val=cfg.train_with_val,
        )
        y_prob = model.predict_proba(X_test_feat)
        y_pred = (y_prob >= 0.5).astype(np.int8, copy=False)

        timeline_view = test_views["raw"]["X"] if "raw" in test_views else next(iter(test_views.values()))["X"]
        eval_result = evaluate_one_fold(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=split.label_cols,
            fold_label=str(fold),
            output_dir=cfg.output_dir,
            X_for_timeline=timeline_view,
            show_images=show_images,
        )

        if len(branch_records) == 1:
            encoder_field = branch_records[0]["encoder"]
        else:
            encoder_field = {"per_branch": {r["name"]: r["encoder"] for r in branch_records}}

        fold_row = {
            "fold": int(fold),
            "validation_experiment_id": int(val_id),
            "train_samples_before_aug": int(len(X_train)),
            "train_samples_after_aug": int(len(X_train_aug)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "augmentation": aug_stats,
            "sampling": {
                "mode": cfg.sampling_feature_mode,
                "downsample_target_hz": int(cfg.downsample_target_hz),
                "downsample_method": str(cfg.downsample_method),
                "downsample_window_size": int(cfg.downsample_window_size),
                "downsample_window_step": int(cfg.downsample_window_step),
            },
            "feature_branches": branch_records,
            "encoder": encoder_field,
            "feature_shape": {
                "train": list(X_train_feat.shape),
                "val": list(X_val_feat.shape),
                "test": list(X_test_feat.shape),
            },
            "metrics": eval_result["metrics"],
            "plots": eval_result["plots"],
        }
        run_summary["folds"].append(fold_row)

        all_true.append(y_test)
        all_pred.append(y_pred)
        all_timeline.append(timeline_view)

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

    print("\n=== EncoderBlock Run Config ===")
    print(f"data_dir             : {cfg.data_dir}")
    print(f"step                 : {cfg.step}")
    print(f"window_size          : {cfg.window_size}")
    print(f"folds                : {cfg.folds}")
    print(f"model                : {cfg.model_type}")
    print(f"sampling_mode        : {cfg.sampling_feature_mode}")
    print(f"downsample_target_hz : {cfg.downsample_target_hz}")
    print(f"downsample_method    : {cfg.downsample_method}")
    print(f"downsample_win_size  : {cfg.downsample_window_size}")
    print(f"downsample_win_step  : {cfg.downsample_window_step}")
    print(f"feature_domain       : {cfg.feature_domain}")
    print(f"spectrum_method      : {cfg.spectrum_method}")
    print(f"fusion_mode          : {cfg.feature_fusion_mode}")
    print(f"special_signal       : {cfg.use_special_signal_combo}")
    print(f"use_synth_signals    : {cfg.use_synth_signals}")
    print(f"use_encoder          : {cfg.use_encoder} ({cfg.encoder_backend})")
    print(f"encoder_axis_mode    : {cfg.encoder_axis_mode}")
    print(f"handcrafted_features : {cfg.include_handcrafted_features}")
    print(f"use_rotation_augment : {cfg.use_rotation_augment} x{cfg.augment_count}")
    print(f"output               : {cfg.output_dir}")
    print("===============================\n")

    summary_path = run(cfg=cfg, show_images=parse_bool(args.show_images, default=False))
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
