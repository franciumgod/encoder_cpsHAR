from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional


BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off", "none", "null", ""}


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in BOOL_TRUE:
        return True
    if text in BOOL_FALSE:
        return False
    return default


def parse_csv_ints(text: str, default: List[int]) -> List[int]:
    if text is None:
        return list(default)
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    if not items:
        return list(default)
    return [int(x) for x in items]


@dataclass
class EncoderBlockConfig:
    # resolved at runtime
    _ROOT: ClassVar[Path] = Path(__file__).resolve().parent

    # project / path
    project_root: Path = _ROOT
    data_dir: Path = _ROOT / "data"
    output_dir: Path = _ROOT / "output"

    # dataset
    raw_dataset_file: str = "cps_data_multi_label.pkl"
    window_dataset_file: Optional[str] = None
    step: int = 500
    window_seconds: float = 2.0
    sample_rate_hz: int = 2000
    test_experiment_id: int = 1
    val_experiment_id: int = 2

    # channel switches
    use_special_signal_combo: bool = False
    use_synth_signals: bool = True

    # domain / spectrum
    feature_domain: str = "time_freq"  # time | freq | time_freq
    spectrum_method: str = "welch_psd"  # rfft | welch_psd | stft | dwt
    use_feature_engineering: bool = True
    include_handcrafted_features: bool = True
    feature_fusion_mode: str = "auto"  # auto | handcrafted_only | encoder_only | hybrid

    # sampling views for feature extraction
    sampling_feature_mode: str = "raw"  # raw | downsampled | both
    downsample_target_hz: int = 100
    downsample_method: str = "interval"  # interval | mean_pool | sliding_window
    downsample_window_size: int = 40
    downsample_window_step: int = 20

    # augmentation
    use_rotation_augment: bool = True
    augment_count: int = 2
    augment_target: str = "multilabel,Lifting(raising),Lifting(lowering)"
    rotation_plane: str = "xz"
    rotation_max_degrees: float = 15.0

    # encoder
    use_encoder: bool = True
    encoder_backend: str = "mlp"  # mlp | cnn1d | rescnn1d | tcn | inceptiontime | none
    encoder_axis_mode: str = "joint"  # joint | per_axis
    encoder_output_dim: int = 128
    encoder_hidden_dim: int = 256
    encoder_channels: List[int] = None  # set in __post_init__
    encoder_kernels: List[int] = None
    encoder_use_se: bool = False
    encoder_dropout: float = 0.1
    encoder_epochs: int = 15
    encoder_batch_size: int = 128
    encoder_lr: float = 1e-3
    drop_zero_label_before_feature_extraction: bool = False
    export_encoder_table: bool = False
    encoder_table_dim: int = 64
    encoder_table_branch: str = "raw"

    # classifier
    model_type: str = "lightgbm"  # lightgbm | xgboost
    train_with_val: bool = True
    random_seed: int = 42
    folds: List[int] = None  # set in __post_init__

    # model hp
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 63
    lgbm_max_depth: int = 6
    lgbm_subsample: float = 0.9
    lgbm_colsample_bytree: float = 0.8
    lgbm_min_child_samples: int = 20
    lgbm_reg_alpha: float = 0.0
    lgbm_reg_lambda: float = 0.0

    xgb_n_estimators: int = 500
    xgb_early_stopping_rounds: int = 100
    xgb_device: str = "cuda"
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128]
        if self.encoder_kernels is None:
            self.encoder_kernels = [3, 5, 7]
        if self.folds is None:
            self.folds = [1, 2, 3, 4]

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def window_size(self) -> int:
        return int(round(self.window_seconds * self.sample_rate_hz))
