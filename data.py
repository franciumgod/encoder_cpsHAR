"""Backward-compatible aliases for encoderblock window dataset helpers.

Prefer importing from `encoderblock.window_dataset`.
"""

from .window_dataset import (
    SPECIAL_SIGNAL_COMBO_BY_CLASS,
    WindowSplit,
    canonical_window_dataset_file,
    ensure_window_dataset,
    get_data_search_dirs,
    load_window_payload,
    split_window_payload,
)

__all__ = [
    "SPECIAL_SIGNAL_COMBO_BY_CLASS",
    "WindowSplit",
    "canonical_window_dataset_file",
    "ensure_window_dataset",
    "get_data_search_dirs",
    "load_window_payload",
    "split_window_payload",
]
