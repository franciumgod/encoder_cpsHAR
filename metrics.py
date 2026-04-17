from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    recall_score,
)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    return obj


def _normalize_multilabel_proba(y_prob):
    if y_prob is None:
        return None
    if isinstance(y_prob, list):
        cols = []
        for arr in y_prob:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cols.append(arr[:, 1])
            else:
                cols.append(arr.ravel())
        return np.column_stack(cols)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        return y_prob.reshape(-1, 1)
    return y_prob


def calculate_mcc_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(float(matthews_corrcoef(y_true[:, i], y_pred[:, i])))
    return float(np.mean(scores))


def evaluate_and_print_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    fold_idx: str,
    split_name: str = "Test",
) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = _normalize_multilabel_proba(y_prob)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_prob is None:
        y_prob = y_pred.astype(float)

    n_classes = y_true.shape[1]
    per_class = []
    for i in range(n_classes):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        y_s = y_prob[:, i]
        try:
            pr_auc_i = average_precision_score(y_t, y_s)
        except ValueError:
            pr_auc_i = np.nan
        per_class.append(
            {
                "name": class_names[i] if i < len(class_names) else f"class_{i}",
                "mcc": float(matthews_corrcoef(y_t, y_p)),
                "f1": float(f1_score(y_t, y_p, zero_division=0)),
                "recall": float(recall_score(y_t, y_p, zero_division=0)),
                "pr_auc": float(pr_auc_i) if not np.isnan(pr_auc_i) else np.nan,
                "brier": float(brier_score_loss(y_t, y_s)),
            }
        )

    macro_mcc = calculate_mcc_multilabel(y_true, y_pred)
    macro_f1 = float(np.mean([x["f1"] for x in per_class]))
    macro_recall = float(np.mean([x["recall"] for x in per_class]))
    macro_pr_auc = float(np.nanmean([x["pr_auc"] for x in per_class]))
    macro_brier = float(np.mean([x["brier"] for x in per_class]))

    print(f"\n[{split_name}] Fold {fold_idx} metrics")
    print(
        f"Macro-MCC={macro_mcc:.4f} | Macro-F1={macro_f1:.4f} | "
        f"Macro-Recall={macro_recall:.4f} | Macro PR-AUC={macro_pr_auc:.4f} | "
        f"Macro Brier={macro_brier:.4f}"
    )
    print(f"{'Class':20s} {'MCC':>8s} {'Recall':>8s} {'F1':>8s} {'PR-AUC':>8s} {'Brier':>8s}")
    for row in per_class:
        pr_auc_txt = f"{row['pr_auc']:.4f}" if not np.isnan(row["pr_auc"]) else "nan"
        print(
            f"{row['name'][:20]:20s} {row['mcc']:8.4f} {row['recall']:8.4f} "
            f"{row['f1']:8.4f} {pr_auc_txt:>8s} {row['brier']:8.4f}"
        )

    return {
        "macro_mcc": macro_mcc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_pr_auc": macro_pr_auc,
        "macro_brier": macro_brier,
        "per_class": per_class,
    }


def _multilabel_to_single_index(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(int)
    return np.argmax(y, axis=1).astype(int)


def _extract_plot_signal_from_windows(X):
    X = np.asarray(X)
    if X.ndim != 3:
        return np.arange(len(X)), np.asarray(X).reshape(-1)
    if X.shape[1] >= X.shape[2]:
        signal = X[:, -1, 0]
    else:
        signal = X[:, 0, -1]
    return np.arange(len(signal)), signal


def _plot_binary_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fold_label: str,
    save_dir: Path,
    show_plots: bool = False,
) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cms = multilabel_confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    n_cols = min(3, max(1, n_classes))
    n_rows = int(math.ceil(n_classes / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for i, cm in enumerate(cms):
        ax = axes[i]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(class_names[i] if i < len(class_names) else f"class_{i}")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(int(cm[r, c])), ha="center", va="center", color="black")
    for j in range(n_classes, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out = save_dir / f"binary_confusion_{fold_label}.png"
    fig.savefig(out, dpi=140)
    if show_plots:
        plt.show()
    plt.close(fig)
    return str(out)


def plot_confusion_and_timeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fold_label: str,
    X_for_timeline: np.ndarray | None,
    save_dir: Path,
    max_timeline_points: int = 8000,
    show_plots: bool = False,
) -> Dict[str, str]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_true_idx = _multilabel_to_single_index(y_true_arr)
    y_pred_idx = _multilabel_to_single_index(y_pred_arr)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig_cm, ax_cm = plt.subplots(figsize=(9, 7))
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax_cm.set_xticks(np.arange(n_classes))
    ax_cm.set_yticks(np.arange(n_classes))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(class_names)
    ax_cm.set_xlabel("Predicted class")
    ax_cm.set_ylabel("True class")
    ax_cm.set_title(f"Normalized Confusion Matrix ({fold_label})")
    for i in range(n_classes):
        for j in range(n_classes):
            ax_cm.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_file = save_dir / f"confusion_{fold_label}.png"
    fig_cm.savefig(cm_file, dpi=140)
    if show_plots:
        plt.show()
    plt.close(fig_cm)

    if X_for_timeline is None:
        t = np.arange(len(y_true_idx))
        signal = np.zeros_like(t, dtype=np.float32)
    else:
        t, signal = _extract_plot_signal_from_windows(X_for_timeline)
    n = min(len(t), len(y_true_idx), len(y_pred_idx))
    t = t[:n]
    signal = signal[:n]
    y_true_idx = y_true_idx[:n]
    y_pred_idx = y_pred_idx[:n]
    if n > max_timeline_points:
        idx = np.linspace(0, n - 1, max_timeline_points, dtype=int)
        t, signal, y_true_idx, y_pred_idx = t[idx], signal[idx], y_true_idx[idx], y_pred_idx[idx]

    cmap = plt.get_cmap("tab10", max(10, n_classes))
    class_colors = [cmap(i) for i in range(n_classes)]
    fig_tl, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for cls in range(n_classes):
        m_true = y_true_idx == cls
        m_pred = y_pred_idx == cls
        if np.any(m_true):
            axes[0].scatter(t[m_true], signal[m_true], s=4, color=class_colors[cls], label=class_names[cls], alpha=0.9)
        if np.any(m_pred):
            axes[1].scatter(t[m_pred], signal[m_pred], s=4, color=class_colors[cls], label=class_names[cls], alpha=0.9)
    axes[0].set_title(f"TRUE class timeline ({fold_label})")
    axes[1].set_title(f"PREDICTED class timeline ({fold_label})")
    axes[1].set_xlabel("Sample index")
    axes[0].set_ylabel("Signal")
    axes[1].set_ylabel("Signal")
    handles, labels_text = axes[1].get_legend_handles_labels()
    uniq = dict(zip(labels_text, handles))
    if uniq:
        axes[1].legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)
    plt.tight_layout()
    tl_file = save_dir / f"timeline_{fold_label}.png"
    fig_tl.savefig(tl_file, dpi=140)
    if show_plots:
        plt.show()
    plt.close(fig_tl)

    binary_file = _plot_binary_confusions(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        class_names=class_names,
        fold_label=fold_label,
        save_dir=save_dir,
        show_plots=show_plots,
    )
    return {
        "confusion_path": str(cm_file),
        "timeline_path": str(tl_file),
        "binary_confusion_path": str(binary_file),
    }


def evaluate_one_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    fold_label: str,
    output_dir: Path,
    X_for_timeline: np.ndarray | None = None,
    show_images: bool = False,
) -> Dict:
    metrics = evaluate_and_print_multilabel_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        fold_idx=fold_label,
        split_name="Test",
    )
    plot_paths = plot_confusion_and_timeline(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        fold_label=f"fold_{fold_label}",
        X_for_timeline=X_for_timeline,
        save_dir=output_dir,
        show_plots=show_images,
    )
    return {"metrics": metrics, "plots": plot_paths}


def aggregate_fold_metrics(fold_rows: List[Dict]) -> Dict:
    if not fold_rows:
        return {}

    mcc = [float(row["metrics"]["macro_mcc"]) for row in fold_rows]
    f1 = [float(row["metrics"]["macro_f1"]) for row in fold_rows]
    pr_auc = [float(row["metrics"]["macro_pr_auc"]) for row in fold_rows]
    brier = [float(row["metrics"]["macro_brier"]) for row in fold_rows]

    return {
        "mcc_per_fold": mcc,
        "macro_f1_per_fold": f1,
        "macro_pr_auc_per_fold": pr_auc,
        "macro_brier_per_fold": brier,
        "avg_mcc": float(np.mean(mcc)),
        "std_mcc": float(np.std(mcc)),
        "avg_macro_f1": float(np.mean(f1)),
        "avg_macro_pr_auc": float(np.nanmean(pr_auc)),
        "avg_macro_brier": float(np.mean(brier)),
    }
