from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


class _ConstantBinaryEstimator:
    def __init__(self, constant: int):
        self.constant = int(constant)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        pos = np.full((n,), float(self.constant), dtype=np.float32)
        neg = 1.0 - pos
        return np.column_stack([neg, pos]).astype(np.float32, copy=False)

    def predict(self, X):
        return np.full((len(X),), self.constant, dtype=np.int8)


@dataclass
class ModelMeta:
    model_type: str
    n_classes: int
    fitted: bool = False


class MultiLabelTreeModel:
    def __init__(self, cfg, label_cols: List[str]):
        self.cfg = cfg
        self.label_cols = list(label_cols)
        self.model_type = str(getattr(cfg, "model_type", "lightgbm")).strip().lower()
        self.estimators: List[object] = []
        self.meta = ModelMeta(model_type=self.model_type, n_classes=len(self.label_cols), fitted=False)

    def _build_lgbm(self):
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not available in the current environment.")
        return LGBMClassifier(
            n_estimators=int(getattr(self.cfg, "lgbm_n_estimators", 500)),
            learning_rate=float(getattr(self.cfg, "lgbm_learning_rate", 0.05)),
            num_leaves=int(getattr(self.cfg, "lgbm_num_leaves", 63)),
            max_depth=int(getattr(self.cfg, "lgbm_max_depth", 6)),
            subsample=float(getattr(self.cfg, "lgbm_subsample", 0.9)),
            colsample_bytree=float(getattr(self.cfg, "lgbm_colsample_bytree", 0.8)),
            min_child_samples=int(getattr(self.cfg, "lgbm_min_child_samples", 20)),
            reg_alpha=float(getattr(self.cfg, "lgbm_reg_alpha", 0.0)),
            reg_lambda=float(getattr(self.cfg, "lgbm_reg_lambda", 0.0)),
            class_weight="balanced",
            random_state=int(getattr(self.cfg, "random_seed", 42)),
            n_jobs=-1,
            verbose=-1,
        )

    def _build_xgb(self):
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available in the current environment.")
        return XGBClassifier(
            n_estimators=int(getattr(self.cfg, "xgb_n_estimators", 500)),
            learning_rate=float(getattr(self.cfg, "xgb_learning_rate", 0.05)),
            max_depth=int(getattr(self.cfg, "xgb_max_depth", 6)),
            subsample=float(getattr(self.cfg, "xgb_subsample", 0.9)),
            colsample_bytree=float(getattr(self.cfg, "xgb_colsample_bytree", 0.8)),
            reg_alpha=float(getattr(self.cfg, "xgb_reg_alpha", 0.0)),
            reg_lambda=float(getattr(self.cfg, "xgb_reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(getattr(self.cfg, "random_seed", 42)),
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )

    def _build_estimator(self):
        if self.model_type == "xgboost":
            return self._build_xgb()
        return self._build_lgbm()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_with_val: bool = True,
    ) -> "MultiLabelTreeModel":
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.int8)

        if train_with_val and X_val is not None and y_val is not None and len(X_val) > 0:
            X_fit = np.concatenate([X_tr, np.asarray(X_val, dtype=np.float32)], axis=0)
            y_fit = np.concatenate([y_tr, np.asarray(y_val, dtype=np.int8)], axis=0)
        else:
            X_fit = X_tr
            y_fit = y_tr

        self.estimators = []
        for cls_idx in range(y_fit.shape[1]):
            y_cls = y_fit[:, cls_idx]
            unique = np.unique(y_cls)
            if unique.size < 2:
                est = _ConstantBinaryEstimator(constant=int(unique[0]))
                est.fit(X_fit, y_cls)
                self.estimators.append(est)
                continue

            est = self._build_estimator()
            est.fit(X_fit, y_cls)
            self.estimators.append(est)

        self.meta.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.meta.fitted:
            raise RuntimeError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        cols = []
        for est in self.estimators:
            prob = est.predict_proba(X_arr)
            prob = np.asarray(prob)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                cols.append(prob[:, 1].astype(np.float32, copy=False))
            else:
                cols.append(prob.reshape(-1).astype(np.float32, copy=False))
        return np.column_stack(cols).astype(np.float32, copy=False)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= float(threshold)).astype(np.int8, copy=False)
