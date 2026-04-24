from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


class _ConstantBinaryEstimator:
    def __init__(self, constant: int):
        self.constant = int(constant)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        pos = np.full((n,), float(self.constant), dtype=np.float32)
        return pos


@dataclass
class ModelMeta:
    model_type: str
    n_classes: int
    fitted: bool = False


class MultiLabelTreeModel:
    """
    Multi-label OvR model with native XGBoost API (xgb.train + DMatrix).
    """

    def __init__(self, cfg, label_cols: List[str]):
        self.cfg = cfg
        self.label_cols = list(label_cols)
        self.estimators: List[object] = []
        self.best_iterations: List[int] = []
        self.meta = ModelMeta(model_type="xgboost_native", n_classes=len(self.label_cols), fitted=False)
        if xgb is None:
            raise RuntimeError("xgboost is not available in the current environment.")

    def _xgb_params(self) -> dict:
        device = str(getattr(self.cfg, "xgb_device", "cuda")).strip().lower()
        return {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": device,
            "eta": float(getattr(self.cfg, "xgb_learning_rate", 0.05)),
            "max_depth": int(getattr(self.cfg, "xgb_max_depth", 6)),
            "subsample": float(getattr(self.cfg, "xgb_subsample", 0.8)),
            "colsample_bytree": float(getattr(self.cfg, "xgb_colsample_bytree", 0.8)),
            "reg_alpha": float(getattr(self.cfg, "xgb_reg_alpha", 0.0)),
            "reg_lambda": float(getattr(self.cfg, "xgb_reg_lambda", 1.0)),
            "eval_metric": "logloss",
            "seed": int(getattr(self.cfg, "random_seed", 42)),
            "verbosity": 0,
        }

    @staticmethod
    def _release_host_memory() -> None:
        gc.collect()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        train_with_val: bool = False,
    ) -> "MultiLabelTreeModel":
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.int8)

        X_va = np.asarray(X_val, dtype=np.float32) if X_val is not None else None
        y_va = np.asarray(y_val, dtype=np.int8) if y_val is not None else None

        params = self._xgb_params()
        num_boost_round = int(getattr(self.cfg, "xgb_n_estimators", 2000))
        early_stopping_rounds = int(getattr(self.cfg, "xgb_early_stopping_rounds", 100))

        self.estimators = []
        self.best_iterations = []

        for cls_idx in range(y_tr.shape[1]):
            y_cls = y_tr[:, cls_idx]
            uniq = np.unique(y_cls)
            if uniq.size < 2:
                const_model = _ConstantBinaryEstimator(constant=int(uniq[0]))
                const_model.fit(X_tr, y_cls)
                self.estimators.append(const_model)
                self.best_iterations.append(0)
                del y_cls, uniq
                self._release_host_memory()
                continue

            dtrain = xgb.DMatrix(X_tr, label=y_cls)
            dval = None

            if X_va is not None and y_va is not None and len(X_va) > 0 and not train_with_val:
                dval = xgb.DMatrix(X_va, label=y_va[:, cls_idx])
                evals = [(dtrain, "train"), (dval, "valid")]
            else:
                evals = [(dtrain, "train")]

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds if len(evals) > 1 else None,
                verbose_eval=False,
            )
            best_it = int(getattr(booster, "best_iteration", num_boost_round - 1))
            if best_it < 0:
                best_it = num_boost_round - 1
            self.estimators.append(booster)
            self.best_iterations.append(best_it)

            del booster, dtrain, dval, evals, y_cls, uniq
            self._release_host_memory()

        self.meta.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.meta.fitted:
            raise RuntimeError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        dtest = xgb.DMatrix(X_arr)
        probs = []

        for est, best_it in zip(self.estimators, self.best_iterations):
            if isinstance(est, _ConstantBinaryEstimator):
                p = est.predict_proba(X_arr)
            else:
                p = est.predict(dtest, iteration_range=(0, int(best_it) + 1))
            probs.append(np.asarray(p, dtype=np.float32).reshape(-1))
        del dtest
        self._release_host_memory()
        return np.column_stack(probs).astype(np.float32, copy=False)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= float(threshold)).astype(np.int8, copy=False)

    def release(self) -> None:
        # Drop booster/estimator references aggressively between folds.
        self.estimators.clear()
        self.best_iterations = []
        self.meta.fitted = False
        self._release_host_memory()
