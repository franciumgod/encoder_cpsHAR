from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .domain import flatten_tensor

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _activation(name: str):
    text = str(name).strip().lower()
    if text == "relu":
        return _relu
    if text == "tanh":
        return _tanh
    if text == "logistic":
        return _logistic
    return _relu


@dataclass
class EncoderMeta:
    backend: str
    latent_dim: int
    axis_mode: str
    effective_backend: str
    note: str = ""


class _EmptyEncoder:
    def fit(self, X: np.ndarray) -> "_EmptyEncoder":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.empty((len(X), 0), dtype=np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


class _PCAEncoder:
    def __init__(self, latent_dim: int, random_seed: int = 42):
        self.latent_dim = int(latent_dim)
        self.random_seed = int(random_seed)
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "_PCAEncoder":
        flat = flatten_tensor(X)
        scaled = self.scaler.fit_transform(flat)
        n_comp = max(1, min(self.latent_dim, scaled.shape[0], scaled.shape[1]))
        self.pca = PCA(n_components=n_comp, random_state=self.random_seed)
        self.pca.fit(scaled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA encoder is not fitted.")
        flat = flatten_tensor(X)
        scaled = self.scaler.transform(flat)
        return self.pca.transform(scaled).astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class _PerAxisPCAEncoder:
    def __init__(self, latent_dim: int, random_seed: int = 42):
        self.latent_dim = int(latent_dim)
        self.random_seed = int(random_seed)
        self.axis_models: List[tuple[StandardScaler, PCA]] = []
        self.axis_out_dims: List[int] = []

    def fit(self, X: np.ndarray) -> "_PerAxisPCAEncoder":
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            # fallback to global PCA when tensor shape is unexpected
            scaler = StandardScaler()
            flat = flatten_tensor(X_arr)
            flat = scaler.fit_transform(flat)
            n_comp = max(1, min(self.latent_dim, flat.shape[0], flat.shape[1]))
            pca = PCA(n_components=n_comp, random_state=self.random_seed)
            pca.fit(flat)
            self.axis_models = [(scaler, pca)]
            self.axis_out_dims = [n_comp]
            return self

        _, t, c = X_arr.shape
        axis_dim = max(1, self.latent_dim // max(1, c))
        self.axis_models = []
        self.axis_out_dims = []
        for ch in range(c):
            xch = X_arr[:, :, ch].reshape(X_arr.shape[0], t)
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(xch)
            n_comp = max(1, min(axis_dim, x_scaled.shape[0], x_scaled.shape[1]))
            pca = PCA(n_components=n_comp, random_state=self.random_seed + ch)
            pca.fit(x_scaled)
            self.axis_models.append((scaler, pca))
            self.axis_out_dims.append(n_comp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.axis_models:
            raise RuntimeError("Per-axis PCA encoder is not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3 or len(self.axis_models) == 1:
            scaler, pca = self.axis_models[0]
            flat = flatten_tensor(X_arr)
            return pca.transform(scaler.transform(flat)).astype(np.float32, copy=False)

        out = []
        for ch, (scaler, pca) in enumerate(self.axis_models):
            xch = X_arr[:, :, ch].reshape(X_arr.shape[0], X_arr.shape[1])
            zch = pca.transform(scaler.transform(xch)).astype(np.float32, copy=False)
            out.append(zch)
        return np.hstack(out).astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class _SklearnMLPAutoEncoder:
    """
    Uses MLPRegressor as a lightweight autoencoder and returns bottleneck activations.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        batch_size: int,
        lr: float,
        epochs: int,
        random_seed: int = 42,
    ):
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.random_seed = int(random_seed)
        self.scaler = StandardScaler()
        self.model: Optional[MLPRegressor] = None

    def fit(self, X: np.ndarray) -> "_SklearnMLPAutoEncoder":
        flat = flatten_tensor(X)
        scaled = self.scaler.fit_transform(flat)
        latent = max(1, min(self.latent_dim, scaled.shape[1]))
        hidden = max(latent, self.hidden_dim)
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden, latent, hidden),
            activation="relu",
            solver="adam",
            alpha=1e-5,
            batch_size=min(max(16, self.batch_size), max(16, len(scaled))),
            learning_rate_init=self.lr,
            max_iter=max(1, self.epochs),
            random_state=self.random_seed,
            shuffle=True,
            early_stopping=False,
            verbose=False,
        )
        self.model.fit(scaled, scaled)
        return self

    def _encode_scaled(self, scaled: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLP encoder is not fitted.")
        act = _activation(self.model.activation)
        a = scaled
        # layer 0: input -> hidden1
        a = act(a @ self.model.coefs_[0] + self.model.intercepts_[0])
        # layer 1: hidden1 -> bottleneck
        a = act(a @ self.model.coefs_[1] + self.model.intercepts_[1])
        return a.astype(np.float32, copy=False)

    def transform(self, X: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(flatten_tensor(X))
        return self._encode_scaled(scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        scaled = self.scaler.transform(flatten_tensor(X))
        return self._encode_scaled(scaled)


if torch is not None:  # pragma: no cover
    class _SEBlock1D(nn.Module):
        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            hidden = max(1, channels // max(1, reduction))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            w = self.pool(x).squeeze(-1)
            w = self.fc(w).unsqueeze(-1)
            return x * w


    class _ResBlock1D(nn.Module):
        def __init__(self, channels: int, kernel_size: int = 3, use_se: bool = False, dropout: float = 0.0):
            super().__init__()
            pad = kernel_size // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
            self.bn2 = nn.BatchNorm1d(channels)
            self.act = nn.ReLU(inplace=True)
            self.se = _SEBlock1D(channels) if use_se else nn.Identity()
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)
            out = self.drop(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.se(out)
            out = out + x
            out = self.act(out)
            return out


    class _ConvAutoEncoder(nn.Module):
        def __init__(
            self,
            in_channels: int,
            time_steps: int,
            latent_dim: int,
            channels: list[int],
            kernels: list[int],
            hidden_dim: int,
            use_residual: bool,
            use_se: bool,
            dropout: float,
        ):
            super().__init__()
            layers = []
            c_in = in_channels
            for i, c_out in enumerate(channels):
                k = kernels[min(i, len(kernels) - 1)]
                pad = k // 2
                layers.extend(
                    [
                        nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, bias=False),
                        nn.BatchNorm1d(c_out),
                        nn.ReLU(inplace=True),
                    ]
                )
                if use_residual:
                    layers.append(_ResBlock1D(c_out, kernel_size=k, use_se=use_se, dropout=dropout))
                c_in = c_out

            self.feature = nn.Sequential(*layers) if layers else nn.Identity()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc_hidden = nn.Linear(c_in, max(1, hidden_dim))
            self.fc_latent = nn.Linear(max(1, hidden_dim), max(1, latent_dim))
            self.fc_decode = nn.Linear(max(1, latent_dim), in_channels * time_steps)
            self.time_steps = int(time_steps)
            self.in_channels = int(in_channels)

        def encode(self, x):
            # x: (n, t, c) -> (n, c, t)
            x = x.transpose(1, 2)
            h = self.feature(x)
            h = self.pool(h).squeeze(-1)
            h = torch.relu(self.fc_hidden(h))
            z = self.fc_latent(h)
            return z

        def forward(self, x):
            z = self.encode(x)
            rec = self.fc_decode(z)
            rec = rec.view(-1, self.time_steps, self.in_channels)
            return rec, z


    class _TorchConvEncoder:
        def __init__(
            self,
            backend: str,
            axis_mode: str,
            latent_dim: int,
            hidden_dim: int,
            channels: list[int],
            kernels: list[int],
            use_se: bool,
            dropout: float,
            epochs: int,
            batch_size: int,
            lr: float,
            random_seed: int,
        ):
            self.backend = str(backend).lower()
            self.axis_mode = str(axis_mode).lower()
            self.latent_dim = int(latent_dim)
            self.hidden_dim = int(hidden_dim)
            self.channels = [int(x) for x in channels]
            self.kernels = [int(x) for x in kernels]
            self.use_se = bool(use_se)
            self.dropout = float(dropout)
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
            self.lr = float(lr)
            self.random_seed = int(random_seed)

            self.model: Optional[_ConvAutoEncoder] = None
            self.axis_models: List[_ConvAutoEncoder] = []
            self.mean: Optional[np.ndarray] = None
            self.std: Optional[np.ndarray] = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _normalize(self, X: np.ndarray) -> np.ndarray:
            if self.mean is None or self.std is None:
                raise RuntimeError("Encoder normalization stats are not fitted.")
            return (X - self.mean) / self.std

        def _fit_one(self, model: _ConvAutoEncoder, X_np: np.ndarray) -> _ConvAutoEncoder:
            ds = TensorDataset(torch.from_numpy(X_np))
            loader = DataLoader(
                ds,
                batch_size=min(max(16, self.batch_size), max(16, len(ds))),
                shuffle=True,
                drop_last=False,
            )
            optim = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            model.train()
            for _ in range(max(1, self.epochs)):
                for (batch_x,) in loader:
                    batch_x = batch_x.to(self.device)
                    optim.zero_grad()
                    rec, _ = model(batch_x)
                    loss = criterion(rec, batch_x)
                    loss.backward()
                    optim.step()
            return model

        def fit(self, X: np.ndarray) -> "_TorchConvEncoder":
            X_arr = np.asarray(X, dtype=np.float32)
            if X_arr.ndim != 3:
                raise ValueError("Torch CNN encoder expects 3D input (n, time, channels).")
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

            self.mean = np.mean(X_arr, axis=(0, 1), keepdims=True).astype(np.float32, copy=False)
            self.std = np.std(X_arr, axis=(0, 1), keepdims=True).astype(np.float32, copy=False) + 1e-6
            X_norm = self._normalize(X_arr)

            n, t, c = X_norm.shape
            use_res = self.backend == "rescnn1d"
            if self.axis_mode == "per_axis":
                axis_dim = max(1, self.latent_dim // max(1, c))
                self.axis_models = []
                for ch in range(c):
                    model = _ConvAutoEncoder(
                        in_channels=1,
                        time_steps=t,
                        latent_dim=axis_dim,
                        hidden_dim=self.hidden_dim,
                        channels=self.channels,
                        kernels=self.kernels,
                        use_residual=use_res,
                        use_se=self.use_se,
                        dropout=self.dropout,
                    ).to(self.device)
                    xch = X_norm[:, :, ch:ch + 1]
                    self.axis_models.append(self._fit_one(model, xch))
                self.model = None
                return self

            self.model = _ConvAutoEncoder(
                in_channels=c,
                time_steps=t,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                channels=self.channels,
                kernels=self.kernels,
                use_residual=use_res,
                use_se=self.use_se,
                dropout=self.dropout,
            ).to(self.device)
            self._fit_one(self.model, X_norm)
            self.axis_models = []
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.float32)
            X_norm = self._normalize(X_arr)
            with torch.no_grad():
                if self.axis_mode == "per_axis":
                    if not self.axis_models:
                        raise RuntimeError("Per-axis torch encoder is not fitted.")
                    outs = []
                    for ch, model in enumerate(self.axis_models):
                        model.eval()
                        xt = torch.from_numpy(X_norm[:, :, ch:ch + 1]).to(self.device)
                        z = model.encode(xt).cpu().numpy().astype(np.float32, copy=False)
                        outs.append(z)
                    return np.hstack(outs).astype(np.float32, copy=False)

                if self.model is None:
                    raise RuntimeError("Torch CNN encoder is not fitted.")
                self.model.eval()
                xt = torch.from_numpy(X_norm).to(self.device)
                z = self.model.encode(xt).cpu().numpy()
                return z.astype(np.float32, copy=False)

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            self.fit(X)
            return self.transform(X)


class EncoderProjector:
    """
    Unified encoder interface.
    - none: no encoder features
    - mlp: sklearn MLP autoencoder bottleneck
    - cnn1d/rescnn1d: torch autoencoder when torch available, otherwise fallback to PCA
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = str(getattr(cfg, "encoder_backend", "mlp")).strip().lower()
        self.axis_mode = str(getattr(cfg, "encoder_axis_mode", "joint")).strip().lower()
        if self.axis_mode not in {"joint", "per_axis"}:
            self.axis_mode = "joint"
        self.use_encoder = bool(getattr(cfg, "use_encoder", True)) and self.backend != "none"
        self.impl = None
        self.meta = EncoderMeta(
            backend=self.backend,
            latent_dim=int(getattr(cfg, "encoder_output_dim", 128)),
            axis_mode=self.axis_mode,
            effective_backend="none",
            note="",
        )
        self._init_impl()

    def _init_impl(self):
        if not self.use_encoder:
            self.impl = _EmptyEncoder()
            self.meta.effective_backend = "none"
            self.meta.note = "Encoder disabled."
            return

        latent_dim = int(getattr(self.cfg, "encoder_output_dim", 128))
        hidden_dim = int(getattr(self.cfg, "encoder_hidden_dim", 256))
        batch_size = int(getattr(self.cfg, "encoder_batch_size", 128))
        lr = float(getattr(self.cfg, "encoder_lr", 1e-3))
        epochs = int(getattr(self.cfg, "encoder_epochs", 15))
        seed = int(getattr(self.cfg, "random_seed", 42))

        if self.backend == "mlp":
            self.impl = _SklearnMLPAutoEncoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                random_seed=seed,
            )
            self.meta.effective_backend = "mlp"
            return

        if self.backend in {"cnn1d", "rescnn1d"}:
            if torch is not None:
                self.impl = _TorchConvEncoder(
                    backend=self.backend,
                    axis_mode=self.axis_mode,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    channels=list(getattr(self.cfg, "encoder_channels", [64, 128])),
                    kernels=list(getattr(self.cfg, "encoder_kernels", [3, 5, 7])),
                    use_se=bool(getattr(self.cfg, "encoder_use_se", False)),
                    dropout=float(getattr(self.cfg, "encoder_dropout", 0.1)),
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    random_seed=seed,
                )
                self.meta.effective_backend = self.backend
                return

            if self.axis_mode == "per_axis":
                self.impl = _PerAxisPCAEncoder(latent_dim=latent_dim, random_seed=seed)
                self.meta.effective_backend = "per_axis_pca_fallback"
            else:
                self.impl = _PCAEncoder(latent_dim=latent_dim, random_seed=seed)
                self.meta.effective_backend = "pca_fallback"
            self.meta.note = "Torch unavailable. Requested CNN encoder fallback to PCA."
            return

        self.impl = _PCAEncoder(latent_dim=latent_dim, random_seed=seed)
        self.meta.effective_backend = "pca"
        self.meta.note = f"Unknown encoder backend '{self.backend}', fallback to PCA."

    def fit(self, X: np.ndarray) -> "EncoderProjector":
        self.impl.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.impl.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.impl.fit_transform(X)
