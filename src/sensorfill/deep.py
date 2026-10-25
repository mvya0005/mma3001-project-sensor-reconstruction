"""Deep-learning imputers in PyTorch (unit topics 5.4, 10.2, 10.3).

Designed to train on a Colab GPU (``device="cuda"``) but runs on CPU too.

* :class:`BiGRUImputer` - bidirectional GRU that reads a window around each
  gap (observed values, the mask, occupancy and calendar channels) and
  predicts the residual from linear interpolation at every step.
* **Physics-informed variant** (``physics_weight > 0``) - adds the residual of
  the CO2 mass balance ``dC/dt - [g N - lambda (C - C_out)]`` evaluated with
  central finite differences on the network output, with ``g``, ``lambda``
  and ``C_out`` learned jointly (positive by construction). This is the PINN
  idea from 10.3 applied as a soft constraint on a data-driven model.
* :class:`NeuralODE` - learns the right-hand side ``dC/dt = f_theta(C, N,
  time-of-day)`` and integrates it with a differentiable RK4 scheme (10.2).
  A gap is filled by integrating from its left boundary; the end mismatch is
  removed with a linear correction so the fill joins the right boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - torch is optional for the classical methods
    raise ImportError("sensorfill.deep needs PyTorch: pip install torch") from exc

from .interpolation import fill_linear, gap_spans
from .masking import block_mask


def get_device(prefer: str | None = None) -> torch.device:
    """Return ``cuda`` when available (Colab GPU runtime), else ``cpu``."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch (CPU and CUDA) for repeatable training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calendar_channels(ts) -> np.ndarray:
    """Time-of-day and day-of-week encoded as sin/cos, shape ``(n, 4)``."""
    ts = pd.DatetimeIndex(ts)
    tod = (ts.hour + ts.minute / 60.0).to_numpy() / 24.0
    dow = ts.dayofweek.to_numpy() / 7.0
    return np.column_stack(
        [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)]
    ).astype(np.float32)


@dataclass
class Scaler:
    """Standardisation constants computed on the training period only."""

    y_mean: float
    y_std: float
    occ_mean: float = 0.0
    occ_std: float = 1.0

    @classmethod
    def fit(cls, y, occ=None) -> "Scaler":
        """Estimate mean and standard deviation (ignoring NaN)."""
        o = np.zeros(1) if occ is None else np.asarray(occ, dtype=float)
        return cls(float(np.nanmean(y)), float(np.nanstd(y)) or 1.0, float(np.nanmean(o)), float(np.nanstd(o)) or 1.0)


def _inputs(y_obs: np.ndarray, occ: np.ndarray | None, cal: np.ndarray, sc: Scaler) -> np.ndarray:
    """Stack network input channels for one window: ``[y_lin, mask, occ, cal...]``."""
    mask = (~np.isnan(y_obs)).astype(np.float32)
    y_lin = (fill_linear(y_obs) - sc.y_mean) / sc.y_std
    o = np.zeros_like(y_lin) if occ is None else (np.nan_to_num(occ) - sc.occ_mean) / sc.occ_std
    return np.column_stack([y_lin, mask, o, cal]).astype(np.float32)


class _GRUNet(nn.Module):
    """Bidirectional GRU encoder with a linear read-out per time step."""

    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (batch, time, channels)
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


@dataclass
class TrainLog:
    """Loss history and timing of a training run."""

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    seconds: float = 0.0
    best_epoch: int = 0


class BiGRUImputer:
    """Window-based bidirectional-GRU gap filler with optional physics loss.

    Args:
        window: Window length in steps (432 = 3 days at 10 min). Gaps must be
            shorter than ``window - 2 * context`` to be filled by one window.
        hidden: GRU hidden size per direction.
        layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers.
        physics_weight: Weight of the CO2 mass-balance residual loss (0 = off).
        dt_hours: Grid spacing in hours (for the physics residual).
        ode_init: Initial ``(g, lambda, C_out)`` for the learnable physics
            parameters, e.g. from :func:`sensorfill.ode.fit_output_error`.
        device: ``"cuda"``, ``"cpu"`` or None for automatic.
        seed: Random seed.
    """

    def __init__(
        self,
        window: int = 432,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        physics_weight: float = 0.0,
        dt_hours: float = 1 / 6,
        ode_init: tuple[float, float, float] = (5.0, 1.0, 420.0),
        device: str | None = None,
        seed: int = 0,
    ):
        set_seed(seed)
        self.window, self.dt = window, dt_hours
        self.device = get_device(device)
        self.net = _GRUNet(3 + 4, hidden, layers, dropout).to(self.device)
        self.physics_weight = physics_weight
        g, lam, cout = ode_init
        # Log-parameterisation keeps g and lambda positive during optimisation.
        self.log_g = nn.Parameter(torch.tensor(np.log(max(g, 1e-3)), dtype=torch.float32, device=self.device))
        self.log_lam = nn.Parameter(torch.tensor(np.log(max(lam, 1e-3)), dtype=torch.float32, device=self.device))
        self.c_out = nn.Parameter(torch.tensor(float(cout), dtype=torch.float32, device=self.device))
        self.sc: Scaler | None = None
        self.log = TrainLog()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ batching
    def _sample_batch(self, y, occ, cal, starts, gap_choices, batch: int):
        W = self.window
        X, T, M, O = [], [], [], []
        for s in self.rng.choice(starts, size=batch):
            yt = y[s : s + W].copy()
            obs = ~np.isnan(yt)
            L = int(self.rng.choice(gap_choices))
            try:
                hide = block_mask(obs, L, 1, self.rng, margin=2, min_observed=0.8)
            except ValueError:
                hide = np.zeros(W, dtype=bool)
            yo = yt.copy()
            yo[hide] = np.nan
            if np.count_nonzero(~np.isnan(yo)) < 2:
                continue
            X.append(_inputs(yo, None if occ is None else occ[s : s + W], cal[s : s + W], self.sc))
            T.append(np.nan_to_num((yt - self.sc.y_mean) / self.sc.y_std))
            M.append(hide.astype(np.float32))
            O.append(np.zeros(W, np.float32) if occ is None else np.nan_to_num(occ[s : s + W]).astype(np.float32))
        f = lambda a: torch.tensor(np.stack(a), device=self.device)
        return f(X), f(T).float(), f(M), f(O)

    def _physics_residual(self, pred_norm, occ):
        """Mean-square CO2 mass-balance residual (ppm/h) of a predicted batch."""
        C = pred_norm * self.sc.y_std + self.sc.y_mean
        dC = (C[:, 2:] - C[:, :-2]) / (2 * self.dt)
        rhs = torch.exp(self.log_g) * occ[:, 1:-1] - torch.exp(self.log_lam) * (C[:, 1:-1] - self.c_out)
        return ((dC - rhs) / self.sc.y_std).pow(2).mean()

    def _forward(self, X):
        # Network predicts the residual from the (normalised) linear baseline in channel 0.
        return X[..., 0] + self.net(X)

    # ------------------------------------------------------------ training
    def fit(
        self,
        y_train,
        ts_train,
        occ_train=None,
        y_val=None,
        ts_val=None,
        occ_val=None,
        epochs: int = 40,
        steps_per_epoch: int = 100,
        batch: int = 64,
        lr: float = 2e-3,
        gap_choices=(1, 3, 6, 12, 24, 48, 96, 144),
        patience: int = 8,
    ) -> TrainLog:
        """Train with randomly placed artificial gaps; early-stop on validation loss.

        The data loss is the mean-squared error on hidden points only. Returns
        the :class:`TrainLog` and restores the best-validation weights.
        """
        y_train = np.asarray(y_train, dtype=float)
        self.sc = Scaler.fit(y_train, occ_train)
        cal_tr = calendar_channels(ts_train)
        W = self.window
        starts = np.arange(0, len(y_train) - W)
        params = list(self.net.parameters())
        if self.physics_weight > 0:
            params += [self.log_g, self.log_lam, self.c_out]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        if y_val is not None:
            cal_va = calendar_channels(ts_val)
            vstarts = np.arange(0, len(y_val) - W, W // 4)
            vrng_state = np.random.default_rng(123)
            saved, self.rng = self.rng, vrng_state
            Xv, Tv, Mv, Ov = self._sample_batch(np.asarray(y_val, float), occ_val, cal_va, vstarts, gap_choices, 256)
            self.rng = saved
        best, best_state, bad = np.inf, None, 0
        t0 = time.perf_counter()
        for ep in range(epochs):
            self.net.train()
            tot = 0.0
            for _ in range(steps_per_epoch):
                X, T, M, O = self._sample_batch(y_train, occ_train, cal_tr, starts, gap_choices, batch)
                pred = self._forward(X)
                loss = ((pred - T) ** 2 * M).sum() / M.sum().clamp(min=1)
                if self.physics_weight > 0:
                    loss = loss + self.physics_weight * self._physics_residual(pred, O)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)  # guards against exploding RNN gradients
                opt.step()
                tot += loss.item()
            sched.step()
            self.log.train_loss.append(tot / steps_per_epoch)
            if y_val is not None:
                self.net.eval()
                with torch.no_grad():
                    pv = self._forward(Xv)
                    vl = (((pv - Tv) ** 2 * Mv).sum() / Mv.sum()).item()
                self.log.val_loss.append(vl)
                if vl < best - 1e-6:
                    best, bad, self.log.best_epoch = vl, 0, ep
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.log.seconds = time.perf_counter() - t0
        return self.log

    @property
    def physics_params(self) -> dict:
        """Current learned ``g`` [ppm/(person h)], ``lambda`` [1/h] and ``C_out`` [ppm]."""
        with torch.no_grad():
            return {"g": float(torch.exp(self.log_g)), "lam": float(torch.exp(self.log_lam)), "c_out": float(self.c_out)}

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def impute(self, y, ts, occ=None) -> np.ndarray:
        """Fill every interior gap using a window centred on it.

        Gaps longer than the window minus 16 steps of context, and edge gaps,
        fall back to linear interpolation (reported, not hidden).
        """
        if self.sc is None:
            raise RuntimeError("call fit() first")
        self.net.eval()
        y = np.asarray(y, dtype=float)
        out = fill_linear(y)
        cal = calendar_channels(ts)
        W, n = self.window, len(y)
        jobs = []
        for a, b in gap_spans(np.isnan(y)):
            if a == 0 or b == n or (b - a) > W - 16:
                continue
            s = int(np.clip((a + b) // 2 - W // 2, 0, n - W))
            jobs.append((a, b, s))
        for k in range(0, len(jobs), 256):
            chunk = jobs[k : k + 256]
            X = np.stack([_inputs(y[s : s + W], None if occ is None else occ[s : s + W], cal[s : s + W], self.sc) for _, _, s in chunk])
            pred = self._forward(torch.tensor(X, device=self.device)).cpu().numpy() * self.sc.y_std + self.sc.y_mean
            for (a, b, s), p in zip(chunk, pred):
                out[a:b] = p[a - s : b - s]
        return out

    def n_parameters(self) -> int:
        """Number of trainable network weights."""
        return sum(p.numel() for p in self.net.parameters())

    def flops_per_window(self) -> int:
        """Approximate forward-pass FLOPs for one window (multiply-add = 2 FLOPs).

        A GRU layer costs ``3 * 2 * (h*i + h*h)`` multiply-adds per step and
        direction; the head adds ``2*(2h*h + h)``.
        """
        h, W, flops, i = self.net.gru.hidden_size, self.window, 0, self.net.gru.input_size
        for layer in range(self.net.gru.num_layers):
            inp = i if layer == 0 else 2 * h
            flops += 2 * 2 * 3 * (h * inp + h * h) * W  # 2 directions, 3 gates, 2 flops per MAC
        flops += 2 * (2 * h * h + h) * W
        return int(flops)


# ------------------------------------------------------------- Neural ODE
class _ODEFunc(nn.Module):
    """MLP right-hand side ``f(C, N, tod)`` returning dC/dt in normalised units per hour."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, c, u):  # c: (B,), u: (B, 3) = [occ, tod_sin, tod_cos]
        return self.net(torch.cat([c[:, None], u], dim=1)).squeeze(-1)


def rk4_rollout(func, c0, U, dt):
    """Differentiable classical RK4 over a sequence of inputs.

    Inputs are held at their left-endpoint value for the half step and use the
    next sample for the full step (linear interpolation at the midpoint).

    Args:
        func: Module ``f(c, u)``.
        c0: Initial states, shape ``(B,)``.
        U: Exogenous inputs, shape ``(B, T, k)``.
        dt: Step in hours.

    Returns:
        States, shape ``(B, T)`` with ``out[:, 0] = c0``.
    """
    cs = [c0]
    c = c0
    for k in range(U.shape[1] - 1):
        u0, u1 = U[:, k], U[:, k + 1]
        um = 0.5 * (u0 + u1)
        k1 = func(c, u0)
        k2 = func(c + 0.5 * dt * k1, um)
        k3 = func(c + 0.5 * dt * k2, um)
        k4 = func(c + dt * k3, u1)
        c = c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        cs.append(c)
    return torch.stack(cs, dim=1)


class NeuralODE:
    """Learned CO2 dynamics integrated with RK4.

    Args:
        horizon: Training segment length in steps (36 = 6 h).
        hidden: Width of the MLP.
        dt_hours: Grid spacing.
        device: Torch device or None.
        seed: Random seed.
    """

    def __init__(self, horizon: int = 36, hidden: int = 64, dt_hours: float = 1 / 6, device=None, seed: int = 0):
        set_seed(seed)
        self.h, self.dt = horizon, dt_hours
        self.device = get_device(device)
        self.func = _ODEFunc(hidden).to(self.device)
        self.sc: Scaler | None = None
        self.log = TrainLog()
        self.rng = np.random.default_rng(seed)

    def _U(self, occ, ts):
        cal = calendar_channels(ts)[:, :2]
        o = (np.nan_to_num(occ) - self.sc.occ_mean) / self.sc.occ_std
        return np.column_stack([o, cal]).astype(np.float32)

    def fit(self, y, ts, occ, epochs: int = 30, steps_per_epoch: int = 100, batch: int = 128, lr: float = 3e-3) -> TrainLog:
        """Fit by multiple shooting on random fully observed segments of length ``horizon``."""
        y = np.asarray(y, dtype=float)
        self.sc = Scaler.fit(y, occ)
        yn = (y - self.sc.y_mean) / self.sc.y_std
        U = self._U(occ, ts)
        ok = ~np.isnan(yn)
        c = np.concatenate([[0], np.cumsum(ok)])
        starts = np.flatnonzero((c[self.h :] - c[: -self.h]) == self.h)
        if starts.size == 0:
            raise ValueError("no fully observed segment of the requested horizon")
        opt = torch.optim.Adam(self.func.parameters(), lr=lr)
        t0 = time.perf_counter()
        for _ in range(epochs):
            tot = 0.0
            for _ in range(steps_per_epoch):
                s = self.rng.choice(starts, size=batch)
                idx = s[:, None] + np.arange(self.h)[None, :]
                Y = torch.tensor(yn[idx], dtype=torch.float32, device=self.device)
                Ub = torch.tensor(U[idx], device=self.device)
                pred = rk4_rollout(self.func, Y[:, 0], Ub, self.dt)
                loss = ((pred - Y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            self.log.train_loss.append(tot / steps_per_epoch)
        self.log.seconds = time.perf_counter() - t0
        return self.log

    @torch.no_grad()
    def impute(self, y, ts, occ, correct_end: bool = True) -> np.ndarray:
        """Integrate across each interior gap from its left boundary.

        Args:
            y: Series with NaNs.
            ts: Timestamps.
            occ: Occupancy on the grid.
            correct_end: Remove the mismatch with the right boundary by adding a
                linearly growing correction (a cheap two-point BVP fix).

        Returns:
            Filled copy (edge gaps use linear/hold).
        """
        y = np.asarray(y, dtype=float)
        out = fill_linear(y)
        U = self._U(occ, ts)
        for a, b in gap_spans(np.isnan(y)):
            if a == 0 or b == len(y):
                continue
            seg = slice(a - 1, b + 1)
            c0 = torch.tensor([(y[a - 1] - self.sc.y_mean) / self.sc.y_std], dtype=torch.float32, device=self.device)
            Ub = torch.tensor(U[seg][None], device=self.device)
            traj = rk4_rollout(self.func, c0, Ub, self.dt)[0].cpu().numpy() * self.sc.y_std + self.sc.y_mean
            if correct_end:
                traj = traj + (y[b] - traj[-1]) * np.linspace(0, 1, traj.size)
            out[a:b] = traj[1:-1]
        return out
