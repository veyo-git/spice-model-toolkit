"""Neural network surrogate model for accelerating SPICE parameter extraction.

The surrogate learns the mapping:
    (parameters, Vgs, Vds) → Ids
so that optimization can query the NN instead of running the full physics
simulation, achieving >10× speedup.

Architecture: MLP with residual connections, trained on physics model output.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from tqdm import tqdm


@dataclass
class SurrogateConfig:
    """Surrogate model hyperparameters."""

    n_params: int = 11          # Number of MOSFET parameters
    hidden_dims: Tuple[int, ...] = (256, 512, 256)
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 200
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class IVSurrogate(nn.Module):
    """NN surrogate: (params, Vgs, Vds) → log10(Ids).

    Input: [param_1, ..., param_D, Vgs, Vds]  → D + 2 dims
    Output: log10(Ids) scalar
    """

    def __init__(self, config: SurrogateConfig = None):
        super().__init__()
        cfg = config or SurrogateConfig()
        self.config = cfg
        input_dim = cfg.n_params + 2  # params + Vgs + Vds

        layers = []
        prev_dim = input_dim
        for h_dim in cfg.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.res_blocks = nn.Sequential(
            ResidualBlock(cfg.hidden_dims[-1], cfg.dropout),
            ResidualBlock(cfg.hidden_dims[-1], cfg.dropout),
        )
        self.head = nn.Linear(cfg.hidden_dims[-1], 1)

        self.to(cfg.device)

    def forward(self, x):
        """x: (batch, n_params + 2) → (batch, 1)"""
        h = self.encoder(x)
        h = self.res_blocks(h)
        return self.head(h).squeeze(-1)

    def predict(self, params, vgs, vds):
        """Predict Ids for given parameters and bias points.

        Parameters
        ----------
        params : ndarray (batch, D) or (D,)
        vgs : float or ndarray (batch,)
        vds : float or ndarray (batch,)

        Returns
        -------
        ids : ndarray (batch,)
        """
        self.eval()
        params = np.atleast_2d(np.asarray(params, dtype=np.float32))
        vgs_arr = np.atleast_1d(np.asarray(vgs, dtype=np.float32))
        vds_arr = np.atleast_1d(np.asarray(vds, dtype=np.float32))

        # Broadcast to same batch size
        batch = params.shape[0]
        if len(vgs_arr) == 1:
            vgs_arr = np.full(batch, vgs_arr[0])
        if len(vds_arr) == 1:
            vds_arr = np.full(batch, vds_arr[0])

        feat = np.column_stack([params, vgs_arr, vds_arr])
        x = torch.tensor(feat, dtype=torch.float32, device=self.config.device)

        with torch.no_grad():
            log_ids = self.forward(x)

        return 10 ** log_ids.cpu().numpy()


def train_surrogate(
    model_fn,
    param_space,
    config=None,
    n_samples=50000,
    seed=42,
    verbose=True,
):
    """Train the surrogate model on data generated from a physics model.

    Parameters
    ----------
    model_fn : callable
        Function (params_dict) → MOSFET model instance.
    param_space : dict
        Dict of param_name → (lo, hi) for uniform sampling.
    config : SurrogateConfig
    n_samples : int
        Number of training samples.
    seed : int
    verbose : bool

    Returns
    -------
    surrogate : IVSurrogate
    history : dict
    """
    cfg = config or SurrogateConfig()
    cfg.n_params = len(param_space)

    rng = np.random.RandomState(seed)

    # Generate training data
    param_names = list(param_space.keys())
    if verbose:
        print(f"Generating {n_samples} training samples...")

    X_params = np.zeros((n_samples, len(param_names)))
    for i, name in enumerate(param_names):
        lo, hi = param_space[name]
        X_params[:, i] = rng.uniform(lo, hi, n_samples)

    X_vgs = rng.uniform(-0.3, 2.5, n_samples).astype(np.float32)
    X_vds = rng.uniform(0.0, 2.5, n_samples).astype(np.float32)

    Y_log = np.zeros(n_samples, dtype=np.float32)

    # Use a fixed model instance for speed
    model = model_fn(dict(zip(param_names, X_params[0])))
    for i in tqdm(range(n_samples), desc="Generating", disable=not verbose):
        params_dict = dict(zip(param_names, X_params[i]))
        try:
            # Update model params (simpler: create new model, but slower)
            m = model_fn(params_dict)
            ids = m.ids(float(X_vgs[i]), float(X_vds[i]))
            Y_log[i] = np.log10(max(abs(ids), 1e-15))
        except Exception:
            Y_log[i] = -12.0  # floor

    # Build dataset
    X_feat = np.column_stack([X_params, X_vgs, X_vds])
    X_tensor = torch.tensor(X_feat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_log, dtype=torch.float32)

    # Train/test split
    n_train = int(n_samples * 0.85)
    idx = rng.permutation(n_samples)
    X_train, X_test = X_tensor[idx[:n_train]], X_tensor[idx[n_train:]]
    Y_train, Y_test = Y_tensor[idx[:n_train]], Y_tensor[idx[n_train:]]

    if verbose:
        print(f"Training on {n_train} samples, testing on {n_samples - n_train}")
        print(f"Device: {cfg.device}")

    # Move to device
    X_train = X_train.to(cfg.device)
    Y_train = Y_train.to(cfg.device)
    X_test = X_test.to(cfg.device)
    Y_test = Y_test.to(cfg.device)

    # Training
    surrogate = IVSurrogate(cfg)
    optimizer = torch.optim.AdamW(surrogate.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        surrogate.train()
        epoch_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(cfg.device), by.to(cfg.device)
            optimizer.zero_grad()
            pred = surrogate(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surrogate.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        epoch_loss /= len(X_train)

        surrogate.eval()
        with torch.no_grad():
            test_pred = surrogate(X_test)
            test_loss = criterion(test_pred, Y_test).item()

        train_losses.append(epoch_loss)
        test_losses.append(test_loss)
        scheduler.step(test_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:4d}/{cfg.epochs} | "
                  f"train loss: {epoch_loss:.4e} | test loss: {test_loss:.4e}")

    # Speed benchmark
    if verbose:
        _benchmark_speed(surrogate, model_fn, param_names, X_params[:100])

    return surrogate, {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "final_test_loss": test_losses[-1],
    }


def _benchmark_speed(surrogate, model_fn, param_names, sample_params):
    """Compare physics vs surrogate inference speed."""
    import time

    # Warmup
    for _ in range(10):
        _ = surrogate.predict(sample_params[0], 1.5, 1.0)

    # Physics model
    model = model_fn(dict(zip(param_names, sample_params[0])))
    t0 = time.perf_counter()
    for i in range(100):
        _ = model.ids(1.5, 1.0)
    t_physics = (time.perf_counter() - t0) / 100

    # Surrogate
    t0 = time.perf_counter()
    for i in range(100):
        _ = surrogate.predict(sample_params[0], 1.5, 1.0)
    t_surrogate = (time.perf_counter() - t0) / 100

    speedup = t_physics / t_surrogate
    print(f"\n  Speed benchmark: Physics={t_physics*1e6:.1f}µs, "
          f"Surrogate={t_surrogate*1e6:.1f}µs, Speedup={speedup:.1f}×")
