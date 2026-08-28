"""
Which AdaMLP block trains best? Controlled A/B on real Hinode spectra.

All four variants have identical parameter sets, so each run starts from
the *same* initial weights and differs only in the forward pass. Two
changes are isolated separately:
  norm : apply LayerNorm to x before the shift/scale modulation
  gate : drop the `+1.0` offset, so the block starts as an exact identity
"""

import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
from inspectorch import DensityEstimator
from inspectorch.velocity_models import AdaMLPBlock


class Variant(AdaMLPBlock):
    NORM = False
    GATE_OFFSET = True

    def __init__(self, hidden_features, cond_dim, **k):
        super().__init__(hidden_features, cond_dim, **k)
        self.ln = (
            nn.LayerNorm(hidden_features, elementwise_affine=False)
            if self.NORM
            else None
        )

    def forward(self, x, cond):
        shift, scale, gate = self.ada_ln(cond).chunk(3, dim=-1)
        if self.GATE_OFFSET:
            gate = gate + 1.0
        h = self.ln(x) if self.ln is not None else x
        return x + gate * self.block((scale + 1) * h + shift)


VARIANTS = {}
for name, norm, off in [
    ("shipped", False, True),
    ("norm_only", True, True),
    ("identity_gate", False, False),
    ("adaln_zero", True, False),
]:
    VARIANTS[name] = type(name, (Variant,), {"NORM": norm, "GATE_OFFSET": off})

# ---- real data: Hinode spectra, 30 wavelengths ----
with np.load("example/hinode_data.npz") as f:
    cube = f["data"] / 186.0
spec = torch.tensor(
    cube.transpose(0, 2, 1).reshape(-1, cube.shape[1]), dtype=torch.float32
)
g = torch.Generator().manual_seed(0)
spec = spec[torch.randperm(spec.shape[0], generator=g)]
train_x, test_x = spec[:12000], spec[12000:15000]


class DS(torch.utils.data.Dataset):
    def __init__(self, patches, mean, std):
        self.patches, self.y_mean, self.y_std = patches, mean, std

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index]

    def normalized_patches(self):
        return (self.patches - self.y_mean) / self.y_std


mean, std = train_x.mean(0), train_x.std(0)
std[std == 0] = 1.0
train_ds, test_ds = DS(train_x, mean, std), DS(test_x, mean, std)
loader = torch.utils.data.DataLoader(
    train_ds, batch_size=1000, shuffle=True, generator=torch.Generator().manual_seed(1)
)

DEPTH, EPOCHS = int(sys.argv[1]), int(sys.argv[2])
print(
    f"depth={DEPTH} epochs={EPOCHS} train={len(train_ds)} test={len(test_ds)} dim={spec.shape[1]}"
)
print(f"\n{'variant':>14} {'seed':>5} {'train loss':>11} {'test logL':>11} {'sec':>6}")
results = {}
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    ref = DensityEstimator(type="flow_matching_sbi")
    ref.create_flow(input_size=spec.shape[1], num_layers=DEPTH, hidden_features=64)
    init_state = {
        k: v.clone() for k, v in ref.backend.velocity_model.state_dict().items()
    }

    for name, cls in VARIANTS.items():
        torch.manual_seed(seed)
        m = DensityEstimator(type="flow_matching_sbi")
        m.create_flow(input_size=spec.shape[1], num_layers=DEPTH, hidden_features=64)
        net = m.backend.velocity_model
        for i in range(1, len(net.layers)):
            old = net.layers[i]
            new = cls(old.block[0].in_features, old.ada_ln[0].in_features)
            new.load_state_dict(old.state_dict(), strict=False)
            net.layers[i] = new
        net.load_state_dict(init_state, strict=False)  # identical starting weights

        t0 = time.time()
        torch.manual_seed(1000 + seed)
        m.train_flow(loader, learning_rate=1e-3, num_epochs=EPOCHS, device="cpu")
        torch.manual_seed(7)
        lp = m.log_prob(
            test_ds,
            device="cpu",
            batch_size=3000,
            solver_method="euler",
            steps=100,
            step_size=0.01,
            exact_divergence=False,
        )
        tl, ll = m.backend.training_loss[-1], float(np.mean(lp))
        results.setdefault(name, []).append(ll)
        print(f"{name:>14} {seed:>5} {tl:>11.4f} {ll:>11.3f} {time.time() - t0:>6.1f}")

print(f"\n{'variant':>14} {'mean test logL':>16} {'std':>8}   (higher is better)")
base = np.mean(results["shipped"])
for name, vals in results.items():
    d = np.mean(vals) - base
    print(
        f"{name:>14} {np.mean(vals):>16.3f} {np.std(vals):>8.3f}   {d:+.3f} vs shipped"
    )
