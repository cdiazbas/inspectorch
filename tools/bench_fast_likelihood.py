#!/usr/bin/env python
"""
Compare density estimators for rare-event search on the same data and budget.

Methods (any subset, comma-separated):
    gaussian    one multivariate Gaussian fitted to the training rows: the floor
                every density estimator should beat
    nf          normalizing_flow backend (maximum likelihood)
    fm          flow_matching_sbi backend (ODE + trace; --fm-steps, --fm-exact)
    meanflow    fast_flows: average-velocity map, Jacobian log-det
    falcon      fast_flows: meanflow + cycle consistency
    f2d2        fast_flows: meanflow + average-divergence head
    regflow     fast_flows: NF trained by regression on minibatch-OT pairs
    regflow_t   fast_flows: NF trained by regression on pairs from a falcon teacher
                (regflow and regflow_t sample well but their densities are not
                usable for ranking; they are left out of the default set)

For every method it reports training time, held-out NLL, evaluation speed
per number of map steps, rank agreement with a reference method (top-k
overlap, Spearman) and the recall of injected synthetic anomalies. Results
go to a JSON file with the run conditions.

Example (server):
    python tools/bench_fast_likelihood.py --data example/hinode_data.npz \\
        --divide 186 --methods gaussian,nf,fm,meanflow,falcon,f2d2 \\
        --reference fm --epochs 50 --device cuda:0 --out bench.json
"""

import argparse
import json
import platform
import subprocess
import time

import numpy as np
import torch

from inspectorch import DensityEstimator, GeneralizedPatchedDataset
from inspectorch import fast_flows as ff


class Rows(torch.utils.data.Dataset):
    """Rows of a matrix with the stats of the training split."""

    def __init__(self, patches):
        self.patches = patches
        self.y_mean = torch.nanmean(patches, 0)
        self.y_std = patches[torch.isfinite(patches).all(1)].std(0)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        return self.patches[i]

    def normalized_patches(self):
        return (self.patches - self.y_mean) / self.y_std


class SliceLoader:
    """Shuffled contiguous slices: no per-row indexing and collation."""

    def __init__(self, dataset, batch_size, seed=0):
        self.dataset, self.batch_size = dataset, batch_size
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return -(-len(self.dataset) // self.batch_size)

    def __iter__(self):
        perm = torch.randperm(len(self.dataset), generator=self.g)
        for idx in perm.split(self.batch_size):
            yield self.dataset.patches[idx]


def inject_anomalies(rows, fraction, shift, seed):
    """Shift a random subset of spectra along the feature axis (edge-padded)."""
    rng = np.random.default_rng(seed)
    n = max(1, int(fraction * len(rows)))
    idx = rng.choice(len(rows), n, replace=False)
    out = rows.clone()
    x = rows[idx]
    out[idx] = (
        torch.cat((x[:, :1].expand(-1, shift), x[:, :-shift]), 1) if shift > 0 else x
    )
    return out, idx


def rank_metrics(lp, ref, ks):
    ok = np.isfinite(lp) & np.isfinite(ref)
    lp, ref = lp[ok], ref[ok]
    from scipy.stats import spearmanr

    out = {"spearman": float(spearmanr(lp, ref).statistic)}
    for k in ks:
        a, b = set(np.argsort(lp)[:k]), set(np.argsort(ref)[:k])
        out[f"top{k}_overlap"] = len(a & b) / k
    return out


def recall(lp, idx, ks):
    """
    recall@k of the injected rows, and their median rank as a percentage of
    the held-out set (lowest log p = 0 %). Recall saturates when the anomalies
    are easy; the median rank still separates methods.
    """
    order = np.argsort(np.where(np.isfinite(lp), lp, np.inf), kind="stable")
    rank = np.empty(len(order))
    rank[order] = np.arange(len(order))
    out = {
        f"recall@{k}": len(set(order[:k]) & set(idx.tolist())) / min(k, len(idx))
        for k in ks
    }
    out["median_rank_pct"] = float(np.median(100.0 * rank[idx] / len(order)))
    return out


def timed_log_prob(fn, repeats=1):
    fn()  # warm-up (compilation, caches)
    t0 = time.perf_counter()
    for _ in range(repeats):
        lp = fn()
    return lp, (time.perf_counter() - t0) / repeats


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data",
        required=True,
        help=".npz with a 'data' cube (y, wav, x) or a .npy of rows (N, D)",
    )
    p.add_argument(
        "--divide",
        type=float,
        default=1.0,
        help="divide the data by this (example: 186)",
    )
    p.add_argument("--methods", default="gaussian,nf,fm,meanflow,falcon,f2d2")
    p.add_argument(
        "--reference",
        default="fm",
        help="method whose ranking the others are compared with",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="wall-clock budget per method (fast_flows only)",
    )
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument(
        "--steps",
        default="1,2,4,8",
        help="map steps evaluated for meanflow/falcon/f2d2",
    )
    p.add_argument("--fm-steps", type=int, default=100)
    p.add_argument(
        "--fm-exact",
        action="store_true",
        help="exact divergence for fm (slower, no estimator noise)",
    )
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument(
        "--inject",
        type=float,
        default=1e-2,
        help="fraction of held-out rows turned into anomalies (1e-2: 174 on Hinode)",
    )
    p.add_argument(
        "--inject-shift",
        type=int,
        default=1,
        help="feature shift of an injected anomaly; at 3 a plain Gaussian already "
        "finds every one, so recall cannot rank methods",
    )
    p.add_argument("--ks", default="10,100,1000")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bench_fast_likelihood.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    steps_list = [int(s) for s in args.steps.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    # --- data: rows, train / held-out split, injected anomalies ---
    if args.data.endswith(".npz"):
        with np.load(args.data) as f:
            cube = f["data"] / args.divide
        rows = GeneralizedPatchedDataset(cube, "y wav x", ["wav"]).patches
    else:
        rows = torch.as_tensor(np.load(args.data) / args.divide, dtype=torch.float32)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rows))
    n_hold = int(args.holdout * len(rows))
    train = Rows(rows[perm[n_hold:]])
    held = rows[perm[:n_hold]]
    held_inj, inj_idx = inject_anomalies(
        held, args.inject, args.inject_shift, args.seed
    )
    loader = SliceLoader(train, args.batch_size, args.seed)
    D = rows.shape[1]
    print(
        f"{len(train)} training rows, {len(held)} held out, D={D}, {len(inj_idx)} injected anomalies"
    )

    results = {
        "conditions": {
            "args": vars(args),
            "torch": torch.__version__,
            "device": args.device,
            "cuda": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "host": platform.node(),
            "commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
        },
        "methods": {},
    }
    scores = {}
    teacher = None

    for m in methods:
        print(f"\n=== {m} ===")
        rec = {}
        t0 = time.perf_counter()
        if m == "gaussian":
            ok = torch.isfinite(train.patches).all(1)
            mu = train.patches[ok].double().mean(0)
            chol = torch.linalg.cholesky(torch.cov(train.patches[ok].double().T))

            def gauss_lp(x=held_inj, mu=mu, chol=chol):
                d = x.double() - mu
                z = torch.cholesky_solve(d.T, chol).T
                return (
                    -0.5 * (d * z).sum(1)
                    - torch.log(torch.diagonal(chol)).sum()
                    - 0.5 * D * np.log(2 * np.pi)
                ).numpy()

            est = None
            evals = {"exact": gauss_lp}
        elif m == "nf":
            est = DensityEstimator("normalizing_flow")
            est.create_flow(input_size=D, num_layers=5, hidden_features=32, num_bins=8)
            est.train_flow(
                loader,
                learning_rate=args.lr,
                num_epochs=args.epochs,
                device=args.device,
            )
            evals = {
                "exact": lambda e=est, x=held_inj: e.log_prob(x, device=args.device)
            }
        elif m == "fm":
            est = DensityEstimator("flow_matching_sbi")
            est.create_flow(input_size=D)
            est.train_flow(
                loader,
                learning_rate=args.lr,
                num_epochs=args.epochs,
                device=args.device,
            )
            evals = {
                f"euler{args.fm_steps}{'_exact' if args.fm_exact else ''}": lambda e=est, x=held_inj: (
                    e.log_prob(
                        x,
                        solver_method="euler",
                        steps=args.fm_steps,
                        exact_divergence=args.fm_exact,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                )
            }
        else:
            kind = "regflow" if m.startswith("regflow") else m
            est = ff.FastFlowEstimator(kind).create_flow(
                D, hidden_features=args.hidden, num_layers=args.layers
            )
            if m == "regflow_t":
                if teacher is None:
                    raise SystemExit(
                        "regflow_t needs falcon earlier in --methods (it is the teacher)"
                    )
                pairs = ff.teacher_pairs(
                    teacher, len(train), D, steps=max(steps_list), device=args.device
                )
                est.y_mean, est.y_std = teacher.y_mean, teacher.y_std
                est.train_flow(
                    pairs=pairs,
                    learning_rate=args.lr,
                    num_epochs=args.epochs,
                    device=args.device,
                    max_seconds=args.max_seconds,
                    batch_size=args.batch_size,
                )
            else:
                est.train_flow(
                    loader,
                    learning_rate=args.lr,
                    num_epochs=args.epochs,
                    device=args.device,
                    max_seconds=args.max_seconds,
                )
            if m == "falcon":
                teacher = est
            if kind == "regflow":
                evals = {
                    "exact": lambda e=est, x=held_inj: e.log_prob(x, device=args.device)
                }
            else:
                evals = {
                    f"steps{s}": (
                        lambda s=s, e=est, x=held_inj: e.log_prob(
                            x, steps=s, device=args.device
                        )
                    )
                    for s in steps_list
                }
        rec["train_seconds"] = time.perf_counter() - t0
        rec["final_loss"] = float(
            np.asarray(
                getattr(getattr(est, "backend", est), "training_loss", [np.nan])
                or [np.nan]
            )[-1]
        )
        rec["eval"] = {}
        for name, fn in evals.items():
            lp, dt = timed_log_prob(fn)
            clean = np.setdiff1d(np.arange(len(lp)), inj_idx)
            rec["eval"][name] = {
                "seconds": dt,
                "rows_per_second": len(lp) / dt,
                "heldout_nll_per_dim": float(-np.nanmean(lp[clean]) / D),
                **recall(lp, inj_idx, ks),
            }
            scores[(m, name)] = lp
            print(
                f"  {name:14s} NLL/dim {rec['eval'][name]['heldout_nll_per_dim']:.4f}  "
                f"{rec['eval'][name]['rows_per_second']:.0f} rows/s  recall@{ks[-1]} "
                f"{rec['eval'][name][f'recall@{ks[-1]}']:.2f}  median rank "
                f"{rec['eval'][name]['median_rank_pct']:.2f} %"
            )
        results["methods"][m] = rec

    # --- agreement with the reference ranking ---
    ref_keys = [k for k in scores if k[0] == args.reference]
    if ref_keys:
        ref = scores[ref_keys[-1]]
        for (m, name), lp in scores.items():
            results["methods"][m]["eval"][name].update(rank_metrics(lp, ref, ks))
        print(f"\nAgreement with {ref_keys[-1]}:")
        for (m, name), lp in scores.items():
            e = results["methods"][m]["eval"][name]
            print(
                f"  {m:10s} {name:14s} spearman {e['spearman']:.3f}  top{ks[-1]} {e[f'top{ks[-1]}_overlap']:.2f}"
            )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
