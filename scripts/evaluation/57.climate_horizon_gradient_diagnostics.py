"""
POST-HOC diagnostic: why do measured climate weights (D) not differ from equal weights (B)?

    python scripts/evaluation/57.climate_horizon_gradient_diagnostics.py

Design (fixed before running; recorded here and in the output):
  folds    1 (2017), 6 (2022), 9 (2025)      training data only (targets <= train_end)
  seeds    0, 1
  states   final (model_state + optimiser_state) of the B checkpoint (v1 = v2) and
           of the corrected D checkpoint -> 12 states
  batches  5 training batches per fold: the first 5 of a permutation with seed 1234,
           batch size 64 (as in training)
  weights  equal [1,1,1,1] vs that fold's frozen measured weights (weights_final)
  noise    identical: train mode (dropout on), torch.manual_seed(10_000 + batch) before
           every forward, for both weightings
=> 60 paired comparisons at identical model + optimiser state.

Also: B-vs-D kernels, gates, gated climate corrections and predictions on matching
test cells (folds 1-9, seeds 0-4), with B seed-to-seed differences as a noise yardstick;
inference latency; compute accounting.

Outputs: results/climate_horizon/correction_2026-09-30/diagnostics_posthoc/
"""

from __future__ import annotations

import copy
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import VARIANT, build_arrays, development_last_period, extension_fold, split_channels  # noqa: E402
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import build_benchmark_folds, load_pipeline, months_for, nb_quantiles  # noqa: E402
from src.models.climate_horizon import nb_loss_terms  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402
from src.training import climate_weighting as cw  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
CORR = ROOT / "correction_2026-09-30"
OUT = CORR / "diagnostics_posthoc"
FOLDS, SEEDS, N_BATCHES = (1, 6, 9), (0, 1), 5


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def one_comparison(model, optimiser, batch, weights, noise_seed):
    """Raw per-horizon and combined climate gradients, clipping and Adam update for one weighting."""

    m = copy.deepcopy(model)
    opt = tr.build_optimiser(m, tr.DEFAULTS)
    opt.load_state_dict(copy.deepcopy(optimiser.state_dict()))
    m.train()
    torch.manual_seed(noise_seed)
    out = m(batch["X_case"], batch["X_climate"], batch["anchor"])
    num, den = nb_loss_terms(out["mu"], out["alpha"], batch["y"], batch["mask"])
    climate = list(m.climate_parameters())
    per_h = [flat(torch.autograd.grad(num[h] / den[h], climate, retain_graph=True)) for h in range(4)]
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    torch.manual_seed(noise_seed)
    log = cw.step(m, opt, batch, torch.tensor(weights, dtype=torch.float32, device=batch["y"].device))
    clim_ids = {id(p) for p in climate}
    names = dict(m.named_parameters())
    upd_clim = torch.cat([(names[n] - before[n]).reshape(-1) for n in names if id(names[n]) in clim_ids])
    upd_other = torch.cat([(names[n] - before[n]).reshape(-1) for n in names if id(names[n]) not in clim_ids])
    grad_clim = None
    torch.manual_seed(noise_seed)
    res = cw.compute_gradients(copy.deepcopy(model).train(), batch,
                               torch.tensor(weights, dtype=torch.float32, device=batch["y"].device))
    grad_clim = flat(res["climate_grads"])
    return per_h, grad_clim, log, upd_clim, upd_other, (den / den.sum()).detach().cpu().numpy()


def gradient_diagnostics(device):
    fm = load_pipeline().folds_module
    calendar = fm.load_calendar()
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    folds = {f["fold_id"]: f for f in build_benchmark_folds(fm)}
    rows, horizon_rows = [], []
    for fold_id in FOLDS:
        arrays = build_arrays(tensors, months_for(fm), extension_fold(folds[fold_id]), fm, channels,
                              last_target_period=development_last_period(calendar))
        train = {k: torch.from_numpy(arrays["train"][k]).to(device) for k in ("X_case", "X_climate", "anchor", "y", "mask")}
        order = torch.randperm(len(train["y"]), generator=torch.Generator().manual_seed(1234))
        measured = json.loads((ROOT / "weights_final" / f"fold{fold_id}_shuffle.json").read_text())["weights"]
        for seed, (source, path) in itertools.product(SEEDS, (("B", ROOT / "checkpoints/sweep/B_target_equal"),
                                                              ("D", CORR / "checkpoints/sweep/D_target_measured"))):
            model, saved = tr.load_checkpoint(path / f"fold{fold_id}_seed{seed}.pt", device, best=False)
            optimiser = tr.build_optimiser(model, tr.DEFAULTS)
            optimiser.load_state_dict(saved["optimiser_state"])
            for b in range(N_BATCHES):
                idx = order[b * 64:(b + 1) * 64].to(device)
                batch = {k: v[idx] for k, v in train.items()}
                eq = one_comparison(model, optimiser, batch, [1.0] * 4, 10_000 + b)
                ms = one_comparison(model, optimiser, batch, measured, 10_000 + b)
                per_h, a = eq[0], eq[5]
                for h in range(4):
                    horizon_rows.append({"fold": fold_id, "seed": seed, "state": source, "batch": b, "horizon": h + 1,
                                         "grad_norm_h": float(per_h[h].norm()), "a_h": float(a[h]),
                                         **{f"cos_h{h + 1}_h{j + 1}": cos(per_h[h], per_h[j]) for j in range(4)}})
                rows.append({
                    "fold": fold_id, "seed": seed, "state": source, "batch": b, "weights": measured,
                    "raw_climate_norm_equal": float(eq[1].norm()), "raw_climate_norm_measured": float(ms[1].norm()),
                    "raw_climate_cosine": cos(eq[1], ms[1]),
                    "raw_total_norm_equal": eq[2]["raw_norm_total"], "raw_total_norm_measured": ms[2]["raw_norm_total"],
                    "clip_coef_equal": eq[2]["clip_coefficient"], "clip_coef_measured": ms[2]["clip_coefficient"],
                    "clipped_equal": eq[2]["clip_coefficient"] < 1, "clipped_measured": ms[2]["clip_coefficient"] < 1,
                    "update_climate_norm_equal": float(eq[3].norm()), "update_climate_norm_measured": float(ms[3].norm()),
                    "update_climate_cosine": cos(eq[3], ms[3]),
                    "update_other_norm_equal": float(eq[4].norm()), "update_other_norm_measured": float(ms[4].norm()),
                    "update_other_cosine": cos(eq[4], ms[4]),
                    "update_climate_rel_diff": float((eq[3] - ms[3]).norm() / eq[3].norm()),
                })
    return pd.DataFrame(rows), pd.DataFrame(horizon_rows)


def behaviour_comparison():
    """B vs D on matching test cells; B seed pairs as the noise yardstick."""

    frames = []
    for arm, root in (("B_target_equal", ROOT), ("D_target_measured", CORR)):
        for p in sorted((root / "predictions" / arm).glob("fold*_seed*.parquet")):
            if "fold10" not in p.name:
                frames.append(pd.read_parquet(p, columns=["method", *ev.KEYS, "seed", "prediction", "gate", "gated_climate"]))
    f = pd.concat(frames)
    wide = f.pivot_table(index=["fold_id", "horizon", "target_period_id", "node_id", "seed"], columns="method",
                         values=["prediction", "gate", "gated_climate"]).dropna()
    rows = []
    for h in (1, 2, 3, 4):
        w = wide.xs(h, level="horizon")
        bd = {v: (w[(v, "D_target_measured")] - w[(v, "B_target_equal")]).abs().mean() for v in ("prediction", "gate", "gated_climate")}
        b = w.xs("B_target_equal", level=1, axis=1)
        s0, s1 = b.xs(0, level="seed"), b.xs(1, level="seed")
        noise = {v: (s0[v] - s1[v]).abs().mean() for v in ("prediction", "gate", "gated_climate")}
        rows.append({"horizon": h, **{f"B_vs_D_same_seed_{k}": v for k, v in bd.items()},
                     **{f"B_seed0_vs_seed1_{k}": v for k, v in noise.items()}})
    kern = []
    for fold, seed in itertools.product(range(1, 10), range(5)):
        ks = {}
        for label, path in (("B", ROOT / "checkpoints/sweep/B_target_equal"), ("D", CORR / "checkpoints/sweep/D_target_measured")):
            model, saved = tr.load_checkpoint(path / f"fold{fold}_seed{seed}.pt", torch.device("cpu"))
            with torch.no_grad():
                ks[label] = model.encoder.weights()[0]                        # [H, N, K, history]
        kern.append({"fold": fold, "seed": seed,
                     "L1_B_vs_D_same_seed": float((ks["B"] - ks["D"]).abs().sum(-1).mean())})
    kern = pd.DataFrame(kern)
    seed_noise = []
    for fold in range(1, 10):
        ks = []
        for seed in (0, 1):
            model, _ = tr.load_checkpoint(ROOT / f"checkpoints/sweep/B_target_equal/fold{fold}_seed{seed}.pt", torch.device("cpu"))
            with torch.no_grad():
                ks.append(model.encoder.weights()[0])
        seed_noise.append(float((ks[0] - ks[1]).abs().sum(-1).mean()))
    return pd.DataFrame(rows), kern, float(np.mean(seed_noise))


def latency():
    results = []
    model, _ = tr.load_checkpoint(CORR / "checkpoints/sweep/D_target_measured/fold9_seed0.pt", torch.device("cpu"))
    for device_name in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        device = torch.device(device_name)
        m = copy.deepcopy(model).to(device).eval()
        for batch in (1, 52):
            x_case = torch.randn(batch, 12, 25, 9, device=device)
            x_clim = torch.randn(batch, 37, 25, 7, device=device)
            anchor = torch.rand(batch, 25, device=device) * 5
            times = []
            with torch.inference_mode():
                for i in range(20 + 200):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = m(x_case, x_clim, anchor)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    if i >= 20:
                        times.append((time.perf_counter() - t0) * 1e3)
            mu, alpha = out["mu"].cpu().numpy(), out["alpha"].cpu().numpy()
            post = []
            for _ in range(50):
                t0 = time.perf_counter()
                nb_quantiles(mu, alpha, levels=(0.5,))
                post.append((time.perf_counter() - t0) * 1e3)
            results.append({"device": device_name, "batch_origins": batch,
                            "forecasts_per_batch": batch * 25 * 4,
                            "forward_ms_median": float(np.median(times)), "forward_ms_p95": float(np.percentile(times, 95)),
                            "nb_median_postprocess_ms_median": float(np.median(post)),
                            "forward_us_per_district_horizon_median": float(np.median(times)) * 1e3 / (batch * 100)})
    return pd.DataFrame(results)


def compute_accounting():
    def units(root, pattern="*/fold*_seed*.json"):
        return [json.loads(p.read_text()) for p in (root / "predictions").glob(pattern)]
    orig, corr = units(ROOT), units(CORR)
    pil = [json.loads(p.read_text())["runtime_s"]["total_s"] for p in (ROOT / "pilots").glob("fold*_block*_seed*.json")]
    rows = [
        {"stage": "utility pilots (60 units, 120 fits)", "fits": 120, "minutes": sum(pil) / 60},
        {"stage": "original v1 retrospective (folds 1-9)", "fits": sum(1 for u in orig if u["fold_id"] != 10),
         "minutes": sum(u["seconds"] for u in orig if u["fold_id"] != 10) / 60},
        {"stage": "original v1 hold-out 2026 (B, D)", "fits": sum(1 for u in orig if u["fold_id"] == 10),
         "minutes": sum(u["seconds"] for u in orig if u["fold_id"] == 10) / 60},
        {"stage": "v2 correction retrospective (C-H)", "fits": sum(1 for u in corr if u["fold_id"] != 10),
         "minutes": sum(u["seconds"] for u in corr if u["fold_id"] != 10) / 60},
        {"stage": "v2 correction hold-out 2026 (D)", "fits": sum(1 for u in corr if u["fold_id"] == 10),
         "minutes": sum(u["seconds"] for u in corr if u["fold_id"] == 10) / 60},
        {"stage": "development matrix (fold 9, 8 arms x 2 seeds, v1)", "fits": 16,
         "minutes": pd.read_json(ROOT / "dev_matrix" / "runs.json")["seconds"].sum() / 60},
    ]
    table = pd.DataFrame(rows)
    table.loc[len(table)] = {"stage": "total listed", "fits": table.fits.sum(), "minutes": table.minutes.sum()}
    return table


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comp, horiz = gradient_diagnostics(device)
    comp.to_csv(OUT / "paired_step_comparisons.csv", index=False)
    horiz.to_csv(OUT / "per_horizon_gradients.csv", index=False)
    behaviour, kernels, kernel_seed_noise = behaviour_comparison()
    behaviour.to_csv(OUT / "b_vs_d_behaviour.csv", index=False)
    kernels.to_csv(OUT / "b_vs_d_kernel_l1.csv", index=False)
    lat = latency()
    lat.to_csv(OUT / "latency.csv", index=False)
    acct = compute_accounting()
    acct.to_csv(OUT / "compute_accounting.csv", index=False)
    meta = {"label": "post-hoc diagnostic", "folds": FOLDS, "seeds": SEEDS, "batches_per_fold": N_BATCHES,
            "comparisons": len(comp), "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor(), "torch": torch.__version__, "precision": "float32",
            "kernel_L1_B_seed0_vs_seed1_mean": kernel_seed_noise}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 220)
    print(comp.describe().T[["mean", "min", "max"]].round(4))
    print(horiz.groupby("horizon")[["grad_norm_h", "a_h", "cos_h1_h4", "cos_h2_h4", "cos_h3_h4"]].mean().round(3))
    print(behaviour.round(4).to_string(index=False))
    print("kernel L1 B vs D mean", round(kernels.L1_B_vs_D_same_seed.mean(), 4), "| B seed0 vs seed1", round(kernel_seed_noise, 4))
    print(lat.round(4).to_string(index=False))
    print(acct.round(1).to_string(index=False))
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
