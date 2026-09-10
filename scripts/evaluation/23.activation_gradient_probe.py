"""
Measure what each simplex activation actually does to the encoder's gradient.

scripts/22 asks whether the activation changes the forecast. This asks the
prior question -- whether it changes the thing it is supposed to change --
because a null result in scripts/22 has two very different explanations and
only this can tell them apart:

    the activation did not alter how the mixture trains, so there was nothing
    to see;

    or it altered it exactly as intended, and the forecasting problem at
    horizon 1 has no room for the difference to show up (README §7).

Those need separating before anything is concluded. So this script trains the
real encoder on the real fold and records, per epoch, the gradient norm
reaching the basis logits, the support size of the mixture, and the width of
the delay kernel.

The headline number is `dead_fraction`: the share of district-feature pairs
whose mixture has collapsed onto a single bump. For sparsemax and entmax15 a
collapsed mixture is an absorbing state -- the map is locally constant there,
so the gradient is exactly zero and nothing can move it again. That is the
failure mode `floored_entmax15` exists to prevent, and this is where it should
be visible if it is real.

Cheap by design: one fold, one seed, no sweep. It is a mechanism probe, not a
result, and it should not be quoted as one.

Outputs (one pair per fold, so repeated runs do not overwrite each other):
    results/models/activation_gradient_probe_fold<N>.csv
    results/models/activation_gradient_probe_fold<N>.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.models.lag_encoder import DEFAULT_CENTRES, LagGCNGRU  # noqa: E402
from src.models.simplex_activations import SimplexActivation  # noqa: E402


RESULTS_DIR = PROJECT_DIR / "results" / "models"

# Named per fold. The probe is cheap enough to run on several folds, and a
# single fixed filename would leave each run silently overwriting the one
# before it -- which is how two different folds end up compared to each other
# by accident.
def output_paths(fold_id: int) -> tuple[Path, Path]:
    return (
        RESULTS_DIR / f"activation_gradient_probe_fold{fold_id}.csv",
        RESULTS_DIR / f"activation_gradient_probe_fold{fold_id}.md",
    )


MODEL_LOOKBACK = 12
LAG_REACH = 26


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_DIR / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def probe_one(
    baseline, activation, arrays, adjacency, config, device, epochs, seed
) -> list[dict]:
    """Train one encoder briefly, recording mixture statistics each epoch.

    A deliberately plain loop rather than `baseline.train_one`: the point is to
    look inside the optimisation as it happens, and the shared trainer returns
    only the fitted model. Everything that affects the comparison between arms
    -- data, seed, optimiser, learning rates -- is identical across arms, which
    is what makes the columns comparable even though the loop is not the
    project's real one.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = arrays["train"]["X"].shape[-1]

    backbone = baseline.GCNGRU(
        n_features=n_features,
        hidden=config["hidden"],
        gcn_layers=config["gcn_layers"],
        horizon=1,
        dropout=config["dropout"],
    )
    model = LagGCNGRU(
        backbone=backbone,
        n_nodes=adjacency.shape[0],
        lagged_indices=config["lagged_indices"],
        lag_reach=config["lag_reach"],
        n_basis=len(DEFAULT_CENTRES),
        embedding_dim=config["embedding_dim"],
        activation=activation,
    ).to(device)

    optimiser = torch.optim.Adam(
        baseline.parameter_groups(model, config),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    x = torch.as_tensor(arrays["train"]["X"], dtype=torch.float32, device=device)
    y = torch.as_tensor(arrays["train"]["y"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(arrays["train"]["mask"], dtype=torch.float32, device=device)
    anchor = torch.as_tensor(
        arrays["train"]["anchor"], dtype=torch.float32, device=device
    )
    graph = torch.as_tensor(adjacency, dtype=torch.float32, device=device)

    # Reuse scripts/16's own target construction rather than restating it, so
    # the probe cannot silently drift from what the project actually trains on.
    target = baseline.to_target(y, anchor, config["target"])

    rows: list[dict] = []
    batch = config["batch_size"]

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x), device=device)
        grad_norms: list[float] = []

        for start in range(0, len(x), batch):
            index = permutation[start : start + batch]
            optimiser.zero_grad()

            # Shapes follow scripts/16 exactly: the model keeps its trailing
            # horizon dimension and the mask is unsqueezed to meet it.
            prediction = model(x[index], graph)
            loss = baseline.masked_mse(
                prediction, target[index], mask[index].unsqueeze(-1)
            )
            loss.backward()

            # The quantity the whole module is about: how much gradient
            # actually reaches the parameters that shape the mixture.
            total = 0.0
            for parameter in (
                model.encoder.node_embedding,
                model.encoder.feature_projection,
            ):
                if parameter.grad is not None:
                    total += float(parameter.grad.norm() ** 2)
            grad_norms.append(total**0.5)

            # scripts/16 clips before stepping. The probe has to clip too, or
            # the gradient norms it reports would be of an optimiser the
            # project does not use.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            weights = model.encoder.mixture_weights().cpu()
            kernels = model.encoder.kernels().cpu()

            tau = torch.arange(model.encoder.lag_reach, dtype=torch.float32)
            centre = (kernels * tau).sum(-1, keepdim=True)
            spread = (kernels * (tau - centre) ** 2).sum(-1).sqrt()

            support = (weights > 1e-6).sum(-1).float()

        rows.append(
            {
                "activation": activation,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "grad_norm": float(np.mean(grad_norms)),
                "mean_support": float(support.mean()),
                # A pair pinned to one bump can never move again under a
                # piecewise-linear map. This is the number to watch.
                "dead_fraction": float((support <= 1).float().mean()),
                "mean_kernel_sd": float(spread.mean()),
                "mean_peak_lag": float(centre.mean()),
                "peak_lag_sd": float(centre.squeeze(-1).std()),
            }
        )

    return rows


def write_report(
    frame: pd.DataFrame, epochs: int, fold_id: int, report_path: Path
) -> None:
    final = frame.groupby("activation").last().reset_index()
    first = frame.groupby("activation").first()

    lines: list[str] = []
    add = lines.append

    add("# Activation gradient probe")
    add("")
    add(
        f"`scripts/23.activation_gradient_probe.py`, fold {fold_id}, one seed, "
        f"{epochs} epochs. A mechanism probe, not a result -- it says whether "
        "each activation changes how the mixture trains, which is the "
        "precondition for `scripts/22`'s MAE column meaning anything."
    )
    add("")
    add("| Activation | Grad norm (first) | Grad norm (last) | Mean support | Dead pairs | Kernel sd | Peak lag sd |")
    add("|---|---|---|---|---|---|---|")

    for row in final.itertuples():
        add(
            f"| `{row.activation}` | {first.loc[row.activation, 'grad_norm']:.2e} | "
            f"{row.grad_norm:.2e} | {row.mean_support:.2f} | "
            f"{row.dead_fraction:.0%} | {row.mean_kernel_sd:.2f} | "
            f"{row.peak_lag_sd:.2f} |"
        )

    add("")
    add("**Columns.** `grad norm` is the norm reaching `node_embedding` and ")
    add("`feature_projection`, the parameters that shape the mixture. `mean ")
    add("support` is how many of the six bumps carry mass. `dead pairs` is the ")
    add("share of district-feature pairs collapsed onto one bump, where a ")
    add("sparse map has an exactly zero gradient and can never recover. ")
    add("`kernel sd` is the width of the delay curve, `peak lag sd` its ")
    add("variation across districts -- a curve that is identical everywhere ")
    add("has learned nothing district-specific, whatever its MAE.")
    add("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=list(SimplexActivation.NAMES))
    arguments = parser.parse_args()

    lag_module = load_script("train_lag_gcn_gru", "training/18.train_lag_gcn_gru.py")
    baseline = lag_module.load_baseline()

    config = dict(baseline.DEFAULTS)
    config["target"] = "residual"
    config["lag_reach"] = LAG_REACH
    config["embedding_dim"] = 8
    config["kernel_learning_rate"] = 2e-2

    folds = json.loads(baseline.FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    fold = next(f for f in folds if f["fold_id"] == arguments.fold)

    with np.load(baseline.ADJACENCY_PATH, allow_pickle=True) as data:
        adjacency = data["A_norm"].astype(np.float32)

    tensors = baseline.folds_module.load_tensors("v0")
    feature_names = list(tensors["feature_names"])
    config["lagged_indices"] = [
        feature_names.index(name) for name in lag_module.LAGGED_FEATURES
    ]

    calendar = baseline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    window = config["lag_reach"] + MODEL_LOOKBACK - 1
    arrays = baseline.build_fold_arrays(
        tensors, months, fold, window, config["horizon"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}   fold {fold['fold_id']} ({fold['test_year']})")
    print(f"Arms:   {arguments.arms}")
    print(f"Epochs: {arguments.epochs}\n")

    rows: list[dict] = []
    for activation in arguments.arms:
        rows.extend(
            probe_one(
                baseline,
                activation,
                arrays,
                adjacency,
                config,
                device,
                arguments.epochs,
                arguments.seed,
            )
        )
        last = rows[-1]
        print(
            f"  {activation:<18} grad {last['grad_norm']:.2e}  "
            f"support {last['mean_support']:.2f}  "
            f"dead {last['dead_fraction']:.0%}  "
            f"kernel sd {last['mean_kernel_sd']:.2f}"
        )

    frame = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, report_path = output_paths(fold["fold_id"])
    frame.to_csv(csv_path, index=False)
    write_report(frame, arguments.epochs, fold["fold_id"], report_path)

    print(f"\nWrote {csv_path.relative_to(PROJECT_DIR)}")
    print(f"Wrote {report_path.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
