"""
Build the district adjacency matrices for the graph model.

The node axis of the model tensors is node_id 0-24 from nodes.csv, so the
adjacency has to use the same index or the graph and the features silently
refer to different districts. Districts are matched to GADM polygons by
gadm_gid, never by name: the registry exists precisely so that name matching
never happens again.

Three matrices are produced from one set of polygons.

    A_binary     queen contiguity. 1 where two districts share a boundary.
    A_distance   great-circle-free centroid distance in km, in a projected CRS.
    A_gaussian   exp(-(d / sigma)^2) on that distance, thresholded.

A_binary is the fixed A_manual of the eventual dual-graph model. A_gaussian is
kept beside it because contiguity is a crude prior in a country this narrow:
Colombo and Gampaha are 30 km apart and share a border, but Colombo and Galle
are 100 km apart and do not, yet are both dense wet-zone coastal districts.

Contiguity is tested on polygons buffered by a small tolerance. GADM boundaries
are digitised, so two districts that share a border in reality can be separated
by a sliver a few metres wide in the file, and a strict touches() test misses
them. The tolerance is applied in a projected CRS and recorded below.

Sri Lanka has land borders everywhere except across the Jaffna lagoon, so an
isolated node is possible. Any node left with no neighbour is connected to its
nearest centroids and the fallback is logged, never applied silently.

Outputs:
    data/processed/adjacency.npz
    results/models/adjacency_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

BOUNDARY_PATH = PROJECT_DIR / "data" / "raw" / "gadm41_LKA_1.json"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

OUTPUT_PATH = PROCESSED_DIR / "adjacency.npz"
REPORT_PATH = RESULTS_DIR / "adjacency_report.md"

ADJACENCY_VERSION = "adjacency-v1"

EXPECTED_NODES = 25

# Equal-area projection used for every distance and area calculation, matching
# the CRS the node registry used for its centroids.
PROJECTED_CRS = "EPSG:32644"

# Half-width of the sliver a shared border may be split by, in metres. 100 m is
# far below the width of any Sri Lankan district and far above GADM's
# digitisation error.
CONTIGUITY_TOLERANCE_M = 100.0

# Length scale of the Gaussian kernel, in km. Roughly the median distance
# between neighbouring district centroids.
GAUSSIAN_SIGMA_KM = 50.0

# Kernel weights below this are set to zero, so the dense kernel stays sparse
# enough to act as a graph rather than a fully connected layer.
GAUSSIAN_THRESHOLD = 0.1

# Neighbours given to a node that contiguity leaves isolated.
ISOLATED_FALLBACK_K = 2


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_nodes(path: Path = NODES_PATH) -> pd.DataFrame:
    """Load the canonical district registry, ordered by node_id."""

    nodes = pd.read_csv(path).sort_values("node_id").reset_index(drop=True)

    if len(nodes) != EXPECTED_NODES:
        raise ValueError(f"nodes.csv has {len(nodes)} rows, expected {EXPECTED_NODES}.")

    if not np.array_equal(nodes["node_id"].to_numpy(), np.arange(EXPECTED_NODES)):
        raise ValueError("node_id must be the dense sequence 0-24.")

    return nodes


def load_boundaries(nodes: pd.DataFrame, path: Path = BOUNDARY_PATH) -> gpd.GeoDataFrame:
    """Load district polygons and order them by node_id via gadm_gid."""

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")

    boundaries = gpd.read_file(path)

    if "GID_1" not in boundaries.columns:
        raise ValueError(f"{path.name} has no GID_1 column.")

    # Join on the GADM identifier carried by the registry. Names are not used:
    # Kalmune is folded into Ampara on the dengue side and the spellings do not
    # agree across sources.
    merged = nodes.merge(
        boundaries[["GID_1", "geometry"]],
        left_on="gadm_gid",
        right_on="GID_1",
        how="left",
        validate="one_to_one",
    )

    missing = merged[merged["geometry"].isna()]
    if len(missing) > 0:
        names = missing["canonical_name"].tolist()
        raise ValueError(f"No polygon matched gadm_gid for: {names}")

    frame = gpd.GeoDataFrame(merged, geometry="geometry", crs=boundaries.crs)

    return frame.sort_values("node_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contiguity
# ---------------------------------------------------------------------------

def build_contiguity(
    boundaries: gpd.GeoDataFrame,
    tolerance_m: float = CONTIGUITY_TOLERANCE_M,
) -> np.ndarray:
    """Return the binary queen-contiguity matrix, symmetric with a zero diagonal."""

    projected = boundaries.to_crs(PROJECTED_CRS)
    buffered = projected.geometry.buffer(tolerance_m)

    n_nodes = len(boundaries)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if buffered.iloc[i].intersects(buffered.iloc[j]):
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    return adjacency


def centroid_distance_km(boundaries: gpd.GeoDataFrame) -> np.ndarray:
    """Return the pairwise centroid distance matrix in kilometres."""

    projected = boundaries.to_crs(PROJECTED_CRS)
    centroids = projected.geometry.centroid

    coordinates = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    deltas = coordinates[:, None, :] - coordinates[None, :, :]

    return np.sqrt((deltas ** 2).sum(axis=2)) / 1000.0


def connect_isolated_nodes(
    adjacency: np.ndarray,
    distance_km: np.ndarray,
    k: int = ISOLATED_FALLBACK_K,
) -> tuple[np.ndarray, list[int]]:
    """Link any node without a neighbour to its k nearest centroids."""

    adjacency = adjacency.copy()
    isolated = [int(index) for index in np.where(adjacency.sum(axis=1) == 0)[0]]

    for index in isolated:
        order = np.argsort(distance_km[index])
        # skip position 0, which is the node itself at distance zero
        for neighbour in order[1 : k + 1]:
            adjacency[index, neighbour] = 1
            adjacency[neighbour, index] = 1

    return adjacency, isolated


def build_gaussian(
    distance_km: np.ndarray,
    sigma_km: float = GAUSSIAN_SIGMA_KM,
    threshold: float = GAUSSIAN_THRESHOLD,
) -> np.ndarray:
    """Return a thresholded Gaussian kernel on centroid distance, zero diagonal."""

    kernel = np.exp(-((distance_km / sigma_km) ** 2))
    kernel[kernel < threshold] = 0.0
    np.fill_diagonal(kernel, 0.0)

    return kernel.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(adjacency: np.ndarray) -> np.ndarray:
    """Return the symmetric normalisation D^-1/2 (A + I) D^-1/2.

    The self-loop is added before the degree is computed, which is what keeps a
    node's own features in its convolved output. A node with no neighbours then
    still has degree 1 rather than 0, so the inverse square root is always
    defined.
    """

    with_self_loops = adjacency.astype(np.float64) + np.eye(len(adjacency))
    degree = with_self_loops.sum(axis=1)

    inverse_sqrt = 1.0 / np.sqrt(degree)
    normalised = with_self_loops * inverse_sqrt[:, None] * inverse_sqrt[None, :]

    return normalised.astype(np.float32)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run_assertions(
    binary: np.ndarray,
    normalised: np.ndarray,
    gaussian: np.ndarray,
    distance_km: np.ndarray,
) -> None:
    """Assert the structural properties every downstream layer relies on."""

    for name, matrix in [
        ("A_binary", binary),
        ("A_norm", normalised),
        ("A_gaussian", gaussian),
        ("A_distance_km", distance_km),
    ]:
        if matrix.shape != (EXPECTED_NODES, EXPECTED_NODES):
            raise ValueError(f"{name} has shape {matrix.shape}.")

        if not np.allclose(matrix, matrix.T, atol=1e-6):
            raise ValueError(f"{name} is not symmetric.")

    if not (np.diag(binary) == 0).all():
        raise ValueError("A_binary has a non-zero diagonal.")

    if set(np.unique(binary).tolist()) - {0, 1}:
        raise ValueError("A_binary is not binary.")

    if (binary.sum(axis=1) == 0).any():
        isolated = np.where(binary.sum(axis=1) == 0)[0].tolist()
        raise ValueError(f"Isolated nodes remain after the fallback: {isolated}")

    if not np.isfinite(normalised).all():
        raise ValueError("A_norm contains non-finite values.")

    # A symmetrically normalised adjacency with self-loops has spectrum in
    # [-1, 1]. A value outside it means the normalisation is wrong and a deep
    # stack of graph convolutions would diverge.
    eigenvalues = np.linalg.eigvalsh(normalised.astype(np.float64))
    if eigenvalues.max() > 1.0 + 1e-6 or eigenvalues.min() < -1.0 - 1e-6:
        raise ValueError(f"A_norm spectrum outside [-1, 1]: {eigenvalues.min()}, {eigenvalues.max()}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def neighbour_table(binary: np.ndarray, nodes: pd.DataFrame) -> pd.DataFrame:
    """Return one row per district with its degree and neighbour names."""

    names = nodes["canonical_name"].tolist()

    records = []
    for index, name in enumerate(names):
        neighbours = [names[j] for j in np.where(binary[index] == 1)[0]]
        records.append(
            {
                "node_id": index,
                "canonical_name": name,
                "degree": len(neighbours),
                "neighbours": ", ".join(neighbours),
            }
        )

    return pd.DataFrame(records)


def write_report(
    table: pd.DataFrame,
    binary: np.ndarray,
    gaussian: np.ndarray,
    distance_km: np.ndarray,
    isolated: list[int],
    nodes: pd.DataFrame,
) -> None:
    """Write the human-readable adjacency report."""

    degrees = binary.sum(axis=1)
    edges = int(binary.sum() // 2)

    off_diagonal = ~np.eye(len(binary), dtype=bool)
    neighbour_distances = distance_km[(binary == 1) & off_diagonal]

    lines = [
        "# District adjacency",
        "",
        f"Version: `{ADJACENCY_VERSION}`",
        "",
        "Queen contiguity over the GADM level 1 polygons, indexed by `node_id`",
        "0-24 from `nodes.csv` and matched on `gadm_gid`.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Nodes | {len(binary)} |",
        f"| Edges | {edges} |",
        f"| Mean degree | {degrees.mean():.2f} |",
        f"| Min degree | {int(degrees.min())} |",
        f"| Max degree | {int(degrees.max())} |",
        f"| Density | {binary.sum() / (len(binary) * (len(binary) - 1)):.3f} |",
        f"| Contiguity tolerance | {CONTIGUITY_TOLERANCE_M:.0f} m |",
        f"| Mean neighbour distance | {neighbour_distances.mean():.1f} km |",
        f"| Max neighbour distance | {neighbour_distances.max():.1f} km |",
        f"| Gaussian sigma | {GAUSSIAN_SIGMA_KM:.0f} km |",
        f"| Gaussian edges | {int((gaussian > 0).sum() // 2)} |",
        "",
    ]

    if isolated:
        names = nodes.loc[isolated, "canonical_name"].tolist()
        lines += [
            "## Isolated node fallback",
            "",
            f"Contiguity left these districts with no neighbour, so each was",
            f"connected to its {ISOLATED_FALLBACK_K} nearest centroids:",
            "",
        ]
        lines += [f"- {name} (node {index})" for index, name in zip(isolated, names)]
        lines.append("")
    else:
        lines += [
            "## Isolated node fallback",
            "",
            "Not used. Every district shares a boundary with at least one other.",
            "",
        ]

    lines += [
        "## Neighbours",
        "",
        "| node_id | District | Degree | Neighbours |",
        "| --- | --- | --- | --- |",
    ]

    for row in table.itertuples():
        lines.append(
            f"| {row.node_id} | {row.canonical_name} | {row.degree} | "
            f"{row.neighbours} |"
        )

    lines += [
        "",
        "## Output files",
        "",
        "- `data/processed/adjacency.npz`",
        "",
        "Keys: `A_binary`, `A_norm`, `A_gaussian`, `A_gaussian_norm`,",
        "`A_distance_km`, `node_id`, `canonical_name`.",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Build and write the adjacency matrices."""

    try:
        nodes = load_nodes()
        boundaries = load_boundaries(nodes)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    distance_km = centroid_distance_km(boundaries)

    binary = build_contiguity(boundaries)
    binary, isolated = connect_isolated_nodes(binary, distance_km)

    gaussian = build_gaussian(distance_km)

    normalised = normalise(binary)
    gaussian_normalised = normalise(gaussian)

    run_assertions(binary, normalised, gaussian, distance_km)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        version=np.array(ADJACENCY_VERSION),
        A_binary=binary,
        A_norm=normalised,
        A_gaussian=gaussian,
        A_gaussian_norm=gaussian_normalised,
        A_distance_km=distance_km.astype(np.float32),
        node_id=nodes["node_id"].to_numpy(dtype=np.int32),
        canonical_name=nodes["canonical_name"].to_numpy(dtype=object),
    )

    table = neighbour_table(binary, nodes)
    write_report(table, binary, gaussian, distance_km, isolated, nodes)

    degrees = binary.sum(axis=1)
    print(f"Nodes:        {len(binary)}")
    print(f"Edges:        {int(binary.sum() // 2)}")
    print(f"Degree:       min {int(degrees.min())}, mean {degrees.mean():.2f}, max {int(degrees.max())}")
    print(f"Isolated:     {len(isolated)} node(s) required the centroid fallback")

    print("\nLeast connected districts")
    for row in table.nsmallest(3, "degree").itertuples():
        print(f"  {row.canonical_name:<15} {row.degree}  {row.neighbours}")

    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
