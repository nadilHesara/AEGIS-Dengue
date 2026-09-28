"""
Tests for the district adjacency matrices.

Structure first: symmetry, a zero diagonal, no isolated nodes, and a normalised
spectrum inside [-1, 1]. Then geography, because a matrix can satisfy every
structural property and still describe the wrong country. Colombo borders
Gampaha; Jaffna does not border Colombo. If those two ever swap, the graph is
wrong in a way no training curve will show.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

BOUNDARY_PATH = PROJECT_DIR / "data" / "raw" / "gadm41_LKA_1.json"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"

spec = importlib.util.spec_from_file_location(
    "build_adjacency",
    PROJECT_DIR / "scripts" / "graph" / "13.build_adjacency.py",
)
adjacency_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adjacency_module)


requires_real_data = pytest.mark.skipif(
    not (BOUNDARY_PATH.exists() and NODES_PATH.exists()),
    reason="Boundary file or node registry missing.",
)


# ---------------------------------------------------------------------------
# Normalisation, on synthetic matrices
# ---------------------------------------------------------------------------

def path_graph(n_nodes=5):
    """Return the adjacency of a simple path, a graph with known structure."""

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for index in range(n_nodes - 1):
        adjacency[index, index + 1] = 1
        adjacency[index + 1, index] = 1

    return adjacency


def test_normalisation_is_symmetric_and_bounded():
    normalised = adjacency_module.normalise(path_graph())

    np.testing.assert_allclose(normalised, normalised.T, atol=1e-6)

    eigenvalues = np.linalg.eigvalsh(normalised.astype(np.float64))
    assert eigenvalues.max() <= 1.0 + 1e-6
    assert eigenvalues.min() >= -1.0 - 1e-6


def test_normalisation_adds_a_self_loop():
    """A node keeps its own features after convolution."""

    normalised = adjacency_module.normalise(path_graph())

    assert (np.diag(normalised) > 0).all()


def test_normalisation_handles_a_node_with_no_neighbours():
    isolated = np.zeros((3, 3), dtype=np.int8)
    isolated[0, 1] = isolated[1, 0] = 1

    normalised = adjacency_module.normalise(isolated)

    assert np.isfinite(normalised).all()
    # the self-loop leaves the lone node passing its own features through
    assert normalised[2, 2] == pytest.approx(1.0)


def test_isolated_nodes_are_connected_to_their_nearest_centroids():
    adjacency = np.zeros((4, 4), dtype=np.int8)
    adjacency[0, 1] = adjacency[1, 0] = 1

    distance = np.array(
        [
            [0.0, 10.0, 50.0, 90.0],
            [10.0, 0.0, 40.0, 80.0],
            [50.0, 40.0, 0.0, 30.0],
            [90.0, 80.0, 30.0, 0.0],
        ]
    )

    connected, isolated = adjacency_module.connect_isolated_nodes(
        adjacency, distance, k=1
    )

    assert isolated == [2, 3]
    assert connected[2, 3] == 1 and connected[3, 2] == 1
    assert (connected.sum(axis=1) > 0).all()
    np.testing.assert_array_equal(connected, connected.T)


def test_gaussian_kernel_falls_off_with_distance():
    distance = np.array([[0.0, 10.0, 400.0], [10.0, 0.0, 390.0], [400.0, 390.0, 0.0]])

    kernel = adjacency_module.build_gaussian(distance, sigma_km=50.0, threshold=0.1)

    assert kernel[0, 1] > 0
    # 400 km at sigma 50 falls far below the threshold
    assert kernel[0, 2] == 0.0
    assert (np.diag(kernel) == 0).all()
    np.testing.assert_allclose(kernel, kernel.T)


# ---------------------------------------------------------------------------
# Real boundaries
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_adjacency():
    nodes = adjacency_module.load_nodes()
    boundaries = adjacency_module.load_boundaries(nodes)

    distance = adjacency_module.centroid_distance_km(boundaries)
    binary, isolated = adjacency_module.connect_isolated_nodes(
        adjacency_module.build_contiguity(boundaries), distance
    )

    return {
        "nodes": nodes,
        "binary": binary,
        "distance": distance,
        "isolated": isolated,
        "normalised": adjacency_module.normalise(binary),
        "gaussian": adjacency_module.build_gaussian(distance),
    }


@requires_real_data
def test_real_adjacency_passes_its_own_assertions(real_adjacency):
    adjacency_module.run_assertions(
        real_adjacency["binary"],
        real_adjacency["normalised"],
        real_adjacency["gaussian"],
        real_adjacency["distance"],
    )


@requires_real_data
def test_shape_and_node_order_match_the_registry(real_adjacency):
    nodes = real_adjacency["nodes"]

    assert real_adjacency["binary"].shape == (25, 25)
    np.testing.assert_array_equal(nodes["node_id"].to_numpy(), np.arange(25))


@requires_real_data
def test_polygons_are_matched_by_gid_not_by_name(real_adjacency):
    """Kalmune has no polygon; it is folded into Ampara on the dengue side."""

    nodes = real_adjacency["nodes"]
    boundaries = adjacency_module.load_boundaries(nodes)

    assert len(boundaries) == 25
    assert boundaries["gadm_gid"].tolist() == boundaries["GID_1"].tolist()
    assert boundaries["geometry"].notna().all()


@requires_real_data
def test_no_district_is_left_isolated(real_adjacency):
    degrees = real_adjacency["binary"].sum(axis=1)

    assert (degrees > 0).all()


@requires_real_data
@pytest.mark.parametrize(
    "left,right,adjacent",
    [
        ("Colombo", "Gampaha", True),
        ("Colombo", "Kalutara", True),
        ("Jaffna", "Kilinochchi", True),
        ("Galle", "Matara", True),
        ("Colombo", "Jaffna", False),
        ("Colombo", "Batticaloa", False),
        ("Jaffna", "Matara", False),
    ],
)
def test_known_geography(real_adjacency, left, right, adjacent):
    names = real_adjacency["nodes"]["canonical_name"].tolist()

    i, j = names.index(left), names.index(right)

    assert bool(real_adjacency["binary"][i, j]) is adjacent


@requires_real_data
def test_neighbours_are_closer_than_non_neighbours_on_average(real_adjacency):
    binary = real_adjacency["binary"]
    distance = real_adjacency["distance"]

    off_diagonal = ~np.eye(len(binary), dtype=bool)

    neighbour_distance = distance[(binary == 1) & off_diagonal].mean()
    other_distance = distance[(binary == 0) & off_diagonal].mean()

    assert neighbour_distance < other_distance


@requires_real_data
def test_the_graph_is_connected(real_adjacency):
    """A disconnected component could never exchange information."""

    binary = real_adjacency["binary"]

    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbour in np.where(binary[node] == 1)[0]:
            if int(neighbour) not in reached:
                reached.add(int(neighbour))
                frontier.append(int(neighbour))

    assert len(reached) == len(binary)
