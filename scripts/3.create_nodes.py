from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


# ---------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DENGUE_PATH = RAW_DIR / "srilanka_weekly_data.csv"
GADM_PATH = RAW_DIR / "gadm41_LKA_1.json"

NAME_MAP_PATH = INTERIM_DIR / "district_name_map.csv"
NODES_PATH = PROCESSED_DIR / "nodes.csv"


# ---------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------

DENGUE_URL = (
    "https://raw.githubusercontent.com/thiyangt/denguedatahub/"
    "refs/heads/main/data-raw/srilanka_weekly_data.csv"
)

GADM_URL = (
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/"
    "gadm41_LKA_1.json"
)


# ---------------------------------------------------------------------
# Permanent node order
#
# This order becomes part of your model definition.
# Do not change it after creating adjacency matrices or model outputs.
# ---------------------------------------------------------------------

CANONICAL_ORDER = [
    "Ampara",
    "Anuradhapura",
    "Badulla",
    "Batticaloa",
    "Colombo",
    "Galle",
    "Gampaha",
    "Hambantota",
    "Jaffna",
    "Kalutara",
    "Kandy",
    "Kegalle",
    "Kilinochchi",
    "Kurunegala",
    "Mannar",
    "Matale",
    "Matara",
    "Moneragala",
    "Mullaitivu",
    "Nuwara Eliya",
    "Polonnaruwa",
    "Puttalam",
    "Ratnapura",
    "Trincomalee",
    "Vavuniya",
]

DISTRICT_TO_PROVINCE = {
    "Ampara": "Eastern",
    "Anuradhapura": "North Central",
    "Badulla": "Uva",
    "Batticaloa": "Eastern",
    "Colombo": "Western",
    "Galle": "Southern",
    "Gampaha": "Western",
    "Hambantota": "Southern",
    "Jaffna": "Northern",
    "Kalutara": "Western",
    "Kandy": "Central",
    "Kegalle": "Sabaragamuwa",
    "Kilinochchi": "Northern",
    "Kurunegala": "North Western",
    "Mannar": "Northern",
    "Matale": "Central",
    "Matara": "Southern",
    "Moneragala": "Uva",
    "Mullaitivu": "Northern",
    "NuwaraEliya": "Central",
    "Polonnaruwa": "North Central",
    "Puttalam": "North Western",
    "Ratnapura": "Sabaragamuwa",
    "Trincomalee": "Eastern",
    "Vavuniya": "Northern",
}

# ---------------------------------------------------------------------
# Explicit dengue-source-name mapping
#
# Never use approximate or fuzzy matching silently.
# Every source label must be explicitly assigned.
# ---------------------------------------------------------------------

DENGUE_TO_CANONICAL = {
    "Ampara": "Ampara",
    "Anuradhapura": "Anuradhapura",
    "Badulla": "Badulla",
    "Batticaloa": "Batticaloa",
    "Colombo": "Colombo",
    "Galle": "Galle",
    "Gampaha": "Gampaha",

    # Source spelling differs from the canonical district spelling.
    "Hambanthota": "Hambantota",

    "Jaffna": "Jaffna",

    # Kalmune is combined with the Ampara district node.
    "Kalmune": "Ampara",

    "Kalutara": "Kalutara",
    "Kandy": "Kandy",
    "Kegalle": "Kegalle",
    "Kilinochchi": "Kilinochchi",
    "Kurunegala": "Kurunegala",
    "Mannar": "Mannar",
    "Matale": "Matale",
    "Matara": "Matara",

    # Source spelling differs from the canonical district spelling.
    "Monaragala": "Moneragala",

    "Mullaitivu": "Mullaitivu",

    # Source does not contain a space.
    "NuwaraEliya": "Nuwara Eliya",

    "Polonnaruwa": "Polonnaruwa",
    "Puttalam": "Puttalam",
    "Ratnapura": "Ratnapura",
    "Trincomalee": "Trincomalee",
    "Vavuniya": "Vavuniya",
}

GADM_TO_CANONICAL = {
    "Ampara": "Ampara",
    "Anuradhapura": "Anuradhapura",
    "Badulla": "Badulla",
    "Batticaloa": "Batticaloa",
    "Colombo": "Colombo",
    "Galle": "Galle",
    "Gampaha": "Gampaha",
    "Hambantota": "Hambantota",
    "Jaffna": "Jaffna",
    "Kalutara": "Kalutara",
    "Kandy": "Kandy",
    "Kegalle": "Kegalle",
    "Kilinochchi": "Kilinochchi",
    "Kurunegala": "Kurunegala",
    "Mannar": "Mannar",
    "Matale": "Matale",
    "Matara": "Matara",
    "Moneragala": "Moneragala",
    "Mullaitivu": "Mullaitivu",
    "NuwaraEliya": "Nuwara Eliya",
    "Polonnaruwa": "Polonnaruwa",
    "Puttalam": "Puttalam",
    "Ratnapura": "Ratnapura",
    "Trincomalee": "Trincomalee",
    "Vavuniya": "Vavuniya",
}

def download_file(url: str, output_path: Path) -> None:
    """Download a file only when it does not already exist."""

    if output_path.exists():
        print(f"Using existing file: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"Downloaded: {output_path}")


def load_dengue_names(dengue_path: Path) -> pd.DataFrame:
    """Load and validate the original dengue district names."""

    dengue = pd.read_csv(dengue_path)

    required_columns = {"district"}
    missing_columns = required_columns - set(dengue.columns)

    if missing_columns:
        raise ValueError(
            f"Dengue file is missing columns: {sorted(missing_columns)}"
        )

    if dengue["district"].isna().any():
        raise ValueError("The dengue district column contains missing values.")

    dengue["district"] = dengue["district"].astype(str).str.strip()

    source_names = set(dengue["district"].unique())
    mapped_names = set(DENGUE_TO_CANONICAL)

    unmapped = source_names - mapped_names
    if unmapped:
        raise ValueError(
            "These dengue names do not have an explicit mapping: "
            f"{sorted(unmapped)}"
        )

    obsolete_mappings = mapped_names - source_names
    if obsolete_mappings:
        raise ValueError(
            "The mapping contains names not found in the source file: "
            f"{sorted(obsolete_mappings)}"
        )

    name_map = pd.DataFrame(
        {
            "dengue_name": sorted(source_names),
        }
    )

    name_map["canonical_name"] = name_map["dengue_name"].map(
        DENGUE_TO_CANONICAL
    )

    if name_map["canonical_name"].isna().any():
        raise ValueError("At least one dengue name was not mapped.")

    return name_map


def load_gadm_districts(gadm_path: Path) -> gpd.GeoDataFrame:
    """
    Load Sri Lanka's 25 district polygons from GADM level 1.

    For Sri Lanka:
        GADM level 1 = districts
        GADM level 2 = smaller divisional-level areas
    """

    gadm = gpd.read_file(gadm_path)

    print(f"Loaded GADM polygons: {len(gadm)}")
    print(f"GADM columns: {gadm.columns.tolist()}")

    required_columns = {
        "GID_1",
        "NAME_1",
        "geometry",
    }

    missing_columns = required_columns - set(gadm.columns)

    if missing_columns:
        raise ValueError(
            f"GADM data is missing columns: {sorted(missing_columns)}"
        )

    gadm = gadm[
        [
            "GID_1",
            "NAME_1",
            "geometry",
        ]
    ].copy()

    gadm = gadm.rename(
        columns={
            "GID_1": "gadm_gid",
            "NAME_1": "gadm_name",
        }
    )

    gadm["gadm_name"] = (
        gadm["gadm_name"]
        .astype(str)
        .str.strip()
    )

    gadm["gadm_gid"] = (
        gadm["gadm_gid"]
        .astype(str)
        .str.strip()
    )

    # Add province using the explicit district-to-province mapping.
    gadm["province"] = gadm["gadm_name"].map(
        DISTRICT_TO_PROVINCE
    )

    if len(gadm) != 25:
        raise ValueError(
            "Expected 25 GADM district polygons from level 1, "
            f"but found {len(gadm)}."
        )

    if gadm["gadm_name"].duplicated().any():
        duplicates = (
            gadm.loc[
                gadm["gadm_name"].duplicated(keep=False),
                "gadm_name",
            ]
            .sort_values()
            .tolist()
        )

        raise ValueError(
            f"Duplicate GADM district names found: {duplicates}"
        )

    if gadm["gadm_gid"].duplicated().any():
        duplicates = (
            gadm.loc[
                gadm["gadm_gid"].duplicated(keep=False),
                "gadm_gid",
            ]
            .sort_values()
            .tolist()
        )

        raise ValueError(
            f"Duplicate GADM IDs found: {duplicates}"
        )

    if gadm["province"].isna().any():
        missing_provinces = (
            gadm.loc[
                gadm["province"].isna(),
                "gadm_name",
            ]
            .sort_values()
            .tolist()
        )

        raise ValueError(
            "Province mapping is missing for these GADM districts: "
            f"{missing_provinces}"
        )

    if gadm.geometry.isna().any():
        missing_geometry = (
            gadm.loc[
                gadm.geometry.isna(),
                "gadm_name",
            ]
            .tolist()
        )

        raise ValueError(
            f"Missing geometry for districts: {missing_geometry}"
        )

    if not gadm.geometry.is_valid.all():
        invalid_names = (
            gadm.loc[
                ~gadm.geometry.is_valid,
                "gadm_name",
            ]
            .tolist()
        )

        raise ValueError(
            f"Invalid GADM geometries found for: {invalid_names}"
        )

    print("GADM districts:")
    print(
        gadm[
            ["gadm_gid", "gadm_name", "province"]
        ]
        .sort_values("gadm_name")
        .to_string(index=False)
    )

    return gadm 

def add_centroids(gadm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate centroids in a projected coordinate system.

    EPSG:32644 is WGS 84 / UTM zone 44N and is suitable for Sri Lanka.
    The resulting centroid coordinates are converted back to latitude
    and longitude.
    """

    if gadm.crs is None:
        raise ValueError("The GADM file does not contain CRS information.")

    gadm = gadm.to_crs("EPSG:4326").copy()

    projected = gadm.to_crs("EPSG:32644")
    projected_centroids = projected.geometry.centroid

    geographic_centroids = gpd.GeoSeries(
        projected_centroids,
        index=gadm.index,
        crs=projected.crs,
    ).to_crs("EPSG:4326")

    gadm["centroid_lon"] = geographic_centroids.x
    gadm["centroid_lat"] = geographic_centroids.y

    return gadm

def build_nodes(
    gadm: gpd.GeoDataFrame,
    name_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the permanent 25-row district node registry.

    Expected GADM columns:
        gadm_gid
        gadm_name
        province
        centroid_lat
        centroid_lon
        geometry

    Expected name_map columns:
        dengue_name
        canonical_name
    """

    gadm = gadm.copy()
    name_map = name_map.copy()

    # --------------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------------

    required_gadm_columns = {
        "gadm_gid",
        "gadm_name",
        "province",
        "centroid_lat",
        "centroid_lon",
    }

    missing_gadm_columns = (
        required_gadm_columns - set(gadm.columns)
    )

    if missing_gadm_columns:
        raise ValueError(
            "GADM data is missing required columns: "
            f"{sorted(missing_gadm_columns)}"
        )

    required_name_map_columns = {
        "dengue_name",
        "canonical_name",
    }

    missing_name_map_columns = (
        required_name_map_columns - set(name_map.columns)
    )

    if missing_name_map_columns:
        raise ValueError(
            "District name mapping is missing required columns: "
            f"{sorted(missing_name_map_columns)}"
        )

    # --------------------------------------------------------------
    # Clean string columns
    # --------------------------------------------------------------

    gadm["gadm_name"] = (
        gadm["gadm_name"]
        .astype(str)
        .str.strip()
    )

    gadm["gadm_gid"] = (
        gadm["gadm_gid"]
        .astype(str)
        .str.strip()
    )

    gadm["province"] = (
        gadm["province"]
        .astype(str)
        .str.strip()
    )

    name_map["dengue_name"] = (
        name_map["dengue_name"]
        .astype(str)
        .str.strip()
    )

    name_map["canonical_name"] = (
        name_map["canonical_name"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------------
    # Convert GADM names to canonical district names
    #
    # Example:
    # NuwaraEliya -> Nuwara Eliya
    # --------------------------------------------------------------

    gadm["canonical_name"] = gadm["gadm_name"].map(
        GADM_TO_CANONICAL
    )

    if gadm["canonical_name"].isna().any():
        unmapped_gadm_names = (
            gadm.loc[
                gadm["canonical_name"].isna(),
                "gadm_name",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise ValueError(
            "These GADM district names do not have a canonical "
            f"mapping: {unmapped_gadm_names}"
        )

    # --------------------------------------------------------------
    # Validate canonical district names
    # --------------------------------------------------------------

    expected_canonical_names = set(CANONICAL_ORDER)
    actual_gadm_names = set(gadm["canonical_name"])

    missing_from_gadm = (
        expected_canonical_names - actual_gadm_names
    )

    unexpected_in_gadm = (
        actual_gadm_names - expected_canonical_names
    )

    if missing_from_gadm or unexpected_in_gadm:
        raise ValueError(
            "Canonical district names and GADM districts do not "
            "match.\n"
            f"Missing from GADM: {sorted(missing_from_gadm)}\n"
            f"Unexpected in GADM: {sorted(unexpected_in_gadm)}"
        )

    if gadm["canonical_name"].duplicated().any():
        duplicate_names = (
            gadm.loc[
                gadm["canonical_name"].duplicated(keep=False),
                [
                    "gadm_name",
                    "canonical_name",
                    "gadm_gid",
                ],
            ]
            .sort_values("canonical_name")
        )

        raise ValueError(
            "Multiple GADM polygons map to the same canonical "
            "district:\n"
            f"{duplicate_names.to_string(index=False)}"
        )

    # --------------------------------------------------------------
    # Validate dengue name mappings
    # --------------------------------------------------------------

    if name_map["dengue_name"].duplicated().any():
        duplicate_dengue_names = (
            name_map.loc[
                name_map["dengue_name"].duplicated(keep=False),
                "dengue_name",
            ]
            .sort_values()
            .tolist()
        )

        raise ValueError(
            "Each original dengue name must map exactly once. "
            f"Duplicates found: {duplicate_dengue_names}"
        )

    invalid_canonical_names = (
        set(name_map["canonical_name"])
        - expected_canonical_names
    )

    if invalid_canonical_names:
        raise ValueError(
            "The dengue mapping contains unknown canonical names: "
            f"{sorted(invalid_canonical_names)}"
        )

    # --------------------------------------------------------------
    # Combine original dengue names for each canonical district
    #
    # Example:
    # Ampara -> Ampara|Kalmune
    # --------------------------------------------------------------

    dengue_aliases = (
        name_map
        .sort_values(
            [
                "canonical_name",
                "dengue_name",
            ]
        )
        .groupby(
            "canonical_name",
            as_index=False,
        )
        .agg(
            dengue_names=(
                "dengue_name",
                lambda values: "|".join(values),
            )
        )
    )

    # --------------------------------------------------------------
    # Attach dengue aliases to each GADM district
    # --------------------------------------------------------------

    nodes = gadm.merge(
        dengue_aliases,
        on="canonical_name",
        how="left",
        validate="one_to_one",
    )

    if nodes["dengue_names"].isna().any():
        missing_dengue_mappings = (
            nodes.loc[
                nodes["dengue_names"].isna(),
                "canonical_name",
            ]
            .sort_values()
            .tolist()
        )

        raise ValueError(
            "No dengue source name was mapped to these districts: "
            f"{missing_dengue_mappings}"
        )

    # --------------------------------------------------------------
    # Assign permanent node IDs using CANONICAL_ORDER
    # --------------------------------------------------------------

    node_id_lookup = {
        district_name: node_id
        for node_id, district_name in enumerate(
            CANONICAL_ORDER
        )
    }

    nodes["node_id"] = nodes["canonical_name"].map(
        node_id_lookup
    )

    if nodes["node_id"].isna().any():
        missing_node_ids = (
            nodes.loc[
                nodes["node_id"].isna(),
                "canonical_name",
            ]
            .sort_values()
            .tolist()
        )

        raise ValueError(
            "Node IDs could not be assigned to these districts: "
            f"{missing_node_ids}"
        )

    nodes["node_id"] = nodes["node_id"].astype(int)

    # Sort using the permanent node order.
    nodes = (
        nodes
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # Keep only columns required in nodes.csv
    # --------------------------------------------------------------

    nodes = nodes[
        [
            "node_id",
            "canonical_name",
            "dengue_names",
            "gadm_name",
            "gadm_gid",
            "province",
            "centroid_lat",
            "centroid_lon",
        ]
    ].copy()

    # --------------------------------------------------------------
    # Final function-level checks
    # --------------------------------------------------------------

    if len(nodes) != 25:
        raise ValueError(
            f"Expected 25 nodes, found {len(nodes)}."
        )

    if nodes["node_id"].tolist() != list(range(25)):
        raise ValueError(
            "node_id values must be exactly 0 through 24."
        )

    if nodes["canonical_name"].tolist() != CANONICAL_ORDER:
        raise ValueError(
            "The node order does not match CANONICAL_ORDER."
        )

    if nodes["node_id"].duplicated().any():
        raise ValueError("Duplicate node IDs found.")

    if nodes["canonical_name"].duplicated().any():
        raise ValueError(
            "Duplicate canonical district names found."
        )

    if nodes["gadm_gid"].duplicated().any():
        raise ValueError("Duplicate GADM IDs found.")

    if nodes["gadm_name"].duplicated().any():
        raise ValueError("Duplicate GADM names found.")

    if nodes.isna().any().any():
        missing_counts = nodes.isna().sum()
        missing_counts = missing_counts[
            missing_counts > 0
        ].to_dict()

        raise ValueError(
            f"Missing values found in nodes data: {missing_counts}"
        )

    return nodes


def validate_nodes(
    nodes: pd.DataFrame,
    name_map: pd.DataFrame,
    gadm: gpd.GeoDataFrame,
) -> None:
    """Run all acceptance checks for nodes.csv."""

    required_columns = {
        "node_id",
        "canonical_name",
        "dengue_names",
        "gadm_name",
        "gadm_gid",
        "province",
        "centroid_lat",
        "centroid_lon",
    }

    missing_columns = required_columns - set(nodes.columns)

    if missing_columns:
        raise AssertionError(
            f"nodes.csv is missing columns: {sorted(missing_columns)}"
        )

    # Acceptance check 1: exactly 25 rows.
    assert len(nodes) == 25, (
        f"Expected 25 rows, found {len(nodes)}."
    )

    # Acceptance check 2: IDs exactly 0-24.
    assert nodes["node_id"].tolist() == list(range(25)), (
        "node_id values must be exactly 0 through 24 in order."
    )

    # Acceptance check 3: no duplicate IDs or names.
    assert not nodes["node_id"].duplicated().any(), (
        "Duplicate node IDs found."
    )

    assert not nodes["canonical_name"].duplicated().any(), (
        "Duplicate canonical district names found."
    )

    assert not nodes["gadm_name"].duplicated().any(), (
        "Duplicate GADM district names found."
    )

    assert not nodes["gadm_gid"].duplicated().any(), (
        "Duplicate GADM IDs found."
    )

    # No missing required values.
    assert not nodes[list(required_columns)].isna().any().any(), (
        "Missing values found in nodes.csv."
    )

    # Every dengue name maps exactly once.
    assert not name_map["dengue_name"].duplicated().any(), (
        "A dengue source name appears more than once in the mapping."
    )

    assert name_map["canonical_name"].isin(
        nodes["canonical_name"]
    ).all(), (
        "A dengue name maps to a canonical node that does not exist."
    )

    expected_dengue_names = set(DENGUE_TO_CANONICAL)
    actual_dengue_names = set(name_map["dengue_name"])

    assert actual_dengue_names == expected_dengue_names, (
        "The dengue-name mapping is incomplete."
    )

    # Every GADM polygon maps exactly once.
    assert len(gadm) == 25, (
        "The original GADM data must contain 25 polygons."
    )

    assert set(nodes["gadm_gid"]) == set(gadm["gadm_gid"]), (
        "Not every GADM polygon was included exactly once."
    )

    # Canonical order is unchanged.
    assert nodes["canonical_name"].tolist() == CANONICAL_ORDER, (
        "The nodes are not in the permanent canonical order."
    )

    # Check coordinate ranges around Sri Lanka.
    assert nodes["centroid_lat"].between(5.0, 10.5).all(), (
        "At least one centroid latitude appears outside Sri Lanka."
    )

    assert nodes["centroid_lon"].between(79.0, 82.5).all(), (
        "At least one centroid longitude appears outside Sri Lanka."
    )

    print("All nodes.csv acceptance checks passed.")


def main() -> None:
    """Create district_name_map.csv and nodes.csv."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    download_file(DENGUE_URL, DENGUE_PATH)
    download_file(GADM_URL, GADM_PATH)

    name_map = load_dengue_names(DENGUE_PATH)

    gadm = load_gadm_districts(GADM_PATH)
    gadm = add_centroids(gadm)

    nodes = build_nodes(
        gadm=gadm,
        name_map=name_map,
    )

    validate_nodes(
        nodes=nodes,
        name_map=name_map,
        gadm=gadm,
    )

    name_map.to_csv(NAME_MAP_PATH, index=False)
    nodes.to_csv(NODES_PATH, index=False)

    print(f"Created: {NAME_MAP_PATH}")
    print(f"Created: {NODES_PATH}")

    print("\nnodes.csv preview:")
    print(nodes.to_string(index=False))


if __name__ == "__main__":
    main()