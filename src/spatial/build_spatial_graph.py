# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import NearestNeighbors


try:
    from torch_geometric.data import Data

    HAS_TORCH_GEOMETRIC = True

except ImportError:
    Data = None
    HAS_TORCH_GEOMETRIC = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASETS = [
    "block_a_section_1",
    "block_a_section_2",
    "visium_fresh_frozen_whole_transcriptome",
    "block_a_section_1_2",
    "block_a_section_2_2",
    "ductal_carcinoma_in_situ_invasive_carcinoma_FFPE",
    "whole_transcriptome_analysis",
]


BASE_MODEL_NAMES = [
    "ridge",
    "xgboost",
    "dnn",
]

GRAPH_METHOD = "knn"
K_NEIGHBORS = 5

INCLUDE_SELF_LOOP = False
MAKE_UNDIRECTED = True
NORMALIZE_POSITIONS = True

FEATURE_SOURCES = [
    "X_spot_pca",
    "predicted_ic50_ridge",
    "predicted_ic50_xgboost",
    "predicted_ic50_dnn",
]

TARGET_SOURCE = "mean_ic50_baseline"

OVERWRITE_OUTPUTS = True
SAVE_LOG = True


# ============================================================
# 2. PATH SETUP
# ============================================================

def get_project_root() -> Path:
    """
    Get project root directory.

    Returns
    -------
    Path
        Project root path.
    """
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict[str, Path]:
    """
    Setup all project paths.

    Returns
    -------
    dict[str, Path]
        Dictionary of important paths.
    """
    project_root = get_project_root()

    spatial_processed_dir = project_root / "data" / "spatial_processed"
    results_spatial_dir = project_root / "results" / "spatial"
    logs_dir = project_root / "logs"

    spatial_processed_dir.mkdir(parents=True, exist_ok=True)
    results_spatial_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "spatial_processed_dir": spatial_processed_dir,
        "results_spatial_dir": results_spatial_dir,
        "logs_dir": logs_dir,
        "log_file": logs_dir / f"build_spatial_graph_{timestamp}.log",
    }


# ============================================================
# 3. LOGGING
# ============================================================

def log_print(message: str = "", log_file: Path | None = None) -> None:
    """
    Print message to terminal and write to log file.

    Parameters
    ----------
    message : str
        Message to print and log.
    log_file : Path | None
        Log file path.
    """
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(str(message) + "\n")


# ============================================================
# 4. READ INPUT DATA
# ============================================================

def read_positions(
    file_path: Path,
    log_file: Path | None = None,
) -> pd.DataFrame:
    """
    Read spot position dataframe.

    Required columns:
    - spot_id
    - x
    - y

    Parameters
    ----------
    file_path : Path
        Path to spot_positions.csv.
    log_file : Path | None
        Log file path.

    Returns
    -------
    pd.DataFrame
        Position dataframe.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file positions: {file_path}")

    positions_df = pd.read_csv(file_path)

    required_cols = [
        "spot_id",
        "x",
        "y",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in positions_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"File positions thiếu cột bắt buộc: {missing_cols}"
        )

    positions_df["spot_id"] = positions_df["spot_id"].astype(str).str.strip()

    positions_df["x"] = pd.to_numeric(
        positions_df["x"],
        errors="coerce",
    )

    positions_df["y"] = pd.to_numeric(
        positions_df["y"],
        errors="coerce",
    )

    positions_df = positions_df.dropna(
        subset=[
            "spot_id",
            "x",
            "y",
        ]
    ).copy()

    duplicated_count = int(
        positions_df["spot_id"].duplicated().sum()
    )

    if duplicated_count > 0:

        log_print(
            f"[WARNING] positions có {duplicated_count} spot_id bị trùng. "
            f"Giữ dòng đầu tiên.",
            log_file,
        )

        positions_df = positions_df.drop_duplicates(
            subset=["spot_id"],
            keep="first",
        ).copy()

    log_print(
        f"Positions shape: {positions_df.shape}",
        log_file,
    )

    return positions_df


def read_matrix(
    file_path: Path,
    log_file: Path | None = None,
) -> pd.DataFrame:
    """
    Read matrix CSV with spot_id as index.

    Used for:
    - X_spot_pca.csv
    - predicted_ic50_ridge.csv
    - predicted_ic50_xgboost.csv
    - predicted_ic50_dnn.csv
    - mean_ic50_baseline.csv

    Parameters
    ----------
    file_path : Path
        Matrix CSV file.
    log_file : Path | None
        Log file path.

    Returns
    -------
    pd.DataFrame
        Numeric matrix dataframe.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy matrix file: {file_path}")

    df = pd.read_csv(
        file_path,
        index_col=0,
    )

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df.index.name = "spot_id"

    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    df = df.fillna(0.0)

    log_print(
        f"{file_path.name} shape: {df.shape}",
        log_file,
    )

    return df


# ============================================================
# 5. READ MODEL PREDICTIONS
# ============================================================

def read_model_predictions(
    dataset_dir: Path,
    base_model_names: list[str],
    log_file: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Read IC50 predictions from all baseline models.

    Parameters
    ----------
    dataset_dir : Path
        Dataset processed directory.
    base_model_names : list[str]
        Baseline model names.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, pd.DataFrame]
        Prediction dictionary.
    """
    prediction_dict: dict[str, pd.DataFrame] = {}

    for model_name in base_model_names:

        prediction_file = (
            dataset_dir
            / f"predicted_ic50_{model_name}.csv"
        )

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Thiếu prediction của model {model_name}: {prediction_file}"
            )

        prediction_df = read_matrix(
            file_path=prediction_file,
            log_file=log_file,
        )

        prediction_dict[model_name] = prediction_df

        log_print(
            f"Loaded {model_name} prediction: {prediction_df.shape}",
            log_file,
        )

    return prediction_dict


# ============================================================
# 6. SYNC GRAPH INPUTS
# ============================================================

def sync_graph_inputs(
    positions_df: pd.DataFrame,
    X_pca: pd.DataFrame,
    prediction_dict: dict[str, pd.DataFrame],
    mean_baseline: pd.DataFrame,
    log_file: Path | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Synchronize positions, PCA features, baseline model predictions,
    and mean baseline target.

    Spot order follows X_pca.index.
    Drug order follows mean_baseline.columns.

    Parameters
    ----------
    positions_df : pd.DataFrame
        Spot positions.
    X_pca : pd.DataFrame
        PCA feature matrix.
    prediction_dict : dict[str, pd.DataFrame]
        Baseline model predictions.
    mean_baseline : pd.DataFrame
        Mean IC50 baseline target.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple
        Synced positions, X_pca, prediction_dict, mean_baseline, sync stats.
    """
    position_spots = set(positions_df["spot_id"].astype(str))
    pca_spots = list(X_pca.index.astype(str))
    baseline_spots = set(mean_baseline.index.astype(str))

    model_spot_sets = {
        model_name: set(pred_df.index.astype(str))
        for model_name, pred_df in prediction_dict.items()
    }

    common_spots = []

    for spot_id in pca_spots:
        if spot_id not in position_spots:
            continue

        if spot_id not in baseline_spots:
            continue

        is_in_all_models = True

        for model_name in BASE_MODEL_NAMES:
            if spot_id not in model_spot_sets[model_name]:
                is_in_all_models = False
                break

        if is_in_all_models:
            common_spots.append(spot_id)

    baseline_drugs = list(mean_baseline.columns.astype(str))

    model_drug_sets = {
        model_name: set(pred_df.columns.astype(str))
        for model_name, pred_df in prediction_dict.items()
    }

    common_drugs = []

    for drug in baseline_drugs:
        is_in_all_models = True

        for model_name in BASE_MODEL_NAMES:
            if drug not in model_drug_sets[model_name]:
                is_in_all_models = False
                break

        if is_in_all_models:
            common_drugs.append(drug)

    sync_stats = {
        "n_position_spots": int(len(position_spots)),
        "n_pca_spots": int(len(pca_spots)),
        "n_baseline_spots": int(len(baseline_spots)),
        "n_common_spots": int(len(common_spots)),
        "n_common_drugs": int(len(common_drugs)),
    }

    for model_name in BASE_MODEL_NAMES:
        sync_stats[f"n_{model_name}_spots"] = int(
            len(model_spot_sets[model_name])
        )

    log_print(f"n_positions_spots: {sync_stats['n_position_spots']}", log_file)
    log_print(f"n_pca_spots: {sync_stats['n_pca_spots']}", log_file)
    log_print(f"n_ridge_spots: {sync_stats['n_ridge_spots']}", log_file)
    log_print(f"n_xgboost_spots: {sync_stats['n_xgboost_spots']}", log_file)
    log_print(f"n_dnn_spots: {sync_stats['n_dnn_spots']}", log_file)
    log_print(f"n_baseline_spots: {sync_stats['n_baseline_spots']}", log_file)
    log_print(f"n_common_spots: {sync_stats['n_common_spots']}", log_file)
    log_print(f"n_common_drugs: {sync_stats['n_common_drugs']}", log_file)

    if len(common_spots) == 0:
        raise ValueError("Không có spot chung giữa toàn bộ graph inputs.")

    if len(common_drugs) == 0:
        raise ValueError("Không có drug chung giữa model predictions và baseline.")

    positions_synced = (
        positions_df
        .set_index("spot_id")
        .loc[common_spots]
        .reset_index()
    )

    X_pca_synced = X_pca.loc[common_spots].copy()
    X_pca_synced.index.name = "spot_id"

    prediction_synced: dict[str, pd.DataFrame] = {}

    for model_name in BASE_MODEL_NAMES:
        prediction_synced[model_name] = prediction_dict[model_name].loc[
            common_spots,
            common_drugs,
        ].copy()

        prediction_synced[model_name].index.name = "spot_id"

    mean_baseline_synced = mean_baseline.loc[
        common_spots,
        common_drugs,
    ].copy()

    mean_baseline_synced.index.name = "spot_id"

    return (
        positions_synced,
        X_pca_synced,
        prediction_synced,
        mean_baseline_synced,
        sync_stats,
    )


# ============================================================
# 7. BUILD NODE FEATURES
# ============================================================

def build_node_feature_matrix(
    X_pca: pd.DataFrame,
    prediction_dict: dict[str, pd.DataFrame],
    base_model_names: list[str],
    log_file: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build node feature matrix by concatenating:

    PCA features
    + Ridge IC50 predictions
    + XGBoost IC50 predictions
    + DNN IC50 predictions.

    Parameters
    ----------
    X_pca : pd.DataFrame
        PCA features.
    prediction_dict : dict[str, pd.DataFrame]
        Model prediction dataframes.
    base_model_names : list[str]
        Baseline model names.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Node feature dataframe and feature statistics.
    """
    feature_parts = []

    X_pca_part = X_pca.copy()
    X_pca_part.columns = [
        f"pca_{col}"
        for col in X_pca_part.columns
    ]

    feature_parts.append(X_pca_part)

    feature_stats: dict[str, Any] = {
        "n_pca_features": int(X_pca_part.shape[1]),
    }

    for model_name in base_model_names:

        pred_part = prediction_dict[model_name].copy()

        pred_part.columns = [
            f"{model_name}_{drug}"
            for drug in pred_part.columns
        ]

        feature_parts.append(pred_part)

        feature_stats[f"n_{model_name}_features"] = int(
            pred_part.shape[1]
        )

    X_node_features = pd.concat(
        feature_parts,
        axis=1,
    )

    X_node_features = X_node_features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_node_features = X_node_features.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    X_node_features = X_node_features.fillna(0.0)

    X_node_features.index.name = "spot_id"

    feature_stats["n_total_node_features"] = int(
        X_node_features.shape[1]
    )

    log_print(
        f"Node feature shape: {X_node_features.shape}",
        log_file,
    )

    log_print(
        f"PCA features: {feature_stats['n_pca_features']}",
        log_file,
    )

    for model_name in base_model_names:
        log_print(
            f"{model_name} prediction features: "
            f"{feature_stats[f'n_{model_name}_features']}",
            log_file,
        )

    log_print(
        f"Total node features: "
        f"{feature_stats['n_total_node_features']}",
        log_file,
    )

    return X_node_features, feature_stats


# ============================================================
# 8. POSITION ARRAY
# ============================================================

def get_position_array(
    positions_df: pd.DataFrame,
    normalize: bool = NORMALIZE_POSITIONS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert positions dataframe to position array.

    If normalize=True, z-score normalize x and y.

    Parameters
    ----------
    positions_df : pd.DataFrame
        Position dataframe.
    normalize : bool
        Whether to normalize coordinates.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Position array and position statistics.
    """
    raw_pos_array = positions_df[
        [
            "x",
            "y",
        ]
    ].to_numpy(dtype=np.float32)

    x_values = raw_pos_array[:, 0]
    y_values = raw_pos_array[:, 1]

    x_mean = float(np.mean(x_values))
    y_mean = float(np.mean(y_values))

    x_std = float(np.std(x_values))
    y_std = float(np.std(y_values))

    if x_std == 0:
        x_std = 1.0

    if y_std == 0:
        y_std = 1.0

    position_stats = {
        "x_min": float(np.min(x_values)),
        "x_max": float(np.max(x_values)),
        "y_min": float(np.min(y_values)),
        "y_max": float(np.max(y_values)),
        "x_mean": x_mean,
        "y_mean": y_mean,
        "x_std": x_std,
        "y_std": y_std,
        "normalized": bool(normalize),
    }

    if normalize:
        pos_array = raw_pos_array.copy()
        pos_array[:, 0] = (
            pos_array[:, 0] - x_mean
        ) / x_std
        pos_array[:, 1] = (
            pos_array[:, 1] - y_mean
        ) / y_std
    else:
        pos_array = raw_pos_array.copy()

    return (
        pos_array.astype(np.float32),
        position_stats,
    )


# ============================================================
# 9. BUILD KNN EDGES
# ============================================================

def remove_self_loops(
    edge_index: np.ndarray,
) -> np.ndarray:
    """
    Remove self-loops from edge index.

    Parameters
    ----------
    edge_index : np.ndarray
        Edge index with shape 2 x n_edges.

    Returns
    -------
    np.ndarray
        Edge index without self-loops.
    """
    mask = edge_index[0] != edge_index[1]

    return edge_index[:, mask]


def make_edges_undirected(
    edge_index: np.ndarray,
) -> np.ndarray:
    """
    Add reverse edges and remove duplicate edges.

    Parameters
    ----------
    edge_index : np.ndarray
        Directed edge index.

    Returns
    -------
    np.ndarray
        Undirected edge index.
    """
    reverse_edges = edge_index[
        [
            1,
            0,
        ],
        :,
    ]

    merged_edges = np.concatenate(
        [
            edge_index,
            reverse_edges,
        ],
        axis=1,
    )

    merged_edges = np.unique(
        merged_edges.T,
        axis=0,
    ).T

    return merged_edges


def build_knn_edges(
    pos_array: np.ndarray,
    k_neighbors: int,
    include_self_loop: bool,
    make_undirected: bool,
    log_file: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build KNN edge index from spatial coordinates.

    Parameters
    ----------
    pos_array : np.ndarray
        Position array with shape n_nodes x 2.
    k_neighbors : int
        Number of neighbors.
    include_self_loop : bool
        Whether to include self-loop.
    make_undirected : bool
        Whether to create undirected graph.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Edge index and edge statistics.
    """
    n_nodes = int(pos_array.shape[0])

    if n_nodes < 2:
        raise ValueError("Cần ít nhất 2 nodes để tạo KNN graph.")

    effective_k = min(
        int(k_neighbors),
        n_nodes - 1,
    )

    n_neighbors_query = (
        effective_k + 1
        if not include_self_loop
        else effective_k
    )

    n_neighbors_query = min(
        n_neighbors_query,
        n_nodes,
    )

    knn = NearestNeighbors(
        n_neighbors=n_neighbors_query,
        metric="euclidean",
    )

    knn.fit(pos_array)

    _, neighbor_indices = knn.kneighbors(
        pos_array
    )

    src_nodes = []
    dst_nodes = []

    for node_idx in range(n_nodes):

        for neighbor_idx in neighbor_indices[node_idx]:

            if (
                not include_self_loop
                and node_idx == int(neighbor_idx)
            ):
                continue

            src_nodes.append(node_idx)
            dst_nodes.append(int(neighbor_idx))

    edge_index = np.array(
        [
            src_nodes,
            dst_nodes,
        ],
        dtype=np.int64,
    )

    if not include_self_loop:
        edge_index = remove_self_loops(
            edge_index
        )

    if make_undirected:
        edge_index = make_edges_undirected(
            edge_index
        )

    edge_index = np.unique(
        edge_index.T,
        axis=0,
    ).T

    n_edges = int(edge_index.shape[1])
    average_degree = float(n_edges / n_nodes)

    edge_stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "average_degree": average_degree,
        "k_neighbors": int(k_neighbors),
        "effective_k_neighbors": int(effective_k),
        "include_self_loop": bool(include_self_loop),
        "make_undirected": bool(make_undirected),
    }

    log_print(
        f"Edge index shape: {edge_index.shape}",
        log_file,
    )

    log_print(
        f"n_nodes: {n_nodes}",
        log_file,
    )

    log_print(
        f"n_edges: {n_edges}",
        log_file,
    )

    log_print(
        f"average_degree: {average_degree:.4f}",
        log_file,
    )

    return (
        edge_index,
        edge_stats,
    )
    
    
# ============================================================
# 10. CREATE GRAPH OBJECT
# ============================================================

def create_graph_object(
    X_node_features: pd.DataFrame,
    y_target: pd.DataFrame,
    pos_array: np.ndarray,
    edge_index: np.ndarray,
    spot_ids: list[str],
    drug_names: list[str],
    dataset_name: str,
) -> Any:
    """
    Create PyTorch Geometric graph object.

    x:
        PCA + Ridge + XGBoost + DNN

    y:
        Mean baseline IC50

    Returns
    -------
    Data
        PyG graph object.
    """
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "torch_geometric chưa được cài đặt."
        )

    graph = Data(
        x=torch.tensor(
            X_node_features.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        y=torch.tensor(
            y_target.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        pos=torch.tensor(
            pos_array,
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            edge_index,
            dtype=torch.long,
        ),
    )

    graph.spot_ids = list(spot_ids)
    graph.drug_names = list(drug_names)
    graph.feature_names = list(X_node_features.columns)
    graph.dataset_name = str(dataset_name)
    graph.feature_sources = FEATURE_SOURCES
    graph.target_source = TARGET_SOURCE

    return graph


# ============================================================
# 11. SAVE GRAPH
# ============================================================

def save_graph(
    graph: Any,
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    """
    Save graph to .pt file.
    """
    torch.save(
        graph,
        output_file,
    )

    log_print(
        f"Saved graph: {output_file}",
        log_file,
    )


# ============================================================
# 12. CREATE SUMMARY
# ============================================================

def create_graph_summary(
    dataset_name: str,
    graph: Any,
    sync_stats: dict[str, Any],
    feature_stats: dict[str, Any],
    edge_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Create graph summary row.
    """
    summary = {
        "dataset_name": dataset_name,
        "n_nodes": int(graph.x.shape[0]),
        "n_node_features": int(graph.x.shape[1]),
        "n_drugs": int(graph.y.shape[1]),
        "n_edges": int(graph.edge_index.shape[1]),
        "average_degree": float(
            edge_stats["average_degree"]
        ),
        "n_common_spots": int(
            sync_stats["n_common_spots"]
        ),
        "n_common_drugs": int(
            sync_stats["n_common_drugs"]
        ),
        "n_pca_features": int(
            feature_stats["n_pca_features"]
        ),
        "n_ridge_features": int(
            feature_stats["n_ridge_features"]
        ),
        "n_xgboost_features": int(
            feature_stats["n_xgboost_features"]
        ),
        "n_dnn_features": int(
            feature_stats["n_dnn_features"]
        ),
        "n_total_node_features": int(
            feature_stats["n_total_node_features"]
        ),
        "graph_method": GRAPH_METHOD,
        "k_neighbors": int(K_NEIGHBORS),
    }

    return summary


# ============================================================
# 13. PROCESS ONE DATASET
# ============================================================

def process_one_dataset(
    dataset_name: str,
    paths: dict[str, Path],
    log_file: Path | None = None,
) -> dict[str, Any] | None:
    """
    Build graph for one dataset.
    """
    log_print("=" * 70, log_file)
    log_print(
        f"DATASET: {dataset_name}",
        log_file,
    )
    log_print("=" * 70, log_file)

    try:
        dataset_dir = (
            paths["spatial_processed_dir"]
            / dataset_name
        )

        positions_df = read_positions(
            dataset_dir / "spot_positions.csv",
            log_file,
        )

        X_pca = read_matrix(
            dataset_dir / "X_spot_pca.csv",
            log_file,
        )

        prediction_dict = read_model_predictions(
            dataset_dir=dataset_dir,
            base_model_names=BASE_MODEL_NAMES,
            log_file=log_file,
        )

        mean_baseline = read_matrix(
            dataset_dir / "predicted_ic50_mean_baseline.csv",
            log_file,
        )

        (
            positions_df,
            X_pca,
            prediction_dict,
            mean_baseline,
            sync_stats,
        ) = sync_graph_inputs(
            positions_df=positions_df,
            X_pca=X_pca,
            prediction_dict=prediction_dict,
            mean_baseline=mean_baseline,
            log_file=log_file,
        )

        X_node_features, feature_stats = (
            build_node_feature_matrix(
                X_pca=X_pca,
                prediction_dict=prediction_dict,
                base_model_names=BASE_MODEL_NAMES,
                log_file=log_file,
            )
        )

        pos_array, position_stats = (
            get_position_array(
                positions_df=positions_df,
                normalize=NORMALIZE_POSITIONS,
            )
        )

        edge_index, edge_stats = (
            build_knn_edges(
                pos_array=pos_array,
                k_neighbors=K_NEIGHBORS,
                include_self_loop=INCLUDE_SELF_LOOP,
                make_undirected=MAKE_UNDIRECTED,
                log_file=log_file,
            )
        )

        graph = create_graph_object(
            X_node_features=X_node_features,
            y_target=mean_baseline,
            pos_array=pos_array,
            edge_index=edge_index,
            spot_ids=list(X_node_features.index),
            drug_names=list(mean_baseline.columns),
            dataset_name=dataset_name,
        )

        graph_file = (
            dataset_dir
            / "spatial_graph.pt"
        )

        save_graph(
            graph=graph,
            output_file=graph_file,
            log_file=log_file,
        )

        summary = create_graph_summary(
            dataset_name=dataset_name,
            graph=graph,
            sync_stats=sync_stats,
            feature_stats=feature_stats,
            edge_stats=edge_stats,
        )

        summary_df = pd.DataFrame(
            [summary]
        )

        summary_file = (
            dataset_dir
            / "graph_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False,
        )

        log_print(
            f"Graph x shape: {tuple(graph.x.shape)}",
            log_file,
        )

        log_print(
            f"Graph y shape: {tuple(graph.y.shape)}",
            log_file,
        )

        log_print(
            f"Graph pos shape: {tuple(graph.pos.shape)}",
            log_file,
        )

        log_print(
            f"Graph edge shape: {tuple(graph.edge_index.shape)}",
            log_file,
        )

        log_print(
            f"Graph file: {graph_file}",
            log_file,
        )

        log_print(
            f"Summary file: {summary_file}",
            log_file,
        )

        log_print("", log_file)

        return summary

    except Exception as error:

        log_print(
            f"[ERROR] Failed to process dataset "
            f"{dataset_name}: {error}",
            log_file,
        )

        log_print("", log_file)

        return None


# ============================================================
# 14. HELPER FOR APP
# ============================================================

def build_graph_for_processed_dataset(
    positions_df: pd.DataFrame,
    X_pca: pd.DataFrame,
    prediction_dict: dict[str, pd.DataFrame],
    mean_baseline: pd.DataFrame,
    dataset_name: str,
) -> Any:
    """
    Build graph directly from memory.

    Used by Streamlit/demo.
    """
    (
        positions_df,
        X_pca,
        prediction_dict,
        mean_baseline,
        _,
    ) = sync_graph_inputs(
        positions_df=positions_df,
        X_pca=X_pca,
        prediction_dict=prediction_dict,
        mean_baseline=mean_baseline,
        log_file=None,
    )

    X_node_features, _ = (
        build_node_feature_matrix(
            X_pca=X_pca,
            prediction_dict=prediction_dict,
            base_model_names=BASE_MODEL_NAMES,
            log_file=None,
        )
    )

    pos_array, _ = get_position_array(
        positions_df,
        normalize=NORMALIZE_POSITIONS,
    )

    edge_index, _ = build_knn_edges(
        pos_array=pos_array,
        k_neighbors=K_NEIGHBORS,
        include_self_loop=INCLUDE_SELF_LOOP,
        make_undirected=MAKE_UNDIRECTED,
        log_file=None,
    )

    graph = create_graph_object(
        X_node_features=X_node_features,
        y_target=mean_baseline,
        pos_array=pos_array,
        edge_index=edge_index,
        spot_ids=list(X_node_features.index),
        drug_names=list(mean_baseline.columns),
        dataset_name=dataset_name,
    )

    return graph


# ============================================================
# 15. MAIN
# ============================================================

def main() -> None:

    paths = setup_paths()

    log_file = (
        paths["log_file"]
        if SAVE_LOG
        else None
    )

    log_print("=" * 70, log_file)
    log_print(
        "BUILD SPATIAL GRAPH",
        log_file,
    )
    log_print("=" * 70, log_file)

    log_print(
        f"PROJECT_ROOT: {paths['project_root']}",
        log_file,
    )

    log_print(
        f"SPATIAL_PROCESSED_DIR: "
        f"{paths['spatial_processed_dir']}",
        log_file,
    )

    log_print(
        f"RESULTS_SPATIAL_DIR: "
        f"{paths['results_spatial_dir']}",
        log_file,
    )

    log_print(
        f"DATASETS: {DATASETS}",
        log_file,
    )

    log_print(
        f"GRAPH_METHOD: {GRAPH_METHOD}",
        log_file,
    )

    log_print(
        f"K_NEIGHBORS: {K_NEIGHBORS}",
        log_file,
    )

    log_print(
        f"FEATURE_SOURCES: {FEATURE_SOURCES}",
        log_file,
    )

    log_print(
        f"TARGET_SOURCE: {TARGET_SOURCE}",
        log_file,
    )

    log_print(
        f"HAS_TORCH_GEOMETRIC: "
        f"{HAS_TORCH_GEOMETRIC}",
        log_file,
    )

    log_print(
        f"LOG_FILE: {log_file}",
        log_file,
    )

    log_print("", log_file)

    if not HAS_TORCH_GEOMETRIC:

        log_print(
            "[ERROR] torch_geometric chưa được cài đặt.",
            log_file,
        )

        return

    report_rows = []

    for dataset_name in DATASETS:

        summary = process_one_dataset(
            dataset_name=dataset_name,
            paths=paths,
            log_file=log_file,
        )

        if summary is not None:
            report_rows.append(summary)

    if len(report_rows) == 0:

        log_print(
            "[WARNING] No dataset was processed successfully.",
            log_file,
        )

        return

    report_df = pd.DataFrame(
        report_rows
    )

    report_file = (
        paths["results_spatial_dir"]
        / "spatial_graph_report.csv"
    )

    report_df.to_csv(
        report_file,
        index=False,
    )

    log_print(
        f"Saved report: {report_file}",
        log_file,
    )

    log_print("", log_file)

    log_print("=" * 70, log_file)
    log_print(
        "FINISHED BUILD SPATIAL GRAPH",
        log_file,
    )
    log_print("=" * 70, log_file)


if __name__ == "__main__":
    main()