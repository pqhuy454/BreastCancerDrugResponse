# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from datetime import datetime
from typing import Any
import random
import copy
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


try:
    from torch_geometric.nn import SAGEConv, GCNConv

    HAS_TORCH_GEOMETRIC = True

except ImportError:
    SAGEConv = None
    GCNConv = None
    HAS_TORCH_GEOMETRIC = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRAIN_DATASETS = [
    "block_a_section_1",
    "block_a_section_2",
    "visium_fresh_frozen_whole_transcriptome",
]

VAL_DATASETS = [
    "block_a_section_1_2",
    "block_a_section_2_2",
]

MODEL_NAME = "GraphSAGE_IC50_Refiner"
GNN_TYPE = "graphsage"

HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.1

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.00005

MAX_EPOCHS = 500
PATIENCE = 30

LAMBDA_SMOOTH = 0.0

DEVICE = "cpu"
RANDOM_STATE = 42
DETERMINISTIC = True

SAVE_LOG = True
OVERWRITE_OUTPUTS = True

EXPECTED_TARGET_SOURCE = "mean_ic50_baseline"

REQUIRED_FEATURE_SOURCES = [
    "X_spot_pca",
    "predicted_ic50_ridge",
    "predicted_ic50_xgboost",
    "predicted_ic50_dnn",
]


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
    Setup project paths.

    Returns
    -------
    dict[str, Path]
        Dictionary of project paths.
    """
    project_root = get_project_root()

    spatial_processed_dir = project_root / "data" / "spatial_processed"
    models_dir = project_root / "models"
    results_spatial_dir = project_root / "results" / "spatial"
    figures_spatial_dir = project_root / "results" / "figures" / "spatial"
    logs_dir = project_root / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_spatial_dir.mkdir(parents=True, exist_ok=True)
    figures_spatial_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "spatial_processed_dir": spatial_processed_dir,
        "models_dir": models_dir,
        "results_spatial_dir": results_spatial_dir,
        "figures_spatial_dir": figures_spatial_dir,
        "logs_dir": logs_dir,
        "log_file": logs_dir / f"train_gnn_{timestamp}.log",
        "train_graph_files": [
            spatial_processed_dir / dataset / "spatial_graph.pt"
            for dataset in TRAIN_DATASETS
        ],
        "val_graph_files": [
            spatial_processed_dir / dataset / "spatial_graph.pt"
            for dataset in VAL_DATASETS
        ],
        "model_output_file": models_dir / "gnn_ic50_refiner.pt",
        "training_history_file": results_spatial_dir / "gnn_training_history.csv",
        "training_summary_file": results_spatial_dir / "gnn_training_summary.csv",
        "loss_curve_file": figures_spatial_dir / "gnn_loss_curve.png",
    }


# ============================================================
# 3. LOGGING
# ============================================================

def log_print(message: str = "", log_file: Path | None = None) -> None:
    """
    Print message to terminal and write message to log file.

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
# 4. SEED AND DEVICE
# ============================================================

def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set random seed.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True)


def get_device(device: str = DEVICE) -> torch.device:
    """
    Get torch device.

    Parameters
    ----------
    device : str
        Device config.

    Returns
    -------
    torch.device
        Torch device.
    """
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device)


# ============================================================
# 5. GRAPH ATTRIBUTE HELPERS
# ============================================================

def get_graph_attr(graph: Any, attr_name: str) -> Any:
    """
    Get graph attribute from PyG Data or dictionary.

    Parameters
    ----------
    graph : Any
        Graph object.
    attr_name : str
        Attribute name.

    Returns
    -------
    Any
        Attribute value.
    """
    if isinstance(graph, dict):
        if attr_name not in graph:
            raise KeyError(
                f"Graph dictionary thiếu key: {attr_name}"
            )

        return graph[attr_name]

    if not hasattr(graph, attr_name):
        raise AttributeError(
            f"Graph object thiếu attribute: {attr_name}"
        )

    return getattr(graph, attr_name)


def set_graph_attr(
    graph: Any,
    attr_name: str,
    value: Any,
) -> Any:
    """
    Set graph attribute for PyG Data or dictionary.

    Parameters
    ----------
    graph : Any
        Graph object.
    attr_name : str
        Attribute name.
    value : Any
        Attribute value.

    Returns
    -------
    Any
        Updated graph.
    """
    if isinstance(graph, dict):
        graph[attr_name] = value

    else:
        setattr(
            graph,
            attr_name,
            value,
        )

    return graph


def move_graph_to_device(
    graph: Any,
    device: torch.device,
) -> Any:
    """
    Move graph tensor attributes to device.

    Parameters
    ----------
    graph : Any
        Graph object.
    device : torch.device
        Target device.

    Returns
    -------
    Any
        Graph on device.
    """
    tensor_attrs = [
        "x",
        "y",
        "edge_index",
        "pos",
    ]

    for attr_name in tensor_attrs:

        try:
            value = get_graph_attr(
                graph,
                attr_name,
            )
        except Exception:
            continue

        if isinstance(value, torch.Tensor):
            graph = set_graph_attr(
                graph=graph,
                attr_name=attr_name,
                value=value.to(device),
            )

    return graph


# ============================================================
# 6. LOAD GRAPH
# ============================================================

def load_graph(
    graph_file: Path,
    device: torch.device,
    log_file: Path | None = None,
) -> Any:
    """
    Load graph from .pt file.

    Supports:
    - PyTorch Geometric Data
    - dictionary

    Parameters
    ----------
    graph_file : Path
        Graph file path.
    device : torch.device
        Device.
    log_file : Path | None
        Log file path.

    Returns
    -------
    Any
        Loaded graph.
    """
    if not graph_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy graph file: {graph_file}"
        )

    graph = torch.load(
        graph_file,
        map_location=device,
        weights_only=False,
    )

    graph = move_graph_to_device(
        graph=graph,
        device=device,
    )

    log_print(
        f"Loaded graph: {graph_file}",
        log_file,
    )

    return graph


def load_graph_list(
    graph_files: list[Path],
    dataset_names: list[str],
    device: torch.device,
    log_file: Path | None = None,
) -> list[Any]:
    """
    Load multiple spatial graphs.

    Parameters
    ----------
    graph_files : list[Path]
        List of graph file paths.
    dataset_names : list[str]
        List of dataset names.
    device : torch.device
        Target device.
    log_file : Path | None
        Log file path.

    Returns
    -------
    list[Any]
        Loaded graph objects.
    """
    if len(graph_files) != len(dataset_names):
        raise ValueError(
            "Số graph_files phải bằng số dataset_names. "
            f"graph_files={len(graph_files)}, dataset_names={len(dataset_names)}"
        )

    graphs: list[Any] = []

    for graph_file, dataset_name in zip(graph_files, dataset_names):
        log_print(f"Loading graph for dataset: {dataset_name}", log_file)

        graph = load_graph(
            graph_file=graph_file,
            device=device,
            log_file=log_file,
        )

        graphs.append(graph)

    return graphs


# ============================================================
# 7. VALIDATE GRAPH
# ============================================================

def validate_required_graph_attrs(
    graph: Any,
    graph_name: str,
) -> None:
    """
    Validate required graph attributes.

    Parameters
    ----------
    graph : Any
        Graph object.
    graph_name : str
        Graph name for error message.
    """
    required_attrs = [
        "x",
        "y",
        "edge_index",
        "spot_ids",
        "drug_names",
        "feature_names",
        "feature_sources",
        "target_source",
    ]

    for attr_name in required_attrs:
        try:
            _ = get_graph_attr(
                graph,
                attr_name,
            )
        except Exception as error:
            raise ValueError(
                f"{graph_name} graph thiếu attribute/key bắt buộc: "
                f"{attr_name}. Error: {error}"
            ) from error


def validate_feature_sources(
    feature_sources: list[str],
    graph_name: str,
) -> None:
    """
    Validate graph feature sources.

    Parameters
    ----------
    feature_sources : list[str]
        Feature sources in graph.
    graph_name : str
        Graph name.
    """
    missing_sources = [
        source
        for source in REQUIRED_FEATURE_SOURCES
        if source not in feature_sources
    ]

    if missing_sources:
        raise ValueError(
            f"{graph_name} graph thiếu feature_sources: {missing_sources}"
        )


def validate_graph(
    graph: Any,
    graph_name: str,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """
    Validate graph object.

    Required:
    - x
    - y
    - edge_index
    - spot_ids
    - drug_names
    - feature_names
    - feature_sources
    - target_source

    Parameters
    ----------
    graph : Any
        Graph object.
    graph_name : str
        Graph display name.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, Any]
        Graph statistics.
    """
    validate_required_graph_attrs(
        graph=graph,
        graph_name=graph_name,
    )

    x = get_graph_attr(
        graph,
        "x",
    )

    y = get_graph_attr(
        graph,
        "y",
    )

    edge_index = get_graph_attr(
        graph,
        "edge_index",
    )

    spot_ids = list(
        get_graph_attr(
            graph,
            "spot_ids",
        )
    )

    drug_names = list(
        get_graph_attr(
            graph,
            "drug_names",
        )
    )

    feature_names = list(
        get_graph_attr(
            graph,
            "feature_names",
        )
    )

    feature_sources = list(
        get_graph_attr(
            graph,
            "feature_sources",
        )
    )

    target_source = str(
        get_graph_attr(
            graph,
            "target_source",
        )
    )

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{graph_name}: x không phải torch.Tensor.")

    if not isinstance(y, torch.Tensor):
        raise TypeError(f"{graph_name}: y không phải torch.Tensor.")

    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(
            f"{graph_name}: edge_index không phải torch.Tensor."
        )

    if x.ndim != 2:
        raise ValueError(f"{graph_name}: x phải là tensor 2D.")

    if y.ndim != 2:
        raise ValueError(f"{graph_name}: y phải là tensor 2D.")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"{graph_name}: edge_index phải có shape 2 x n_edges."
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"{graph_name}: x và y khác số node. "
            f"x={x.shape[0]}, y={y.shape[0]}"
        )

    if x.shape[0] != len(spot_ids):
        raise ValueError(
            f"{graph_name}: số spot_ids không khớp số node. "
            f"x={x.shape[0]}, spot_ids={len(spot_ids)}"
        )

    if y.shape[1] != len(drug_names):
        raise ValueError(
            f"{graph_name}: số drug_names không khớp output_dim. "
            f"y={y.shape[1]}, drug_names={len(drug_names)}"
        )

    if x.shape[1] != len(feature_names):
        raise ValueError(
            f"{graph_name}: số feature_names không khớp input_dim. "
            f"x={x.shape[1]}, feature_names={len(feature_names)}"
        )

    if target_source != EXPECTED_TARGET_SOURCE:
        raise ValueError(
            f"{graph_name}: target_source không đúng. "
            f"Expected={EXPECTED_TARGET_SOURCE}, got={target_source}"
        )

    validate_feature_sources(
        feature_sources=feature_sources,
        graph_name=graph_name,
    )

    if not torch.is_floating_point(x):
        x = x.float()
        graph = set_graph_attr(
            graph,
            "x",
            x,
        )

    if not torch.is_floating_point(y):
        y = y.float()
        graph = set_graph_attr(
            graph,
            "y",
            y,
        )

    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
        graph = set_graph_attr(
            graph,
            "edge_index",
            edge_index,
        )

    n_nodes = int(x.shape[0])
    n_edges = int(edge_index.shape[1])
    input_dim = int(x.shape[1])
    output_dim = int(y.shape[1])

    stats = {
        "graph_name": graph_name,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "n_drugs": int(len(drug_names)),
        "n_features": int(len(feature_names)),
        "feature_sources": feature_sources,
        "target_source": target_source,
    }

    log_print(f"{graph_name} graph:", log_file)
    log_print(f"n_nodes: {n_nodes}", log_file)
    log_print(f"n_edges: {n_edges}", log_file)
    log_print(f"input_dim: {input_dim}", log_file)
    log_print(f"output_dim: {output_dim}", log_file)
    log_print(f"feature_sources: {feature_sources}", log_file)
    log_print(f"target_source: {target_source}", log_file)
    log_print("", log_file)

    return stats


def validate_graph_list(
    graphs: list[Any],
    dataset_names: list[str],
    split_name: str,
    log_file: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Validate a list of graphs and check graph compatibility.

    All graphs in one split must share:
    - input_dim
    - output_dim
    - n_drugs
    - n_features
    - feature_sources
    - target_source

    Parameters
    ----------
    graphs : list[Any]
        Graph objects.
    dataset_names : list[str]
        Dataset names.
    split_name : str
        Split name, such as Train or Validation.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Per-graph stats and aggregated split stats.
    """
    if len(graphs) == 0:
        raise ValueError(f"{split_name}: graph list rỗng.")

    if len(graphs) != len(dataset_names):
        raise ValueError(
            f"{split_name}: số graph khác số dataset. "
            f"graphs={len(graphs)}, dataset_names={len(dataset_names)}"
        )

    stats_list: list[dict[str, Any]] = []

    for graph, dataset_name in zip(graphs, dataset_names):
        stats = validate_graph(
            graph=graph,
            graph_name=f"{split_name} - {dataset_name}",
            log_file=log_file,
        )

        stats["dataset_name"] = dataset_name
        stats_list.append(stats)

    reference = stats_list[0]

    for stats in stats_list[1:]:
        for key in [
            "input_dim",
            "output_dim",
            "n_drugs",
            "n_features",
            "target_source",
        ]:
            if stats[key] != reference[key]:
                raise ValueError(
                    f"{split_name}: graph không thống nhất {key}. "
                    f"Reference={reference[key]}, "
                    f"{stats['dataset_name']}={stats[key]}"
                )

        if stats["feature_sources"] != reference["feature_sources"]:
            raise ValueError(
                f"{split_name}: feature_sources không thống nhất giữa các graph."
            )

    split_stats = {
        "split_name": split_name,
        "n_graphs": int(len(graphs)),
        "dataset_names": list(dataset_names),
        "total_nodes": int(sum(stats["n_nodes"] for stats in stats_list)),
        "total_edges": int(sum(stats["n_edges"] for stats in stats_list)),
        "input_dim": int(reference["input_dim"]),
        "output_dim": int(reference["output_dim"]),
        "n_drugs": int(reference["n_drugs"]),
        "n_features": int(reference["n_features"]),
        "feature_sources": list(reference["feature_sources"]),
        "target_source": str(reference["target_source"]),
    }

    log_print(f"{split_name} graphs: {split_stats['n_graphs']}", log_file)
    log_print(f"{split_name} total nodes: {split_stats['total_nodes']}", log_file)
    log_print(f"{split_name} total edges: {split_stats['total_edges']}", log_file)
    log_print(f"{split_name} input_dim: {split_stats['input_dim']}", log_file)
    log_print(f"{split_name} output_dim: {split_stats['output_dim']}", log_file)
    log_print("", log_file)

    return stats_list, split_stats


# ============================================================
# 8. GNN MODEL
# ============================================================

class GNNIC50Refiner(nn.Module):
    """
    Graph-based ensemble refiner for IC50 prediction.

    Input:
        PCA features + Ridge predictions + XGBoost predictions + DNN predictions.

    Target:
        mean_ic50_baseline pseudo-label.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        gnn_type: str,
    ) -> None:
        """
        Initialize GNN IC50 refiner.

        Parameters
        ----------
        input_dim : int
            Number of input node features.
        hidden_dim : int
            Hidden dimension.
        output_dim : int
            Number of output drugs.
        num_layers : int
            Number of GNN layers.
        dropout : float
            Dropout rate.
        gnn_type : str
            GNN type: graphsage or gcn.
        """
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric chưa được cài đặt.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()

        if self.gnn_type not in ["graphsage", "gcn"]:
            raise ValueError(
                f"GNN_TYPE không hỗ trợ: {gnn_type}"
            )

        conv_class = (
            SAGEConv
            if self.gnn_type == "graphsage"
            else GCNConv
        )

        self.convs = nn.ModuleList()

        self.convs.append(
            conv_class(
                input_dim,
                hidden_dim,
            )
        )

        if num_layers >= 2:
            for _ in range(num_layers - 1):
                self.convs.append(
                    conv_class(
                        hidden_dim,
                        hidden_dim,
                    )
                )

        self.output_layer = nn.Linear(
            hidden_dim,
            output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Graph edges.

        Returns
        -------
        torch.Tensor
            Refined IC50 predictions.
        """
        h = x

        for conv in self.convs:
            h = conv(
                h,
                edge_index,
            )

            h = F.relu(h)

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

        output = self.output_layer(h)

        return output
    
    
# ============================================================
# 9. LOSS FUNCTIONS
# ============================================================

def mse_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Prediction loss.

    Parameters
    ----------
    prediction : torch.Tensor
        Model prediction.
    target : torch.Tensor
        Target IC50.

    Returns
    -------
    torch.Tensor
        MSE loss.
    """
    return F.mse_loss(
        prediction,
        target,
    )


def spatial_smoothness_loss(
    prediction: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial smoothness regularization.

    Neighboring spots should have similar
    refined IC50 predictions.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted IC50.
    edge_index : torch.Tensor
        Graph edges.

    Returns
    -------
    torch.Tensor
        Smoothness loss.
    """
    if edge_index.shape[1] == 0:
        return torch.tensor(
            0.0,
            device=prediction.device,
        )

    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    src_pred = prediction[src_nodes]
    dst_pred = prediction[dst_nodes]

    smooth_loss = torch.mean(
        (src_pred - dst_pred) ** 2
    )

    return smooth_loss


def compute_total_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
    lambda_smooth: float = LAMBDA_SMOOTH,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute total training loss.

    Total loss:

    loss =
        mse_loss
        +
        lambda_smooth * smoothness_loss

    Parameters
    ----------
    prediction : torch.Tensor
        Model prediction.
    target : torch.Tensor
        Ground truth target.
    edge_index : torch.Tensor
        Graph edges.
    lambda_smooth : float
        Smoothness weight.

    Returns
    -------
    tuple
        total_loss,
        mse_loss,
        smoothness_loss
    """
    mse_loss = mse_prediction_loss(
        prediction=prediction,
        target=target,
    )

    smooth_loss = spatial_smoothness_loss(
        prediction=prediction,
        edge_index=edge_index,
    )

    total_loss = (
        mse_loss
        + lambda_smooth * smooth_loss
    )

    return (
        total_loss,
        mse_loss,
        smooth_loss,
    )


# ============================================================
# 10. METRIC FUNCTIONS
# ============================================================

def compute_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute MAE.
    """
    return float(
        torch.mean(
            torch.abs(
                prediction - target
            )
        ).item()
    )


def compute_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute RMSE.
    """
    mse = torch.mean(
        (prediction - target) ** 2
    )

    rmse = torch.sqrt(
        mse
    )

    return float(
        rmse.item()
    )


def average_metric_dicts(
    metric_rows: list[dict[str, float]],
) -> dict[str, float]:
    """
    Average a list of metric dictionaries.

    Parameters
    ----------
    metric_rows : list[dict[str, float]]
        Metric dictionaries.

    Returns
    -------
    dict[str, float]
        Averaged metrics.
    """
    if len(metric_rows) == 0:
        raise ValueError("metric_rows rỗng.")

    keys = list(metric_rows[0].keys())

    averaged = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in keys
    }

    return averaged


# ============================================================
# 11. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    train_graphs: list[Any],
    optimizer: torch.optim.Optimizer,
    lambda_smooth: float,
) -> dict[str, float]:
    """
    Train one epoch using multiple training graphs.

    Parameters
    ----------
    model : nn.Module
        GNN model.
    train_graphs : list[Any]
        Training graphs.
    optimizer : torch.optim.Optimizer
        Optimizer.
    lambda_smooth : float
        Smoothness weight.

    Returns
    -------
    dict[str, float]
        Averaged training metrics across all train graphs.
    """
    if len(train_graphs) == 0:
        raise ValueError("train_graphs rỗng.")

    model.train()
    optimizer.zero_grad()

    total_loss_all = torch.tensor(
        0.0,
        device=get_graph_attr(train_graphs[0], "x").device,
    )

    metric_rows: list[dict[str, float]] = []

    for graph in train_graphs:
        x = get_graph_attr(graph, "x")
        y = get_graph_attr(graph, "y")
        edge_index = get_graph_attr(graph, "edge_index")

        prediction = model(
            x,
            edge_index,
        )

        (
            total_loss,
            mse_loss,
            smooth_loss,
        ) = compute_total_loss(
            prediction=prediction,
            target=y,
            edge_index=edge_index,
            lambda_smooth=lambda_smooth,
        )

        total_loss_all = total_loss_all + total_loss

        metric_rows.append(
            {
                "loss": float(total_loss.detach().item()),
                "mse_loss": float(mse_loss.detach().item()),
                "smooth_loss": float(smooth_loss.detach().item()),
                "mae": compute_mae(
                    prediction=prediction.detach(),
                    target=y,
                ),
                "rmse": compute_rmse(
                    prediction=prediction.detach(),
                    target=y,
                ),
            }
        )

    mean_loss = total_loss_all / len(train_graphs)
    mean_loss.backward()
    optimizer.step()

    return average_metric_dicts(metric_rows)


# ============================================================
# 12. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_graphs: list[Any],
    lambda_smooth: float,
) -> dict[str, float]:
    """
    Evaluate model using multiple validation graphs.

    Parameters
    ----------
    model : nn.Module
        GNN model.
    val_graphs : list[Any]
        Validation graphs.
    lambda_smooth : float
        Smoothness weight.

    Returns
    -------
    dict[str, float]
        Averaged validation metrics across all validation graphs.
    """
    if len(val_graphs) == 0:
        raise ValueError("val_graphs rỗng.")

    model.eval()

    metric_rows: list[dict[str, float]] = []

    for graph in val_graphs:
        x = get_graph_attr(graph, "x")
        y = get_graph_attr(graph, "y")
        edge_index = get_graph_attr(graph, "edge_index")

        prediction = model(
            x,
            edge_index,
        )

        (
            total_loss,
            mse_loss,
            smooth_loss,
        ) = compute_total_loss(
            prediction=prediction,
            target=y,
            edge_index=edge_index,
            lambda_smooth=lambda_smooth,
        )

        metric_rows.append(
            {
                "loss": float(total_loss.item()),
                "mse_loss": float(mse_loss.item()),
                "smooth_loss": float(smooth_loss.item()),
                "mae": compute_mae(
                    prediction=prediction,
                    target=y,
                ),
                "rmse": compute_rmse(
                    prediction=prediction,
                    target=y,
                ),
            }
        )

    return average_metric_dicts(metric_rows)


# ============================================================
# 13. EARLY STOPPING
# ============================================================

def is_improved(
    current_loss: float,
    best_loss: float,
    min_delta: float = 1e-6,
) -> bool:
    """
    Check whether validation loss improved.

    Parameters
    ----------
    current_loss : float
        Current validation loss.
    best_loss : float
        Best validation loss.
    min_delta : float
        Minimum improvement.

    Returns
    -------
    bool
        True if improved.
    """
    return (
        current_loss
        < best_loss - min_delta
    )


def create_history_row(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict[str, Any]:
    """
    Create one history row.

    Parameters
    ----------
    epoch : int
        Current epoch.
    train_metrics : dict[str, float]
        Training metrics.
    val_metrics : dict[str, float]
        Validation metrics.

    Returns
    -------
    dict[str, Any]
        History row.
    """
    return {
        "epoch": int(epoch),

        "train_loss":
            train_metrics["loss"],

        "train_mse":
            train_metrics["mse_loss"],

        "train_smooth":
            train_metrics["smooth_loss"],

        "train_mae":
            train_metrics["mae"],

        "train_rmse":
            train_metrics["rmse"],

        "val_loss":
            val_metrics["loss"],

        "val_mse":
            val_metrics["mse_loss"],

        "val_smooth":
            val_metrics["smooth_loss"],

        "val_mae":
            val_metrics["mae"],

        "val_rmse":
            val_metrics["rmse"],
    }


# ============================================================
# 14. MODEL INITIALIZATION
# ============================================================

def build_model(
    input_dim: int,
    output_dim: int,
) -> GNNIC50Refiner:
    """
    Build GNN model.

    Parameters
    ----------
    input_dim : int
        Number of node features.
    output_dim : int
        Number of drugs.

    Returns
    -------
    GNNIC50Refiner
        Initialized model.
    """
    model = GNNIC50Refiner(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        gnn_type=GNN_TYPE,
    )

    return model


def build_optimizer(
    model: nn.Module,
) -> torch.optim.Optimizer:
    """
    Build optimizer.

    Parameters
    ----------
    model : nn.Module
        Model.

    Returns
    -------
    torch.optim.Optimizer
        Adam optimizer.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


# ============================================================
# 15. TRAINING LOOP
# ============================================================

def train_gnn_model(
    model: nn.Module,
    train_graphs: list[Any],
    val_graphs: list[Any],
    log_file: Path | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    """
    Train GNN model with early stopping.

    Parameters
    ----------
    model : nn.Module
        GNN model.
    train_graphs : list[Any]
        Training graphs.
    val_graphs : list[Any]
        Validation graphs.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]
        Best model, training history, and best information.
    """
    optimizer = build_optimizer(model)

    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_val_mse_loss = float("inf")
    best_val_smoothness_loss = float("inf")

    patience_counter = 0
    history: list[dict[str, Any]] = []

    training_start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        epoch_start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            train_graphs=train_graphs,
            optimizer=optimizer,
            lambda_smooth=LAMBDA_SMOOTH,
        )

        val_metrics = evaluate_model(
            model=model,
            val_graphs=val_graphs,
            lambda_smooth=LAMBDA_SMOOTH,
        )

        elapsed_seconds = time.time() - epoch_start_time

        history_row = create_history_row(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        history_row["elapsed_seconds"] = float(elapsed_seconds)

        history.append(history_row)

        current_val_loss = float(
            val_metrics["loss"]
        )

        if is_improved(
            current_loss=current_val_loss,
            best_loss=best_val_loss,
        ):

            best_val_loss = current_val_loss
            best_train_loss = float(
                train_metrics["loss"]
            )
            best_val_mse_loss = float(
                val_metrics["mse_loss"]
            )
            best_val_smoothness_loss = float(
                val_metrics["smooth_loss"]
            )
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            log_print(
                f"[BEST] Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f}",
                log_file,
            )

        else:

            patience_counter += 1

        should_log = (
            epoch == 1
            or epoch % 10 == 0
            or epoch == MAX_EPOCHS
        )

        if should_log:

            log_print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"train_mse={train_metrics['mse_loss']:.6f} | "
                f"train_smooth={train_metrics['smooth_loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_mse={val_metrics['mse_loss']:.6f} | "
                f"val_smooth={val_metrics['smooth_loss']:.6f}",
                log_file,
            )

        if patience_counter >= PATIENCE:

            log_print(
                f"Early stopping at epoch {epoch}",
                log_file,
            )

            break

    total_training_time = time.time() - training_start_time

    model.load_state_dict(
        best_model_state
    )

    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_train_loss": float(best_train_loss),
        "best_val_mse_loss": float(best_val_mse_loss),
        "best_val_smoothness_loss": float(best_val_smoothness_loss),
        "training_time_seconds": float(total_training_time),
    }

    log_print("", log_file)
    log_print(
        f"Best epoch: {best_epoch}",
        log_file,
    )
    log_print(
        f"Best val loss: {best_val_loss:.6f}",
        log_file,
    )
    log_print(
        f"Training time seconds: {total_training_time:.2f}",
        log_file,
    )
    log_print("", log_file)

    return (
        model,
        history,
        best_info,
    )


# ============================================================
# 16. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model: nn.Module,
    output_file: Path,
    best_info: dict[str, Any],
    train_graphs: list[Any],
    val_graphs: list[Any],
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    log_file: Path | None = None,
) -> None:
    """
    Save best GNN model checkpoint.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    output_file : Path
        Output checkpoint file.
    best_info : dict[str, Any]
        Best epoch information.
    train_graphs : list[Any]
        Train graphs.
    val_graphs : list[Any]
        Validation graphs.
    train_stats : dict[str, Any]
        Aggregated train graph statistics.
    val_stats : dict[str, Any]
        Aggregated validation graph statistics.
    log_file : Path | None
        Log file path.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_file.exists() and not OVERWRITE_OUTPUTS:

        log_print(
            f"[WARNING] Model file exists, skip saving: {output_file}",
            log_file,
        )

        return

    reference_graph = train_graphs[0]

    checkpoint = {
        "model_name": MODEL_NAME,
        "gnn_type": GNN_TYPE,
        "model_state_dict": model.state_dict(),
        "input_dim": int(train_stats["input_dim"]),
        "hidden_dim": int(HIDDEN_DIM),
        "output_dim": int(train_stats["output_dim"]),
        "num_layers": int(NUM_LAYERS),
        "dropout": float(DROPOUT),
        "lambda_smooth": float(LAMBDA_SMOOTH),
        "best_epoch": int(best_info["best_epoch"]),
        "best_val_loss": float(best_info["best_val_loss"]),
        "best_train_loss": float(best_info["best_train_loss"]),
        "best_val_mse_loss": float(best_info["best_val_mse_loss"]),
        "best_val_smoothness_loss": float(
            best_info["best_val_smoothness_loss"]
        ),
        "training_time_seconds": float(
            best_info["training_time_seconds"]
        ),
        "train_datasets": list(TRAIN_DATASETS),
        "val_datasets": list(VAL_DATASETS),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "drug_names": list(
            get_graph_attr(
                reference_graph,
                "drug_names",
            )
        ),
        "feature_names": list(
            get_graph_attr(
                reference_graph,
                "feature_names",
            )
        ),
        "feature_sources": list(
            get_graph_attr(
                reference_graph,
                "feature_sources",
            )
        ),
        "target_source": str(
            get_graph_attr(
                reference_graph,
                "target_source",
            )
        ),
        "note": (
            "Graph-based ensemble refiner trained without voting. "
            "Target is mean_ic50_baseline pseudo-label. "
            "Model trained on multiple spatial graphs."
        ),
    }

    torch.save(
        checkpoint,
        output_file,
    )

    log_print(
        f"Saved model checkpoint: {output_file}",
        log_file,
    )


# ============================================================
# 17. SAVE HISTORY AND SUMMARY
# ============================================================

def save_training_history(
    history: list[dict[str, Any]],
    output_file: Path,
    log_file: Path | None = None,
) -> pd.DataFrame:
    """
    Save training history to CSV.

    Parameters
    ----------
    history : list[dict[str, Any]]
        Training history.
    output_file : Path
        Output file path.
    log_file : Path | None
        Log file path.

    Returns
    -------
    pd.DataFrame
        History dataframe.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        output_file,
        index=False,
    )

    log_print(
        f"Saved training history: {output_file}",
        log_file,
    )

    return history_df


def save_training_summary(
    summary: dict[str, Any],
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    """
    Save training summary to CSV.

    Parameters
    ----------
    summary : dict[str, Any]
        Summary dictionary.
    output_file : Path
        Output file path.
    log_file : Path | None
        Log file path.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        output_file,
        index=False,
    )

    log_print(
        f"Saved training summary: {output_file}",
        log_file,
    )


# ============================================================
# 18. PLOT LOSS CURVE
# ============================================================

def plot_loss_curve(
    history_df: pd.DataFrame,
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    """
    Plot train and validation loss curve.

    Parameters
    ----------
    history_df : pd.DataFrame
        Training history dataframe.
    output_file : Path
        Output image file.
    log_file : Path | None
        Log file path.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        "GNN Training Loss Curve"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    log_print(
        f"Saved loss curve: {output_file}",
        log_file,
    )


# ============================================================
# 19. VALIDATE TRAIN VAL COMPATIBILITY
# ============================================================

def validate_train_val_compatibility(
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
) -> None:
    """
    Validate train and validation graph dimensions.

    Parameters
    ----------
    train_stats : dict[str, Any]
        Aggregated train graph statistics.
    val_stats : dict[str, Any]
        Aggregated validation graph statistics.
    """
    if train_stats["input_dim"] != val_stats["input_dim"]:
        raise ValueError(
            "Train/Val input_dim mismatch: "
            f"{train_stats['input_dim']} vs {val_stats['input_dim']}"
        )

    if train_stats["output_dim"] != val_stats["output_dim"]:
        raise ValueError(
            "Train/Val output_dim mismatch: "
            f"{train_stats['output_dim']} vs {val_stats['output_dim']}"
        )

    if train_stats["n_drugs"] != val_stats["n_drugs"]:
        raise ValueError(
            "Train/Val n_drugs mismatch: "
            f"{train_stats['n_drugs']} vs {val_stats['n_drugs']}"
        )

    if train_stats["n_features"] != val_stats["n_features"]:
        raise ValueError(
            "Train/Val n_features mismatch: "
            f"{train_stats['n_features']} vs {val_stats['n_features']}"
        )

    if train_stats["feature_sources"] != val_stats["feature_sources"]:
        raise ValueError("Train/Val feature_sources mismatch.")

    if train_stats["target_source"] != val_stats["target_source"]:
        raise ValueError("Train/Val target_source mismatch.")


# ============================================================
# 20. MAIN
# ============================================================

def main() -> None:
    """
    Main training pipeline.
    """
    paths = setup_paths()

    log_file = (
        paths["log_file"]
        if SAVE_LOG
        else None
    )

    log_print("=" * 70, log_file)
    log_print(
        "TRAIN GNN IC50 REFINER",
        log_file,
    )
    log_print("=" * 70, log_file)

    if not HAS_TORCH_GEOMETRIC:

        log_print(
            "[ERROR] torch_geometric chưa được cài đặt. "
            "Không thể train GNN.",
            log_file,
        )

        return

    set_seed(
        RANDOM_STATE
    )

    device = get_device(
        DEVICE
    )

    log_print(
        f"PROJECT_ROOT: {paths['project_root']}",
        log_file,
    )

    log_print(
        f"TRAIN_DATASETS: {TRAIN_DATASETS}",
        log_file,
    )

    log_print(
        f"VAL_DATASETS: {VAL_DATASETS}",
        log_file,
    )

    log_print(
        f"MODEL_NAME: {MODEL_NAME}",
        log_file,
    )

    log_print(
        f"GNN_TYPE: {GNN_TYPE}",
        log_file,
    )

    log_print(
        f"HIDDEN_DIM: {HIDDEN_DIM}",
        log_file,
    )

    log_print(
        f"NUM_LAYERS: {NUM_LAYERS}",
        log_file,
    )

    log_print(
        f"DROPOUT: {DROPOUT}",
        log_file,
    )

    log_print(
        f"LEARNING_RATE: {LEARNING_RATE}",
        log_file,
    )

    log_print(
        f"WEIGHT_DECAY: {WEIGHT_DECAY}",
        log_file,
    )

    log_print(
        f"LAMBDA_SMOOTH: {LAMBDA_SMOOTH}",
        log_file,
    )

    log_print(
        f"DEVICE: {device}",
        log_file,
    )

    log_print(
        f"LOG_FILE: {log_file}",
        log_file,
    )

    log_print("", log_file)

    try:

        train_graphs = load_graph_list(
            graph_files=paths["train_graph_files"],
            dataset_names=TRAIN_DATASETS,
            device=device,
            log_file=log_file,
        )

        val_graphs = load_graph_list(
            graph_files=paths["val_graph_files"],
            dataset_names=VAL_DATASETS,
            device=device,
            log_file=log_file,
        )

        (
            train_stats_list,
            train_stats,
        ) = validate_graph_list(
            graphs=train_graphs,
            dataset_names=TRAIN_DATASETS,
            split_name="Train",
            log_file=log_file,
        )

        (
            val_stats_list,
            val_stats,
        ) = validate_graph_list(
            graphs=val_graphs,
            dataset_names=VAL_DATASETS,
            split_name="Validation",
            log_file=log_file,
        )

        validate_train_val_compatibility(
            train_stats=train_stats,
            val_stats=val_stats,
        )

        input_dim = int(
            train_stats["input_dim"]
        )

        output_dim = int(
            train_stats["output_dim"]
        )

        model = build_model(
            input_dim=input_dim,
            output_dim=output_dim,
        )

        model = model.to(
            device
        )

        log_print(
            f"Train graphs: {len(train_graphs)}",
            log_file,
        )

        log_print(
            f"Validation graphs: {len(val_graphs)}",
            log_file,
        )

        log_print(
            f"Input dim: {input_dim}",
            log_file,
        )

        log_print(
            f"Output dim: {output_dim}",
            log_file,
        )

        log_print("", log_file)

        (
            best_model,
            history,
            best_info,
        ) = train_gnn_model(
            model=model,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            log_file=log_file,
        )

        save_checkpoint(
            model=best_model,
            output_file=paths["model_output_file"],
            best_info=best_info,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            train_stats=train_stats,
            val_stats=val_stats,
            log_file=log_file,
        )

        history_df = save_training_history(
            history=history,
            output_file=paths["training_history_file"],
            log_file=log_file,
        )

        summary = {
            "model_name": MODEL_NAME,
            "gnn_type": GNN_TYPE,
            "train_datasets": json.dumps(
                TRAIN_DATASETS,
                ensure_ascii=False,
            ),
            "val_datasets": json.dumps(
                VAL_DATASETS,
                ensure_ascii=False,
            ),
            "n_train_graphs": int(len(train_graphs)),
            "n_val_graphs": int(len(val_graphs)),
            "total_train_nodes": int(train_stats["total_nodes"]),
            "total_val_nodes": int(val_stats["total_nodes"]),
            "total_train_edges": int(train_stats["total_edges"]),
            "total_val_edges": int(val_stats["total_edges"]),
            "input_dim": input_dim,
            "hidden_dim": int(HIDDEN_DIM),
            "output_dim": output_dim,
            "num_layers": int(NUM_LAYERS),
            "dropout": float(DROPOUT),
            "learning_rate": float(LEARNING_RATE),
            "weight_decay": float(WEIGHT_DECAY),
            "lambda_smooth": float(LAMBDA_SMOOTH),
            "max_epochs": int(MAX_EPOCHS),
            "patience": int(PATIENCE),
            "best_epoch": int(best_info["best_epoch"]),
            "best_val_loss": float(best_info["best_val_loss"]),
            "best_train_loss": float(best_info["best_train_loss"]),
            "device": str(device),
            "feature_sources": json.dumps(
                train_stats["feature_sources"],
                ensure_ascii=False,
            ),
            "target_source": str(train_stats["target_source"]),
            "model_file": str(paths["model_output_file"]),
        }

        save_training_summary(
            summary=summary,
            output_file=paths["training_summary_file"],
            log_file=log_file,
        )

        plot_loss_curve(
            history_df=history_df,
            output_file=paths["loss_curve_file"],
            log_file=log_file,
        )

        log_print("", log_file)
        log_print("=" * 70, log_file)
        log_print(
            "FINISHED TRAIN GNN IC50 REFINER",
            log_file,
        )
        log_print("=" * 70, log_file)

        log_print(
            f"Train graphs: {len(train_graphs)}",
            log_file,
        )

        log_print(
            f"Validation graphs: {len(val_graphs)}",
            log_file,
        )

        log_print(
            f"Input dim: {input_dim}",
            log_file,
        )

        log_print(
            f"Output dim: {output_dim}",
            log_file,
        )

        log_print(
            f"Best epoch: {best_info['best_epoch']}",
            log_file,
        )

        log_print(
            f"Best val loss: {best_info['best_val_loss']:.6f}",
            log_file,
        )

    except Exception as error:

        log_print(
            f"[ERROR] Training pipeline failed: {error}",
            log_file,
        )


if __name__ == "__main__":
    main()
