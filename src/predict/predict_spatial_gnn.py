# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from datetime import datetime
from typing import Any
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from torch_geometric.nn import SAGEConv, GCNConv

    HAS_TORCH_GEOMETRIC = True

except ImportError:
    SAGEConv = None
    GCNConv = None
    HAS_TORCH_GEOMETRIC = False


# ============================================================
# 1. SAFE IMPORT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.prediction import (
    create_drug_ranking,
    save_dataframe,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASETS = [
    "ductal_carcinoma_in_situ_invasive_carcinoma_FFPE",
    "whole_transcriptome_analysis",
]

TOP_K_DRUGS = 10

DEVICE = "auto"

SAVE_REFINED_IC50 = True
SAVE_TOP_DRUG_RANKING = True
SAVE_TOP_OVERALL_DRUGS = True
SAVE_SUMMARY = True

SAVE_LOG = True
OVERWRITE_OUTPUTS = True


# ============================================================
# 3. PATH SETUP
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
    logs_dir = project_root / "logs"

    results_spatial_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return {
        "project_root": project_root,
        "spatial_processed_dir": spatial_processed_dir,
        "models_dir": models_dir,
        "results_spatial_dir": results_spatial_dir,
        "logs_dir": logs_dir,
        "log_file": logs_dir / f"predict_spatial_gnn_{timestamp}.log",
        "gnn_model_file": models_dir / "gnn_ic50_refiner.pt",
    }


# ============================================================
# 4. LOGGING
# ============================================================

def log_print(
    message: str = "",
    log_file: Path | None = None,
) -> None:
    """
    Print message to terminal and write to log file.

    Parameters
    ----------
    message : str
        Message to log.
    log_file : Path | None
        Log file path.
    """
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(
            log_file,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(str(message) + "\n")


# ============================================================
# 5. DEVICE
# ============================================================

def get_device(
    device: str = DEVICE,
) -> torch.device:
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
# 6. GRAPH HELPERS
# ============================================================

def get_graph_attr(
    graph: Any,
    attr_name: str,
) -> Any:
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

    return getattr(
        graph,
        attr_name,
    )


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
    Move graph tensors to device.

    Parameters
    ----------
    graph : Any
        Graph object.
    device : torch.device
        Device.

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


def validate_graph_metadata(
    graph: Any,
    graph_file: Path,
) -> None:
    """
    Validate required graph metadata for GNN inference.

    Parameters
    ----------
    graph : Any
        Graph object.
    graph_file : Path
        Graph file path.
    """
    required_attrs = [
        "x",
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
                f"Graph file {graph_file} thiếu metadata bắt buộc: "
                f"{attr_name}. Error: {error}"
            ) from error


def load_graph(
    graph_file: Path,
    device: torch.device,
    log_file: Path | None = None,
) -> Any:
    """
    Load spatial graph.

    Parameters
    ----------
    graph_file : Path
        Path to spatial_graph.pt.
    device : torch.device
        Torch device.
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

    validate_graph_metadata(
        graph=graph,
        graph_file=graph_file,
    )

    log_print(
        f"Loaded graph: {graph_file}",
        log_file,
    )

    return graph


# ============================================================
# 7. GNN MODEL CLASS
# ============================================================

class GNNIC50Refiner(nn.Module):
    """
    Graph-based ensemble refiner for spatial IC50 prediction.
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
            Input node feature dimension.
        hidden_dim : int
            Hidden dimension.
        output_dim : int
            Output drug dimension.
        num_layers : int
            Number of GNN layers.
        dropout : float
            Dropout rate.
        gnn_type : str
            GNN type, either graphsage or gcn.
        """
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric chưa được cài đặt."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = str(gnn_type).lower()

        if self.gnn_type not in [
            "graphsage",
            "gcn",
        ]:
            raise ValueError(
                f"Không hỗ trợ gnn_type: {gnn_type}"
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
            Node feature tensor.
        edge_index : torch.Tensor
            Edge index tensor.

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
# 8. LOAD CHECKPOINT
# ============================================================

def validate_checkpoint_keys(
    checkpoint: dict[str, Any],
) -> None:
    """
    Validate required checkpoint keys.

    Parameters
    ----------
    checkpoint : dict[str, Any]
        Loaded checkpoint.
    """
    required_keys = [
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "num_layers",
        "dropout",
        "gnn_type",
        "drug_names",
        "feature_names",
        "feature_sources",
        "target_source",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint thiếu key bắt buộc: {key}"
            )


def load_gnn_checkpoint(
    model_file: Path,
    device: torch.device,
    log_file: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load trained GNN checkpoint and rebuild model.

    Parameters
    ----------
    model_file : Path
        Path to gnn_ic50_refiner.pt.
    device : torch.device
        Torch device.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple[nn.Module, dict[str, Any]]
        Loaded model and checkpoint.
    """
    if not model_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy GNN model file: {model_file}"
        )

    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint không phải dict."
        )

    validate_checkpoint_keys(
        checkpoint
    )

    model = GNNIC50Refiner(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        gnn_type=str(checkpoint["gnn_type"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        device
    )

    model.eval()

    log_print(
        f"Loaded model: {model_file}",
        log_file,
    )

    log_print(
        f"MODEL: {checkpoint.get('model_name', 'unknown')}",
        log_file,
    )

    log_print(
        f"GNN_TYPE: {checkpoint['gnn_type']}",
        log_file,
    )

    log_print(
        f"input_dim: {checkpoint['input_dim']}",
        log_file,
    )

    log_print(
        f"hidden_dim: {checkpoint['hidden_dim']}",
        log_file,
    )

    log_print(
        f"output_dim: {checkpoint['output_dim']}",
        log_file,
    )

    log_print(
        f"num_layers: {checkpoint['num_layers']}",
        log_file,
    )

    log_print(
        f"dropout: {checkpoint['dropout']}",
        log_file,
    )

    log_print(
        f"feature_sources: {checkpoint['feature_sources']}",
        log_file,
    )

    log_print(
        f"target_source: {checkpoint['target_source']}",
        log_file,
    )

    log_print("", log_file)

    return (
        model,
        checkpoint,
    )
    
    
# ============================================================
# 9. VALIDATE GRAPH AGAINST CHECKPOINT
# ============================================================

def validate_graph_against_checkpoint(
    graph: Any,
    checkpoint: dict[str, Any],
) -> None:
    """
    Validate graph compatibility with checkpoint.

    Parameters
    ----------
    graph : Any
        Loaded graph.
    checkpoint : dict[str, Any]
        Loaded checkpoint.
    """
    x = get_graph_attr(
        graph,
        "x",
    )

    graph_drug_names = list(
        get_graph_attr(
            graph,
            "drug_names",
        )
    )

    graph_feature_names = list(
        get_graph_attr(
            graph,
            "feature_names",
        )
    )

    graph_feature_sources = list(
        get_graph_attr(
            graph,
            "feature_sources",
        )
    )

    graph_target_source = str(
        get_graph_attr(
            graph,
            "target_source",
        )
    )

    checkpoint_drug_names = list(
        checkpoint["drug_names"]
    )

    checkpoint_feature_names = list(
        checkpoint["feature_names"]
    )

    checkpoint_feature_sources = list(
        checkpoint["feature_sources"]
    )

    checkpoint_target_source = str(
        checkpoint["target_source"]
    )

    expected_input_dim = int(
        checkpoint["input_dim"]
    )

    expected_output_dim = int(
        checkpoint["output_dim"]
    )

    actual_input_dim = int(
        x.shape[1]
    )

    if actual_input_dim != expected_input_dim:
        raise ValueError(
            f"Input dimension mismatch. "
            f"Graph={actual_input_dim}, "
            f"Checkpoint={expected_input_dim}"
        )

    if len(graph_drug_names) != expected_output_dim:
        raise ValueError(
            f"Drug dimension mismatch. "
            f"Graph={len(graph_drug_names)}, "
            f"Checkpoint={expected_output_dim}"
        )

    if graph_drug_names != checkpoint_drug_names:
        raise ValueError(
            "Drug names giữa graph và checkpoint không khớp."
        )

    if graph_feature_names != checkpoint_feature_names:
        raise ValueError(
            "Feature names giữa graph và checkpoint không khớp."
        )

    if graph_feature_sources != checkpoint_feature_sources:
        raise ValueError(
            "Feature sources giữa graph và checkpoint không khớp."
        )

    if graph_target_source != checkpoint_target_source:
        raise ValueError(
            "Target source giữa graph và checkpoint không khớp."
        )


# ============================================================
# 10. GNN INFERENCE
# ============================================================

@torch.no_grad()
def predict_refined_ic50(
    model: nn.Module,
    graph: Any,
    device: torch.device,
) -> np.ndarray:
    """
    Predict refined IC50 using trained GNN.

    Parameters
    ----------
    model : nn.Module
        Trained GNN model.
    graph : Any
        Spatial graph.
    device : torch.device
        Torch device.

    Returns
    -------
    np.ndarray
        Prediction matrix.
    """
    x = get_graph_attr(
        graph,
        "x",
    )

    edge_index = get_graph_attr(
        graph,
        "edge_index",
    )

    prediction = model(
        x.to(device),
        edge_index.to(device),
    )

    prediction = (
        prediction
        .detach()
        .cpu()
        .numpy()
    )

    return prediction


# ============================================================
# 11. CREATE PREDICTION DATAFRAME
# ============================================================

def create_prediction_dataframe(
    prediction_array: np.ndarray,
    spot_ids: list[str],
    drug_names: list[str],
) -> pd.DataFrame:
    """
    Convert prediction array to dataframe.

    Parameters
    ----------
    prediction_array : np.ndarray
        Prediction matrix.
    spot_ids : list[str]
        Spot identifiers.
    drug_names : list[str]
        Drug names.

    Returns
    -------
    pd.DataFrame
        Refined IC50 dataframe.
    """
    prediction_df = pd.DataFrame(
        prediction_array,
        index=spot_ids,
        columns=drug_names,
    )

    prediction_df.index.name = "spot_id"

    return prediction_df


# ============================================================
# 12. TOP K DRUGS
# ============================================================

def create_top10_drugs_per_spot(
    prediction_df: pd.DataFrame,
    top_k: int = TOP_K_DRUGS,
) -> pd.DataFrame:
    """
    Create top-k most sensitive drugs.

    Lower LN_IC50 = higher sensitivity.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Refined IC50 dataframe.
    top_k : int
        Number of drugs per spot.

    Returns
    -------
    pd.DataFrame
        Drug ranking dataframe.
    """
    ranking_df = create_drug_ranking(
        pred_df=prediction_df,
        top_k=top_k,
    )

    if "sample_id" in ranking_df.columns:
        ranking_df = ranking_df.rename(
            columns={
                "sample_id": "spot_id",
            }
        )

    expected_columns = [
        "spot_id",
        "rank",
        "drug",
        "predicted_LN_IC50",
    ]

    ranking_df = ranking_df[
        expected_columns
    ].copy()

    ranking_df = ranking_df.sort_values(
        [
            "spot_id",
            "rank",
        ]
    )

    ranking_df.reset_index(
        drop=True,
        inplace=True,
    )

    return ranking_df


# ============================================================
# 13. TOP OVERALL DRUGS
# ============================================================

def create_top10_drugs_overall(
    prediction_df: pd.DataFrame,
    top_k: int = TOP_K_DRUGS,
) -> pd.DataFrame:
    """
    Create top-k overall drugs for the whole spatial dataset.

    Lower mean LN_IC50 means higher predicted sensitivity.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Refined IC50 dataframe with shape n_spots x n_drugs.
    top_k : int
        Number of top overall drugs.

    Returns
    -------
    pd.DataFrame
        Overall top-k drug dataframe.
    """
    rows = []

    for drug in prediction_df.columns:
        drug_values = pd.to_numeric(
            prediction_df[drug],
            errors="coerce",
        )

        rows.append(
            {
                "drug": str(drug),
                "mean_refined_LN_IC50": float(drug_values.mean()),
                "median_refined_LN_IC50": float(drug_values.median()),
                "min_refined_LN_IC50": float(drug_values.min()),
                "max_refined_LN_IC50": float(drug_values.max()),
                "std_refined_LN_IC50": float(drug_values.std()),
            }
        )

    overall_df = pd.DataFrame(rows)

    overall_df = overall_df.sort_values(
        by="mean_refined_LN_IC50",
        ascending=True,
    ).head(top_k).copy()

    overall_df.insert(
        0,
        "rank",
        range(1, len(overall_df) + 1),
    )

    overall_df = overall_df[
        [
            "rank",
            "drug",
            "mean_refined_LN_IC50",
            "median_refined_LN_IC50",
            "min_refined_LN_IC50",
            "max_refined_LN_IC50",
            "std_refined_LN_IC50",
        ]
    ].copy()

    overall_df.reset_index(
        drop=True,
        inplace=True,
    )

    return overall_df


# ============================================================
# 14. SAVE HELPERS
# ============================================================

def save_output_dataframe(
    df: pd.DataFrame,
    output_file: Path,
    log_file: Path | None = None,
    index: bool = True,
) -> None:
    """
    Save dataframe using project utility.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    output_file : Path
        Output file.
    log_file : Path | None
        Log file path.
    index : bool
        Save index or not.
    """
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        save_dataframe(
            df=df,
            output_file=output_file,
            index=index,
            overwrite=OVERWRITE_OUTPUTS,
        )

    except TypeError:

        df.to_csv(
            output_file,
            index=index,
        )

    log_print(
        f"Saved: {output_file}",
        log_file,
    )


# ============================================================
# 15. CREATE DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    dataset_name: str,
    prediction_df: pd.DataFrame,
    top10_df: pd.DataFrame,
    top10_overall_df: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """
    Create prediction summary.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    prediction_df : pd.DataFrame
        Refined IC50 dataframe.
    top10_df : pd.DataFrame
        Drug ranking dataframe per spot.
    top10_overall_df : pd.DataFrame
        Overall top-k drug dataframe.
    checkpoint : dict[str, Any]
        Loaded checkpoint.

    Returns
    -------
    dict[str, Any]
        Summary dictionary.
    """
    return {
        "dataset_name": dataset_name,
        "n_spots": int(
            prediction_df.shape[0]
        ),
        "n_drugs": int(
            prediction_df.shape[1]
        ),
        "prediction_shape": str(
            prediction_df.shape
        ),
        "top_k_drugs": int(
            TOP_K_DRUGS
        ),
        "top10_rows": int(
            top10_df.shape[0]
        ),
        "overall_top1_drug": (
            str(top10_overall_df.iloc[0]["drug"])
            if not top10_overall_df.empty
            else ""
        ),
        "overall_top1_mean_ic50": (
            float(top10_overall_df.iloc[0]["mean_refined_LN_IC50"])
            if not top10_overall_df.empty
            else np.nan
        ),
        "model_name": checkpoint.get(
            "model_name",
            "unknown",
        ),
        "gnn_type": checkpoint[
            "gnn_type"
        ],
        "input_dim": int(
            checkpoint["input_dim"]
        ),
        "hidden_dim": int(
            checkpoint["hidden_dim"]
        ),
        "output_dim": int(
            checkpoint["output_dim"]
        ),
        "num_layers": int(
            checkpoint["num_layers"]
        ),
        "dropout": float(
            checkpoint["dropout"]
        ),
        "feature_sources": ",".join(
            checkpoint["feature_sources"]
        ),
        "target_source": str(
            checkpoint["target_source"]
        ),
    }
    
    
# ============================================================
# 16. PROCESS ONE DATASET
# ============================================================

def process_one_dataset(
    dataset_name: str,
    model: nn.Module,
    checkpoint: dict[str, Any],
    paths: dict[str, Path],
    device: torch.device,
    log_file: Path | None = None,
) -> dict[str, Any] | None:
    """
    Process one spatial dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    model : nn.Module
        Loaded GNN model.
    checkpoint : dict[str, Any]
        Checkpoint dictionary.
    paths : dict[str, Path]
        Project paths.
    device : torch.device
        Torch device.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, Any] | None
        Summary dictionary.
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

        graph_file = (
            dataset_dir
            / "spatial_graph.pt"
        )

        graph = load_graph(
            graph_file=graph_file,
            device=device,
            log_file=log_file,
        )

        validate_graph_against_checkpoint(
            graph=graph,
            checkpoint=checkpoint,
        )

        prediction_array = predict_refined_ic50(
            model=model,
            graph=graph,
            device=device,
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

        prediction_df = create_prediction_dataframe(
            prediction_array=prediction_array,
            spot_ids=spot_ids,
            drug_names=drug_names,
        )

        top10_df = create_top10_drugs_per_spot(
            prediction_df=prediction_df,
            top_k=TOP_K_DRUGS,
        )

        top10_overall_df = create_top10_drugs_overall(
            prediction_df=prediction_df,
            top_k=TOP_K_DRUGS,
        )

        summary = create_dataset_summary(
            dataset_name=dataset_name,
            prediction_df=prediction_df,
            top10_df=top10_df,
            top10_overall_df=top10_overall_df,
            checkpoint=checkpoint,
        )

        summary_df = pd.DataFrame(
            [summary]
        )

        # ----------------------------------------------------
        # Save refined IC50
        # ----------------------------------------------------

        if SAVE_REFINED_IC50:

            refined_file = (
                dataset_dir
                / "refined_ic50_gnn.csv"
            )

            save_output_dataframe(
                df=prediction_df,
                output_file=refined_file,
                log_file=log_file,
                index=True,
            )

        # ----------------------------------------------------
        # Save top drugs
        # ----------------------------------------------------

        if SAVE_TOP_DRUG_RANKING:

            ranking_file = (
                dataset_dir
                / "refined_top10_drugs_per_spot.csv"
            )

            save_output_dataframe(
                df=top10_df,
                output_file=ranking_file,
                log_file=log_file,
                index=False,
            )

        # ----------------------------------------------------
        # Save overall top drugs
        # ----------------------------------------------------

        if SAVE_TOP_OVERALL_DRUGS:

            overall_file = (
                dataset_dir
                / "refined_top10_drugs_overall.csv"
            )

            save_output_dataframe(
                df=top10_overall_df,
                output_file=overall_file,
                log_file=log_file,
                index=False,
            )

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        if SAVE_SUMMARY:

            summary_file = (
                dataset_dir
                / "gnn_prediction_summary.csv"
            )

            save_output_dataframe(
                df=summary_df,
                output_file=summary_file,
                log_file=log_file,
                index=False,
            )

        log_print(
            f"Nodes: {prediction_df.shape[0]}",
            log_file,
        )

        log_print(
            f"Drugs: {prediction_df.shape[1]}",
            log_file,
        )

        log_print(
            f"Prediction shape: {prediction_df.shape}",
            log_file,
        )

        log_print(
            f"Top10 rows: {top10_df.shape[0]}",
            log_file,
        )

        log_print(
            f"Overall Top10 Drugs: {top10_overall_df.shape[0]}",
            log_file,
        )

        log_print(
            top10_overall_df.to_string(index=False),
            log_file,
        )

        log_print(
            f"[DONE] {dataset_name}",
            log_file,
        )

        log_print("", log_file)

        return summary

    except Exception as error:

        log_print(
            f"[ERROR] Failed dataset "
            f"{dataset_name}: {error}",
            log_file,
        )

        log_print("", log_file)

        return None


# ============================================================
# 17. STREAMLIT / DEMO HELPER
# ============================================================

def predict_uploaded_spatial_gnn(
    graph: Any,
    model: nn.Module,
    checkpoint: dict[str, Any],
    device: torch.device,
    top_k: int = TOP_K_DRUGS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Predict refined IC50 for uploaded graph.

    Used by Streamlit demo.

    Parameters
    ----------
    graph : Any
        Spatial graph.
    model : nn.Module
        Trained GNN model.
    checkpoint : dict[str, Any]
        Checkpoint dictionary.
    device : torch.device
        Torch device.
    top_k : int
        Number of top drugs.

    Returns
    -------
    tuple
        prediction_df,
        top10_df,
        overall_top10_df,
        summary
    """
    validate_graph_against_checkpoint(
        graph=graph,
        checkpoint=checkpoint,
    )

    prediction_array = predict_refined_ic50(
        model=model,
        graph=graph,
        device=device,
    )

    prediction_df = create_prediction_dataframe(
        prediction_array=prediction_array,
        spot_ids=list(
            get_graph_attr(
                graph,
                "spot_ids",
            )
        ),
        drug_names=list(
            get_graph_attr(
                graph,
                "drug_names",
            )
        ),
    )

    top10_df = create_top10_drugs_per_spot(
        prediction_df=prediction_df,
        top_k=top_k,
    )

    overall_top10_df = create_top10_drugs_overall(
        prediction_df=prediction_df,
        top_k=top_k,
    )

    summary = {
        "n_spots": int(
            prediction_df.shape[0]
        ),
        "n_drugs": int(
            prediction_df.shape[1]
        ),
        "prediction_shape": str(
            prediction_df.shape
        ),
        "top_k": int(top_k),
        "model_name": checkpoint.get(
            "model_name",
            "unknown",
        ),
        "gnn_type": checkpoint[
            "gnn_type"
        ],
        "overall_top1_drug": (
            str(overall_top10_df.iloc[0]["drug"])
            if not overall_top10_df.empty
            else ""
        ),
        "overall_top1_mean_ic50": (
            float(overall_top10_df.iloc[0]["mean_refined_LN_IC50"])
            if not overall_top10_df.empty
            else np.nan
        ),
    }

    return (
        prediction_df,
        top10_df,
        overall_top10_df,
        summary,
    )


# ============================================================
# 18. MAIN
# ============================================================

def main() -> None:
    """
    Main GNN prediction pipeline.
    """
    paths = setup_paths()

    log_file = (
        paths["log_file"]
        if SAVE_LOG
        else None
    )

    log_print("=" * 70, log_file)
    log_print(
        "PREDICT REFINED IC50",
        log_file,
    )
    log_print("=" * 70, log_file)

    if not HAS_TORCH_GEOMETRIC:

        log_print(
            "[ERROR] torch_geometric chưa được cài đặt.",
            log_file,
        )

        return

    device = get_device(
        DEVICE
    )

    log_print(
        f"PROJECT_ROOT: {paths['project_root']}",
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

        model, checkpoint = load_gnn_checkpoint(
            model_file=paths["gnn_model_file"],
            device=device,
            log_file=log_file,
        )

        report_rows = []

        for dataset_name in DATASETS:

            summary = process_one_dataset(
                dataset_name=dataset_name,
                model=model,
                checkpoint=checkpoint,
                paths=paths,
                device=device,
                log_file=log_file,
            )

            if summary is not None:
                report_rows.append(summary)

        if len(report_rows) == 0:

            log_print(
                "[WARNING] No dataset predicted successfully.",
                log_file,
            )

            return

        report_df = pd.DataFrame(
            report_rows
        )

        report_file = (
            paths["results_spatial_dir"]
            / "gnn_prediction_report.csv"
        )

        save_output_dataframe(
            df=report_df,
            output_file=report_file,
            log_file=log_file,
            index=False,
        )

        log_print("", log_file)

        log_print("=" * 70, log_file)
        log_print(
            "GNN PREDICTION REPORT",
            log_file,
        )
        log_print("=" * 70, log_file)

        log_print(
            report_df.to_string(
                index=False
            ),
            log_file,
        )

        log_print("", log_file)

        log_print("=" * 70, log_file)
        log_print(
            "FINISHED PREDICT REFINED IC50",
            log_file,
        )
        log_print("=" * 70, log_file)

    except Exception as error:

        log_print(
            f"[ERROR] Prediction pipeline failed: {error}",
            log_file,
        )


if __name__ == "__main__":
    main()