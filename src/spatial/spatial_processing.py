# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from datetime import datetime
from typing import Any
import sys

import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# 1. SAFE IMPORT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.prediction import (
    read_selected_genes,
    load_scaler,
    load_pca,
    preprocess_expression_to_pca,
    save_dataframe,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASETS = [
    "block_a_section_1",
    "block_a_section_2",
    "ductal_carcinoma_in_situ_invasive_carcinoma_FFPE",
    "visium_fresh_frozen_whole_transcriptome",
    "whole_transcriptome_analysis",
    "block_a_section_1_2",
    "block_a_section_2_2",
]

FILL_MISSING_GENE_VALUE = 0.0

MERGE_DUPLICATE_GENES = True
DUPLICATE_GENE_AGG_METHOD = "mean"

SAVE_SPOT_POSITIONS = True
SAVE_X_SPOT_PCA = True
SAVE_PREPROCESSING_SUMMARY = True

OVERWRITE_OUTPUTS = True
SAVE_LOG = True


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

    spatial_raw_dir = project_root / "data" / "spatial_raw"
    spatial_processed_dir = project_root / "data" / "spatial_processed"
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    results_spatial_dir = project_root / "results" / "spatial"
    logs_dir = project_root / "logs"

    spatial_processed_dir.mkdir(parents=True, exist_ok=True)
    results_spatial_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "spatial_raw_dir": spatial_raw_dir,
        "spatial_processed_dir": spatial_processed_dir,
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "results_spatial_dir": results_spatial_dir,
        "logs_dir": logs_dir,
        "log_file": logs_dir / f"spatial_preprocessing_{timestamp}.log",
        "selected_genes_file": processed_dir / "selected_genes.txt",
        "scaler_file": models_dir / "scaler.pkl",
        "pca_file": models_dir / "pca.pkl",
    }


# ============================================================
# 4. LOGGING
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
# 5. CHECK CORE INPUT FILES
# ============================================================

def check_core_input_files(
    paths: dict[str, Path],
    log_file: Path | None = None,
) -> bool:
    """
    Check required input files for spatial preprocessing.

    Required files:
    - selected_genes.txt
    - scaler.pkl
    - pca.pkl

    Parameters
    ----------
    paths : dict[str, Path]
        Project paths.
    log_file : Path | None
        Log file path.

    Returns
    -------
    bool
        True if all required files exist, otherwise False.
    """
    required_files = [
        paths["selected_genes_file"],
        paths["scaler_file"],
        paths["pca_file"],
    ]

    is_valid = True

    log_print("=" * 70, log_file)
    log_print("CHECK CORE INPUT FILES", log_file)
    log_print("=" * 70, log_file)

    for file_path in required_files:
        if file_path.exists():
            log_print(f"[OK] Found: {file_path}", log_file)
        else:
            log_print(f"[ERROR] Missing required file: {file_path}", log_file)
            is_valid = False

    log_print("", log_file)

    return is_valid


# ============================================================
# 6. FIND INPUT FILES
# ============================================================

def find_h5_file(dataset_dir: Path) -> Path:
    """
    Find the single filtered feature barcode matrix H5 file in a dataset folder.

    Parameters
    ----------
    dataset_dir : Path
        Dataset raw directory.

    Returns
    -------
    Path
        H5 file path.

    Raises
    ------
    FileNotFoundError
        If dataset directory does not exist or does not contain exactly one H5 file.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset raw directory không tồn tại: {dataset_dir}"
        )

    h5_files = list(
        dataset_dir.glob("*_filtered_feature_bc_matrix.h5")
    )

    if len(h5_files) != 1:
        raise FileNotFoundError(
            f"Yêu cầu đúng 1 file '*_filtered_feature_bc_matrix.h5' "
            f"trong {dataset_dir}, nhưng tìm thấy {len(h5_files)} file."
        )

    return h5_files[0]


def find_tissue_position_file(
    dataset_dir: Path,
    log_file: Path | None = None,
) -> Path | None:
    """
    Find tissue position file.

    Priority:
    1. spatial/tissue_positions.csv
    2. spatial/tissue_positions_list.csv

    Parameters
    ----------
    dataset_dir : Path
        Dataset raw directory.
    log_file : Path | None
        Log file path.

    Returns
    -------
    Path | None
        Tissue position file path if found, otherwise None.
    """
    candidates = [
        dataset_dir / "spatial" / "tissue_positions.csv",
        dataset_dir / "spatial" / "tissue_positions_list.csv",
    ]

    for file_path in candidates:
        if file_path.exists():
            return file_path

    log_print("[ERROR] Không tìm thấy tissue position file.", log_file)

    for file_path in candidates:
        log_print(f"Checked: {file_path}", log_file)

    return None


# ============================================================
# 7. LOAD VISIUM EXPRESSION
# ============================================================

def handle_duplicate_genes(
    X_expression: pd.DataFrame,
    log_file: Path | None = None,
) -> pd.DataFrame:
    """
    Handle duplicate gene columns.

    If MERGE_DUPLICATE_GENES=True, duplicate gene columns are merged
    by mean. Otherwise, only the first duplicated gene column is kept.

    Parameters
    ----------
    X_expression : pd.DataFrame
        Spot x gene expression matrix.
    log_file : Path | None
        Log file path.

    Returns
    -------
    pd.DataFrame
        Expression matrix after duplicate gene handling.
    """
    duplicate_gene_count = int(X_expression.columns.duplicated().sum())

    if duplicate_gene_count == 0:
        return X_expression

    if MERGE_DUPLICATE_GENES:
        if DUPLICATE_GENE_AGG_METHOD != "mean":
            raise ValueError(
                "Hiện tại chỉ hỗ trợ DUPLICATE_GENE_AGG_METHOD='mean'."
            )

        log_print(
            f"Gộp gene trùng bằng mean. Duplicate gene columns: {duplicate_gene_count}",
            log_file,
        )

        X_expression = X_expression.T.groupby(level=0).mean().T

    else:
        log_print(
            "MERGE_DUPLICATE_GENES=False. Giữ cột gene đầu tiên khi bị trùng.",
            log_file,
        )

        X_expression = X_expression.loc[
            :,
            ~X_expression.columns.duplicated(keep="first"),
        ].copy()

    return X_expression


def load_visium_expression(
    h5_file: Path,
    log_file: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load Visium expression matrix from 10x Genomics h5 file.

    This function does not use adata.var_names_make_unique() as the
    main solution for duplicated genes, because it can rename genes
    as GENE-1, GENE-2 and reduce overlap with selected_genes.

    Parameters
    ----------
    h5_file : Path
        Path to filtered_feature_bc_matrix.h5.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Expression matrix and expression statistics.
    """
    adata = sc.read_10x_h5(str(h5_file))

    if hasattr(adata.X, "toarray"):
        expression_values = adata.X.toarray()
    else:
        expression_values = np.asarray(adata.X)

    X_expression = pd.DataFrame(
        expression_values,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    X_expression.index = X_expression.index.astype(str).str.strip()
    X_expression.columns = X_expression.columns.astype(str).str.strip()
    X_expression.index.name = "spot_id"

    raw_shape = X_expression.shape
    duplicate_gene_count = int(X_expression.columns.duplicated().sum())

    log_print(
        f"Raw expression shape before duplicate handling: {raw_shape}",
        log_file,
    )
    log_print(f"Duplicate genes found: {duplicate_gene_count}", log_file)

    X_expression = handle_duplicate_genes(
        X_expression=X_expression,
        log_file=log_file,
    )

    X_expression = X_expression.apply(pd.to_numeric, errors="coerce")
    X_expression = X_expression.replace([np.inf, -np.inf], np.nan)
    X_expression = X_expression.fillna(0.0)

    after_shape = X_expression.shape

    log_print(
        f"Expression shape after duplicate handling: {after_shape}",
        log_file,
    )
    log_print(f"Number of expression spots: {after_shape[0]}", log_file)
    log_print(f"Number of expression genes: {after_shape[1]}", log_file)

    expression_stats = {
        "raw_expression_shape": str(raw_shape),
        "duplicate_gene_count": duplicate_gene_count,
        "expression_shape_after_duplicate_handling": str(after_shape),
        "n_spots_expression": int(after_shape[0]),
        "n_genes_expression": int(after_shape[1]),
        "n_raw_genes_before_duplicate_handling": int(raw_shape[1]),
        "n_raw_genes_after_duplicate_handling": int(after_shape[1]),
    }

    return X_expression, expression_stats


# ============================================================
# 8. LOAD TISSUE POSITIONS
# ============================================================

def load_tissue_positions(
    position_file: Path,
    log_file: Path | None = None,
) -> pd.DataFrame:
    """
    Load Visium tissue positions.

    Supports:
    - tissue_positions.csv
    - tissue_positions_list.csv

    Parameters
    ----------
    position_file : Path
        Tissue position file.
    log_file : Path | None
        Log file path.

    Returns
    -------
    pd.DataFrame
        Position dataframe with:
        spot_id, x, y, in_tissue, array_row, array_col.
    """
    if position_file.name == "tissue_positions_list.csv":
        columns = [
            "spot_id",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]

        positions_df = pd.read_csv(
            position_file,
            header=None,
            names=columns,
        )

    else:
        positions_df = pd.read_csv(position_file)
        positions_df = positions_df.rename(
            columns={
                "barcode": "spot_id",
            }
        )

    required_cols = [
        "spot_id",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in positions_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Position file thiếu cột: {missing_cols}")

    positions_df["spot_id"] = positions_df["spot_id"].astype(str).str.strip()

    positions_df["x"] = pd.to_numeric(
        positions_df["pxl_col_in_fullres"],
        errors="coerce",
    )

    positions_df["y"] = pd.to_numeric(
        positions_df["pxl_row_in_fullres"],
        errors="coerce",
    )

    for col in ["in_tissue", "array_row", "array_col"]:
        positions_df[col] = pd.to_numeric(positions_df[col], errors="coerce")

    positions_df = positions_df[
        [
            "spot_id",
            "x",
            "y",
            "in_tissue",
            "array_row",
            "array_col",
        ]
    ].copy()

    positions_df = positions_df.dropna(
        subset=[
            "spot_id",
            "x",
            "y",
        ]
    ).copy()

    duplicated_spots = int(positions_df["spot_id"].duplicated().sum())

    if duplicated_spots > 0:
        log_print(
            f"[WARNING] positions có {duplicated_spots} spot_id bị trùng. "
            f"Giữ dòng đầu tiên.",
            log_file,
        )

        positions_df = positions_df.drop_duplicates(
            subset=["spot_id"],
            keep="first",
        ).copy()

    log_print(f"Position shape: {positions_df.shape}", log_file)

    return positions_df


# ============================================================
# 9. SYNC EXPRESSION AND POSITIONS
# ============================================================

def sync_expression_and_positions(
    X_expression: pd.DataFrame,
    positions_df: pd.DataFrame,
    log_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Synchronize expression matrix and spot positions.

    Only spots appearing in both expression and positions are kept.
    The final order follows X_expression.index.

    Parameters
    ----------
    X_expression : pd.DataFrame
        Spot x gene expression matrix.
    positions_df : pd.DataFrame
        Spot positions dataframe.
    log_file : Path | None
        Log file path.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]
        Synced expression, synced positions, and sync statistics.
    """
    expression_spots = list(X_expression.index)
    position_spot_set = set(positions_df["spot_id"].astype(str))

    common_spots = [
        spot_id for spot_id in expression_spots
        if spot_id in position_spot_set
    ]

    sync_stats = {
        "spots_expression": int(len(expression_spots)),
        "spots_positions": int(len(position_spot_set)),
        "spots_intersection": int(len(common_spots)),
    }

    log_print(f"spots_expression: {sync_stats['spots_expression']}", log_file)
    log_print(f"spots_positions: {sync_stats['spots_positions']}", log_file)
    log_print(f"spots_intersection: {sync_stats['spots_intersection']}", log_file)

    if len(common_spots) == 0:
        raise ValueError("Không có spot chung giữa expression và positions.")

    X_expression_synced = X_expression.loc[common_spots].copy()

    positions_synced = (
        positions_df
        .set_index("spot_id")
        .loc[common_spots]
        .reset_index()
    )

    return X_expression_synced, positions_synced, sync_stats


# ============================================================
# 10. LOAD PREPROCESSING OBJECTS
# ============================================================

def load_preprocessing_objects(
    paths: dict[str, Path],
    log_file: Path | None = None,
) -> dict[str, Any]:
    """
    Load preprocessing objects from training stage.

    Objects:
    - selected_genes.txt
    - scaler.pkl
    - pca.pkl

    Parameters
    ----------
    paths : dict[str, Path]
        Project paths.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, Any]
        Loaded preprocessing objects.
    """
    selected_genes = read_selected_genes(
        paths["selected_genes_file"]
    )

    scaler = load_scaler(
        paths["scaler_file"]
    )

    pca = load_pca(
        paths["pca_file"]
    )

    log_print("=" * 70, log_file)
    log_print("LOADED PREPROCESSING OBJECTS", log_file)
    log_print("=" * 70, log_file)

    log_print(
        f"Selected genes: {len(selected_genes)}",
        log_file,
    )

    try:
        n_features = scaler.n_features_in_

        log_print(
            f"Scaler input features: {n_features}",
            log_file,
        )

    except Exception:
        pass

    try:
        n_components = pca.n_components_

        log_print(
            f"PCA components: {n_components}",
            log_file,
        )

    except Exception:
        pass

    log_print("", log_file)

    return {
        "selected_genes": selected_genes,
        "scaler": scaler,
        "pca": pca,
    }


# ============================================================
# 11. SAVE OUTPUT DATAFRAME
# ============================================================

def save_output_dataframe(
    df: pd.DataFrame,
    output_file: Path,
    log_file: Path | None = None,
    index: bool = True,
) -> bool:
    """
    Save dataframe using project utility.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    output_file : Path
        Output path.
    log_file : Path | None
        Log file path.
    index : bool
        Whether to save index.

    Returns
    -------
    bool
        True if saved.
    """
    saved = save_dataframe(
        df=df,
        output_file=output_file,
        index=index,
        overwrite=OVERWRITE_OUTPUTS,
    )

    if saved:
        log_print(
            f"Saved: {output_file}",
            log_file,
        )

    else:
        log_print(
            f"[WARNING] File exists and OVERWRITE_OUTPUTS=False: "
            f"{output_file}",
            log_file,
        )

    return saved


# ============================================================
# 12. CREATE SUMMARY
# ============================================================

def create_dataset_summary(
    dataset_name: str,
    expression_stats: dict[str, Any],
    sync_stats: dict[str, int],
    feature_stats: dict[str, Any],
    X_spot_pca: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create preprocessing summary.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    expression_stats : dict[str, Any]
        Expression statistics.
    sync_stats : dict[str, int]
        Spot synchronization statistics.
    feature_stats : dict[str, Any]
        Feature preprocessing statistics.
    X_spot_pca : pd.DataFrame
        PCA feature dataframe.

    Returns
    -------
    dict[str, Any]
        Summary dictionary.
    """
    summary = {
        "dataset_name": dataset_name,
        "n_spots": int(X_spot_pca.shape[0]),
        "n_pca_features": int(X_spot_pca.shape[1]),
        "n_raw_genes_before_duplicate_handling": int(
            expression_stats[
                "n_raw_genes_before_duplicate_handling"
            ]
        ),
        "duplicate_gene_count": int(
            expression_stats[
                "duplicate_gene_count"
            ]
        ),
        "n_raw_genes_after_duplicate_handling": int(
            expression_stats[
                "n_raw_genes_after_duplicate_handling"
            ]
        ),
        "n_selected_genes": int(
            feature_stats["n_selected_genes"]
        ),
        "n_matched_genes": int(
            feature_stats["n_matched_genes"]
        ),
        "n_missing_genes": int(
            feature_stats["n_missing_genes"]
        ),
        "coverage_percent": round(
            float(
                feature_stats["coverage_percent"]
            ),
            4,
        ),
        "spots_expression": int(
            sync_stats["spots_expression"]
        ),
        "spots_positions": int(
            sync_stats["spots_positions"]
        ),
        "spots_intersection": int(
            sync_stats["spots_intersection"]
        ),
        "expression_shape": str(
            (
                sync_stats["spots_intersection"],
                expression_stats[
                    "n_raw_genes_after_duplicate_handling"
                ],
            )
        ),
        "pca_shape": str(
            X_spot_pca.shape
        ),
    }

    return summary


# ============================================================
# 13. PROCESS ONE DATASET
# ============================================================

def process_one_dataset(
    dataset_name: str,
    paths: dict[str, Path],
    preprocessing_objects: dict[str, Any],
    log_file: Path | None = None,
) -> dict[str, Any] | None:
    """
    Process one spatial dataset.

    Pipeline:
    - Find H5 file
    - Find tissue positions
    - Load expression
    - Load positions
    - Sync spots
    - Apply selected genes
    - Scale
    - PCA transform
    - Save outputs
    - Create summary

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    paths : dict[str, Path]
        Project paths.
    preprocessing_objects : dict[str, Any]
        Loaded preprocessing objects.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, Any] | None
        Summary dictionary.
    """
    log_print("=" * 70, log_file)
    log_print(f"DATASET: {dataset_name}", log_file)
    log_print("=" * 70, log_file)

    try:
        dataset_raw_dir = (
            paths["spatial_raw_dir"]
            / dataset_name
        )

        dataset_output_dir = (
            paths["spatial_processed_dir"]
            / dataset_name
        )

        dataset_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        h5_file = find_h5_file(
            dataset_dir=dataset_raw_dir,
        )

        tissue_position_file = find_tissue_position_file(
            dataset_dir=dataset_raw_dir,
            log_file=log_file,
        )

        if tissue_position_file is None:
            log_print(
                f"[SKIP] Missing tissue position file for {dataset_name}",
                log_file,
            )
            return None

        log_print(
            f"Dataset: {dataset_name}",
            log_file,
        )

        log_print(
            f"Found H5: {h5_file}",
            log_file,
        )

        log_print(
            f"Found tissue positions: {tissue_position_file}",
            log_file,
        )

        X_expression, expression_stats = load_visium_expression(
            h5_file=h5_file,
            log_file=log_file,
        )

        positions_df = load_tissue_positions(
            position_file=tissue_position_file,
            log_file=log_file,
        )

        (
            X_expression,
            positions_df,
            sync_stats,
        ) = sync_expression_and_positions(
            X_expression=X_expression,
            positions_df=positions_df,
            log_file=log_file,
        )

        X_spot_pca, feature_stats = preprocess_expression_to_pca(
            X_expression=X_expression,
            selected_genes=preprocessing_objects["selected_genes"],
            scaler=preprocessing_objects["scaler"],
            pca=preprocessing_objects["pca"],
            fill_missing_gene_value=FILL_MISSING_GENE_VALUE,
        )

        X_spot_pca.index.name = "spot_id"

        log_print(
            f"Matched genes: "
            f"{feature_stats['n_matched_genes']} / "
            f"{feature_stats['n_selected_genes']}",
            log_file,
        )

        log_print(
            f"Coverage: "
            f"{float(feature_stats['coverage_percent']):.2f}%",
            log_file,
        )

        log_print(
            f"PCA shape: {X_spot_pca.shape}",
            log_file,
        )
        if SAVE_SPOT_POSITIONS:
            save_output_dataframe(
                df=positions_df,
                output_file=dataset_output_dir / "spot_positions.csv",
                log_file=log_file,
                index=False,
            )

        if SAVE_X_SPOT_PCA:
            save_output_dataframe(
                df=X_spot_pca,
                output_file=dataset_output_dir / "X_spot_pca.csv",
                log_file=log_file,
                index=True,
            )

        summary = create_dataset_summary(
            dataset_name=dataset_name,
            expression_stats=expression_stats,
            sync_stats=sync_stats,
            feature_stats=feature_stats,
            X_spot_pca=X_spot_pca,
        )

        summary_df = pd.DataFrame([summary])

        if SAVE_PREPROCESSING_SUMMARY:
            save_output_dataframe(
                df=summary_df,
                output_file=dataset_output_dir / "spatial_preprocessing_summary.csv",
                log_file=log_file,
                index=False,
            )

        log_print("", log_file)
        log_print("OUTPUT FILES", log_file)
        log_print("-" * 70, log_file)

        if SAVE_SPOT_POSITIONS:
            log_print(
                dataset_output_dir / "spot_positions.csv",
                log_file,
            )

        if SAVE_X_SPOT_PCA:
            log_print(
                dataset_output_dir / "X_spot_pca.csv",
                log_file,
            )

        if SAVE_PREPROCESSING_SUMMARY:
            log_print(
                dataset_output_dir / "spatial_preprocessing_summary.csv",
                log_file,
            )

        log_print("", log_file)
        log_print(
            f"[DONE] Finished dataset: {dataset_name}",
            log_file,
        )

        log_print("", log_file)

        return summary

    except Exception as error:

        log_print(
            f"[ERROR] Failed dataset {dataset_name}: {error}",
            log_file,
        )

        log_print("", log_file)

        return None


# ============================================================
# 14. STREAMLIT / DEMO HELPER
# ============================================================

def preprocess_uploaded_spatial_dataset(
    h5_file: Path,
    tissue_position_file: Path,
    selected_genes: list[str],
    scaler: Any,
    pca: Any,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Helper function for Streamlit demo.

    Preprocess uploaded spatial dataset and return:
    - X_spot_pca
    - positions_df
    - summary

    Parameters
    ----------
    h5_file : Path
        Uploaded H5 file.
    tissue_position_file : Path
        Uploaded tissue position file.
    selected_genes : list[str]
        Selected training genes.
    scaler : Any
        Trained scaler.
    pca : Any
        Trained PCA.

    Returns
    -------
    tuple
        X_spot_pca,
        positions_df,
        summary
    """
    X_expression, expression_stats = load_visium_expression(
        h5_file=h5_file,
        log_file=None,
    )

    positions_df = load_tissue_positions(
        position_file=tissue_position_file,
        log_file=None,
    )

    (
        X_expression,
        positions_df,
        sync_stats,
    ) = sync_expression_and_positions(
        X_expression=X_expression,
        positions_df=positions_df,
        log_file=None,
    )

    X_spot_pca, feature_stats = preprocess_expression_to_pca(
        X_expression=X_expression,
        selected_genes=selected_genes,
        scaler=scaler,
        pca=pca,
        fill_missing_gene_value=FILL_MISSING_GENE_VALUE,
    )

    X_spot_pca.index.name = "spot_id"

    summary = {
        "n_spots": int(X_spot_pca.shape[0]),
        "n_pca_features": int(X_spot_pca.shape[1]),
        "n_selected_genes": int(
            feature_stats["n_selected_genes"]
        ),
        "n_matched_genes": int(
            feature_stats["n_matched_genes"]
        ),
        "coverage_percent": float(
            feature_stats["coverage_percent"]
        ),
        "spots_intersection": int(
            sync_stats["spots_intersection"]
        ),
    }

    return (
        X_spot_pca,
        positions_df,
        summary,
    )


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Main preprocessing pipeline.
    """
    paths = setup_paths()

    log_file = (
        paths["log_file"]
        if SAVE_LOG
        else None
    )

    log_print("=" * 70, log_file)
    log_print("SPATIAL PREPROCESSING", log_file)
    log_print("=" * 70, log_file)

    log_print(
        f"PROJECT_ROOT: {paths['project_root']}",
        log_file,
    )

    log_print(
        f"SPATIAL_RAW_DIR: {paths['spatial_raw_dir']}",
        log_file,
    )

    log_print(
        f"SPATIAL_PROCESSED_DIR: {paths['spatial_processed_dir']}",
        log_file,
    )

    log_print(
        f"RESULTS_SPATIAL_DIR: {paths['results_spatial_dir']}",
        log_file,
    )

    log_print(
        f"DATASETS: {DATASETS}",
        log_file,
    )

    log_print(
        f"FILL_MISSING_GENE_VALUE: "
        f"{FILL_MISSING_GENE_VALUE}",
        log_file,
    )

    log_print(
        f"MERGE_DUPLICATE_GENES: "
        f"{MERGE_DUPLICATE_GENES}",
        log_file,
    )

    log_print(
        f"DUPLICATE_GENE_AGG_METHOD: "
        f"{DUPLICATE_GENE_AGG_METHOD}",
        log_file,
    )

    log_print(
        f"LOG_FILE: {log_file}",
        log_file,
    )

    log_print("", log_file)

    if not check_core_input_files(
        paths=paths,
        log_file=log_file,
    ):
        log_print(
            "[ERROR] Missing required preprocessing files.",
            log_file,
        )
        return

    preprocessing_objects = load_preprocessing_objects(
        paths=paths,
        log_file=log_file,
    )

    report_rows = []

    for dataset_name in DATASETS:

        summary = process_one_dataset(
            dataset_name=dataset_name,
            paths=paths,
            preprocessing_objects=preprocessing_objects,
            log_file=log_file,
        )

        if summary is not None:
            report_rows.append(summary)

    if len(report_rows) == 0:

        log_print(
            "[WARNING] No dataset processed successfully.",
            log_file,
        )

        return

    report_df = pd.DataFrame(
        report_rows
    )

    report_file = (
        paths["results_spatial_dir"]
        / "spatial_preprocessing_report.csv"
    )

    save_output_dataframe(
        df=report_df,
        output_file=report_file,
        log_file=log_file,
        index=False,
    )

    log_print("=" * 70, log_file)
    log_print(
        "SPATIAL PREPROCESSING REPORT",
        log_file,
    )
    log_print("=" * 70, log_file)

    log_print(
        report_df.to_string(index=False),
        log_file,
    )

    log_print("", log_file)

    log_print("=" * 70, log_file)
    log_print(
        "FINISHED SPATIAL PREPROCESSING",
        log_file,
    )
    log_print("=" * 70, log_file)


if __name__ == "__main__":
    main()