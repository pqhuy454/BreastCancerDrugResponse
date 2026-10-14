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


# ============================================================
# 1. SAFE IMPORT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.prediction import (
    read_selected_drugs,
    load_sklearn_model,
    load_dnn_model,
    get_torch_device,
    read_matrix_csv,
    predict_with_sklearn_model,
    predict_with_dnn_model,
    create_ensemble_prediction,
    create_drug_ranking,
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

PREDICT_MODELS = [
    "ridge",
    "xgboost",
    "dnn",
]

BASELINE_METHOD = "mean"

TOP_K_DRUGS = 10

DEVICE = "auto"

SAVE_INDIVIDUAL_MODEL_PREDICTIONS = True
SAVE_MEAN_BASELINE = True
SAVE_TOP_DRUG_RANKING = True
SAVE_SUMMARY = True

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
    Setup all paths for spatial baseline prediction.

    Returns
    -------
    dict[str, Path]
        Dictionary of project paths.
    """
    project_root = get_project_root()

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
        "spatial_processed_dir": spatial_processed_dir,
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "results_spatial_dir": results_spatial_dir,
        "logs_dir": logs_dir,
        "log_file": logs_dir / f"predict_spatial_baseline_{timestamp}.log",
        "selected_drugs_file": processed_dir / "selected_drugs.txt",
        "ridge_model_file": models_dir / "ridge_model.pkl",
        "xgboost_model_file": models_dir / "xgboost_model.pkl",
        "dnn_model_file": models_dir / "dnn_model.pkl",
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
# 5. CHECK INPUT FILES
# ============================================================

def check_core_input_files(
    paths: dict[str, Path],
    log_file: Path | None = None,
) -> bool:
    """
    Check selected_drugs.txt and available baseline model files.

    Parameters
    ----------
    paths : dict[str, Path]
        Project paths.
    log_file : Path | None
        Log file path.

    Returns
    -------
    bool
        True if selected_drugs.txt and at least one model exist.
    """
    log_print("=" * 70, log_file)
    log_print("CHECK CORE INPUT FILES", log_file)
    log_print("=" * 70, log_file)

    is_valid = True

    if paths["selected_drugs_file"].exists():
        log_print(f"[OK] Found selected drugs: {paths['selected_drugs_file']}", log_file)
    else:
        log_print(f"[ERROR] Missing selected drugs: {paths['selected_drugs_file']}", log_file)
        is_valid = False

    model_file_map = {
        "ridge": paths["ridge_model_file"],
        "xgboost": paths["xgboost_model_file"],
        "dnn": paths["dnn_model_file"],
    }

    existing_model_count = 0

    for model_name in PREDICT_MODELS:
        model_file = model_file_map.get(model_name)

        if model_file is None:
            log_print(f"[WARNING] Unknown model name: {model_name}", log_file)
            continue

        if model_file.exists():
            existing_model_count += 1
            log_print(f"[OK] Found model {model_name}: {model_file}", log_file)
        else:
            log_print(f"[WARNING] Missing model {model_name}: {model_file}", log_file)

    if existing_model_count == 0:
        log_print("[ERROR] Không có model baseline nào tồn tại.", log_file)
        is_valid = False

    log_print("", log_file)

    return is_valid


def check_dataset_input_file(
    dataset_name: str,
    paths: dict[str, Path],
    log_file: Path | None = None,
) -> Path | None:
    """
    Check X_spot_pca.csv for a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    paths : dict[str, Path]
        Project paths.
    log_file : Path | None
        Log file path.

    Returns
    -------
    Path | None
        X_spot_pca.csv path if exists.
    """
    x_spot_pca_file = (
        paths["spatial_processed_dir"]
        / dataset_name
        / "X_spot_pca.csv"
    )

    if not x_spot_pca_file.exists():
        log_print(
            f"[ERROR] Missing X_spot_pca.csv for {dataset_name}: {x_spot_pca_file}",
            log_file,
        )
        return None

    return x_spot_pca_file


# ============================================================
# 6. LOAD X_SPOT_PCA
# ============================================================

def load_x_spot_pca(
    file_path: Path,
    log_file: Path | None = None,
) -> pd.DataFrame:
    """
    Load X_spot_pca.csv.

    Parameters
    ----------
    file_path : Path
        Path to X_spot_pca.csv.
    log_file : Path | None
        Log file path.

    Returns
    -------
    pd.DataFrame
        PCA feature dataframe.
    """
    X_spot_pca = read_matrix_csv(
        file_path=file_path,
        index_col=0,
    )

    X_spot_pca.index = X_spot_pca.index.astype(str).str.strip()
    X_spot_pca.columns = X_spot_pca.columns.astype(str).str.strip()
    X_spot_pca.index.name = "spot_id"

    X_spot_pca = X_spot_pca.apply(pd.to_numeric, errors="coerce")
    X_spot_pca = X_spot_pca.replace([np.inf, -np.inf], np.nan)
    X_spot_pca = X_spot_pca.fillna(0.0)

    log_print(f"X_spot_pca shape: {X_spot_pca.shape}", log_file)

    return X_spot_pca


# ============================================================
# 7. LOAD SELECTED DRUGS AND MODELS
# ============================================================

def load_selected_drugs_file(
    paths: dict[str, Path],
    log_file: Path | None = None,
) -> list[str]:
    """
    Load selected drugs list.

    Parameters
    ----------
    paths : dict[str, Path]
        Project paths.
    log_file : Path | None
        Log file path.

    Returns
    -------
    list[str]
        Selected drugs.
    """
    selected_drugs = read_selected_drugs(
        paths["selected_drugs_file"]
    )

    log_print(
        f"Selected drugs: {len(selected_drugs)}",
        log_file,
    )

    return selected_drugs


def load_prediction_objects(
    paths: dict[str, Path],
    selected_drugs: list[str],
    input_dim: int,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """
    Load available baseline prediction models.

    Parameters
    ----------
    paths : dict[str, Path]
        Project paths.
    selected_drugs : list[str]
        Selected drugs.
    input_dim : int
        PCA input dimension.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, Any]
        Loaded model dictionary.
    """
    models: dict[str, Any] = {}

    if "ridge" in PREDICT_MODELS:
        if paths["ridge_model_file"].exists():
            try:
                models["ridge"] = load_sklearn_model(
                    paths["ridge_model_file"]
                )
                log_print(
                    f"Loaded ridge_model.pkl: {paths['ridge_model_file']}",
                    log_file,
                )
            except Exception as error:
                log_print(
                    f"[WARNING] Cannot load ridge model: {error}",
                    log_file,
                )
        else:
            log_print(
                f"[WARNING] Missing ridge model: {paths['ridge_model_file']}",
                log_file,
            )

    if "xgboost" in PREDICT_MODELS:
        if paths["xgboost_model_file"].exists():
            try:
                models["xgboost"] = load_sklearn_model(
                    paths["xgboost_model_file"]
                )
                log_print(
                    f"Loaded xgboost_model.pkl: {paths['xgboost_model_file']}",
                    log_file,
                )
            except Exception as error:
                log_print(
                    f"[WARNING] Cannot load xgboost model: {error}",
                    log_file,
                )
        else:
            log_print(
                f"[WARNING] Missing xgboost model: {paths['xgboost_model_file']}",
                log_file,
            )

    if "dnn" in PREDICT_MODELS:
        if paths["dnn_model_file"].exists():
            try:
                device = get_torch_device(DEVICE)

                models["dnn"] = load_dnn_model(
                    model_file=paths["dnn_model_file"],
                    input_dim=input_dim,
                    output_dim=len(selected_drugs),
                    device=device,
                )

                log_print(
                    f"Loaded dnn_model.pkl: {paths['dnn_model_file']}",
                    log_file,
                )

            except Exception as error:
                log_print(
                    f"[WARNING] Cannot load dnn model: {error}",
                    log_file,
                )
        else:
            log_print(
                f"[WARNING] Missing dnn model: {paths['dnn_model_file']}",
                log_file,
            )

    if len(models) == 0:
        raise ValueError("Không load được model baseline nào.")

    return models


# ============================================================
# 8. PREDICT BASELINE MODELS
# ============================================================

def predict_all_baseline_models(
    X_spot_pca: pd.DataFrame,
    selected_drugs: list[str],
    models: dict[str, Any],
    device: torch.device,
    log_file: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Predict IC50 for spatial spots using all available baseline models.

    Parameters
    ----------
    X_spot_pca : pd.DataFrame
        PCA feature dataframe.
    selected_drugs : list[str]
        Selected drug names.
    models : dict[str, Any]
        Loaded baseline models.
    device : torch.device
        Torch device.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of prediction dataframes.
    """
    prediction_dict: dict[str, pd.DataFrame] = {}

    if "ridge" in models:
        try:
            pred_df = predict_with_sklearn_model(
                model=models["ridge"],
                X=X_spot_pca,
                selected_drugs=selected_drugs,
            )

            pred_df.index.name = "spot_id"
            prediction_dict["ridge"] = pred_df

            log_print(
                f"Ridge prediction shape: {pred_df.shape}",
                log_file,
            )

        except Exception as error:
            log_print(
                f"[WARNING] Ridge prediction failed: {error}",
                log_file,
            )

    if "xgboost" in models:
        try:
            pred_df = predict_with_sklearn_model(
                model=models["xgboost"],
                X=X_spot_pca,
                selected_drugs=selected_drugs,
            )

            pred_df.index.name = "spot_id"
            prediction_dict["xgboost"] = pred_df

            log_print(
                f"XGBoost prediction shape: {pred_df.shape}",
                log_file,
            )

        except Exception as error:
            log_print(
                f"[WARNING] XGBoost prediction failed: {error}",
                log_file,
            )

    if "dnn" in models:
        try:
            pred_df = predict_with_dnn_model(
                model=models["dnn"],
                X=X_spot_pca,
                selected_drugs=selected_drugs,
                device=device,
            )

            pred_df.index.name = "spot_id"
            prediction_dict["dnn"] = pred_df

            log_print(
                f"DNN prediction shape: {pred_df.shape}",
                log_file,
            )

        except Exception as error:
            log_print(
                f"[WARNING] DNN prediction failed: {error}",
                log_file,
            )

    if len(prediction_dict) == 0:
        raise ValueError("Không có model baseline nào dự đoán thành công.")

    return prediction_dict


# ============================================================
# 9. MEAN BASELINE
# ============================================================

def create_mean_baseline_prediction(
    prediction_dict: dict[str, pd.DataFrame],
    log_file: Path | None = None,
) -> pd.DataFrame:
    """
    Create mean baseline prediction from successful models.

    This is not voting and does not use weights.

    Parameters
    ----------
    prediction_dict : dict[str, pd.DataFrame]
        Dictionary of prediction dataframes.
    log_file : Path | None
        Log file path.

    Returns
    -------
    pd.DataFrame
        Mean baseline prediction dataframe.
    """
    valid_predictions = {
        model_name: pred_df
        for model_name, pred_df in prediction_dict.items()
        if pred_df is not None
        and isinstance(pred_df, pd.DataFrame)
        and not pred_df.empty
    }

    if len(valid_predictions) == 0:
        raise ValueError("Không có prediction hợp lệ để tạo mean baseline.")

    if len(valid_predictions) == 1:
        model_name = list(valid_predictions.keys())[0]
        log_print(
            f"[WARNING] Chỉ có 1 model thành công. Mean baseline = {model_name}.",
            log_file,
        )
        mean_df = list(valid_predictions.values())[0].copy()
        mean_df.index.name = "spot_id"
        return mean_df

    model_names = list(valid_predictions.keys())

    common_index = valid_predictions[model_names[0]].index
    common_columns = pd.Index(valid_predictions[model_names[0]].columns)

    for model_name in model_names[1:]:
        common_index = common_index.intersection(
            valid_predictions[model_name].index
        )

        common_columns = common_columns.intersection(
            pd.Index(valid_predictions[model_name].columns)
        )

    if len(common_index) == 0:
        raise ValueError("Không có spot chung giữa các model prediction.")

    if len(common_columns) == 0:
        raise ValueError("Không có drug chung giữa các model prediction.")

    stacked_arrays = []

    for model_name in model_names:
        aligned_df = valid_predictions[model_name].loc[
            common_index,
            common_columns,
        ]

        stacked_arrays.append(
            aligned_df.to_numpy(dtype=float)
        )

        log_print(
            f"Included in mean baseline: {model_name}",
            log_file,
        )

    stacked = np.stack(
        stacked_arrays,
        axis=0,
    )

    mean_values = np.mean(
        stacked,
        axis=0,
    )

    mean_df = pd.DataFrame(
        mean_values,
        index=common_index,
        columns=common_columns,
    )

    mean_df.index.name = "spot_id"

    log_print(
        f"Mean baseline shape: {mean_df.shape}",
        log_file,
    )

    return mean_df


# ============================================================
# 10. TOP 10 DRUGS
# ============================================================

def create_top10_drugs_per_spot(
    baseline_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Create top-k most sensitive drugs for each spot.

    Lower predicted LN_IC50 means higher sensitivity.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline IC50 prediction dataframe.
    top_k : int
        Number of drugs per spot.

    Returns
    -------
    pd.DataFrame
        Top-k drug ranking dataframe.
    """
    ranking_df = create_drug_ranking(
        pred_df=baseline_df,
        top_k=top_k,
    )

    ranking_df = ranking_df.rename(
        columns={
            "sample_id": "spot_id",
        }
    )

    ranking_df = ranking_df[
        [
            "spot_id",
            "rank",
            "drug",
            "predicted_LN_IC50",
        ]
    ].copy()

    return ranking_df


# ============================================================
# 11. SAVE OUTPUT
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
        Output file path.
    log_file : Path | None
        Log file path.
    index : bool
        Whether to save index.

    Returns
    -------
    bool
        True if saved, otherwise False.
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
            f"[WARNING] File exists and OVERWRITE_OUTPUTS=False: {output_file}",
            log_file,
        )

    return saved


# ============================================================
# 12. CREATE SUMMARY
# ============================================================

def create_dataset_summary(
    dataset_name: str,
    X_spot_pca: pd.DataFrame,
    selected_drugs: list[str],
    prediction_dict: dict[str, pd.DataFrame],
    mean_baseline_df: pd.DataFrame,
    top10_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create summary for one dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    X_spot_pca : pd.DataFrame
        PCA feature dataframe.
    selected_drugs : list[str]
        Selected drug names.
    prediction_dict : dict[str, pd.DataFrame]
        Model prediction dictionary.
    mean_baseline_df : pd.DataFrame
        Mean baseline dataframe.
    top10_df : pd.DataFrame
        Top-k ranking dataframe.

    Returns
    -------
    dict[str, Any]
        Summary dictionary.
    """
    models_success = list(prediction_dict.keys())

    summary = {
        "dataset_name": dataset_name,
        "n_spots": int(X_spot_pca.shape[0]),
        "n_pca_features": int(X_spot_pca.shape[1]),
        "n_selected_drugs": int(len(selected_drugs)),
        "models_requested": ",".join(PREDICT_MODELS),
        "models_success": ",".join(models_success),
        "baseline_method": BASELINE_METHOD,
        "baseline_prediction_shape": str(mean_baseline_df.shape),
        "top_k_drugs": int(TOP_K_DRUGS),
        "top10_rows": int(top10_df.shape[0]),
    }

    return summary


# ============================================================
# 13. PROCESS ONE DATASET
# ============================================================

def process_one_dataset(
    dataset_name: str,
    paths: dict[str, Path],
    selected_drugs: list[str],
    models: dict[str, Any],
    device: torch.device,
    log_file: Path | None = None,
) -> dict[str, Any] | None:
    """
    Process one spatial dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    paths : dict[str, Path]
        Project paths.
    selected_drugs : list[str]
        Selected drugs.
    models : dict[str, Any]
        Loaded models.
    device : torch.device
        Torch device.
    log_file : Path | None
        Log file path.

    Returns
    -------
    dict[str, Any] | None
        Dataset summary.
    """
    log_print("=" * 70, log_file)
    log_print(f"DATASET: {dataset_name}", log_file)
    log_print("=" * 70, log_file)

    try:
        x_spot_pca_file = check_dataset_input_file(
            dataset_name=dataset_name,
            paths=paths,
            log_file=log_file,
        )

        if x_spot_pca_file is None:
            return None

        X_spot_pca = load_x_spot_pca(
            file_path=x_spot_pca_file,
            log_file=log_file,
        )

        prediction_dict = predict_all_baseline_models(
            X_spot_pca=X_spot_pca,
            selected_drugs=selected_drugs,
            models=models,
            device=device,
            log_file=log_file,
        )

        mean_baseline_df = create_mean_baseline_prediction(
            prediction_dict=prediction_dict,
            log_file=log_file,
        )

        top10_df = create_top10_drugs_per_spot(
            baseline_df=mean_baseline_df,
            top_k=TOP_K_DRUGS,
        )

        dataset_output_dir = (
            paths["spatial_processed_dir"]
            / dataset_name
        )

        dataset_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Individual model predictions
        # ----------------------------------------------------

        if SAVE_INDIVIDUAL_MODEL_PREDICTIONS:

            for model_name, pred_df in prediction_dict.items():

                output_file = (
                    dataset_output_dir
                    / f"predicted_ic50_{model_name}.csv"
                )

                save_output_dataframe(
                    df=pred_df,
                    output_file=output_file,
                    log_file=log_file,
                    index=True,
                )

        # ----------------------------------------------------
        # Mean baseline
        # ----------------------------------------------------

        if SAVE_MEAN_BASELINE:

            save_output_dataframe(
                df=mean_baseline_df,
                output_file=(
                    dataset_output_dir
                    / "predicted_ic50_mean_baseline.csv"
                ),
                log_file=log_file,
                index=True,
            )

        # ----------------------------------------------------
        # Top drug ranking
        # ----------------------------------------------------

        if SAVE_TOP_DRUG_RANKING:

            save_output_dataframe(
                df=top10_df,
                output_file=(
                    dataset_output_dir
                    / "baseline_top10_drugs_per_spot.csv"
                ),
                log_file=log_file,
                index=False,
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = create_dataset_summary(
            dataset_name=dataset_name,
            X_spot_pca=X_spot_pca,
            selected_drugs=selected_drugs,
            prediction_dict=prediction_dict,
            mean_baseline_df=mean_baseline_df,
            top10_df=top10_df,
        )

        summary_df = pd.DataFrame(
            [summary]
        )

        if SAVE_SUMMARY:

            save_output_dataframe(
                df=summary_df,
                output_file=(
                    dataset_output_dir
                    / "baseline_prediction_summary.csv"
                ),
                log_file=log_file,
                index=False,
            )

        log_print(
            f"Baseline prediction shape: {mean_baseline_df.shape}",
            log_file,
        )

        log_print(
            f"Top10 rows: {top10_df.shape[0]}",
            log_file,
        )

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
# 14. STREAMLIT HELPER
# ============================================================

def predict_uploaded_spatial_baseline(
    X_spot_pca: pd.DataFrame,
    selected_drugs: list[str],
    models: dict[str, Any],
    device: torch.device,
    top_k: int = TOP_K_DRUGS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Predict baseline IC50 for uploaded spatial dataset.

    Used by Streamlit app.

    Parameters
    ----------
    X_spot_pca : pd.DataFrame
        PCA feature dataframe.
    selected_drugs : list[str]
        Selected drugs.
    models : dict[str, Any]
        Loaded baseline models.
    device : torch.device
        Torch device.
    top_k : int
        Top drugs per spot.

    Returns
    -------
    tuple
        baseline_prediction_df,
        top10_df,
        summary
    """
    prediction_dict = predict_all_baseline_models(
        X_spot_pca=X_spot_pca,
        selected_drugs=selected_drugs,
        models=models,
        device=device,
        log_file=None,
    )

    baseline_df = create_mean_baseline_prediction(
        prediction_dict=prediction_dict,
        log_file=None,
    )

    top10_df = create_top10_drugs_per_spot(
        baseline_df=baseline_df,
        top_k=top_k,
    )

    summary = {
        "n_spots": int(baseline_df.shape[0]),
        "n_drugs": int(baseline_df.shape[1]),
        "models_success": ",".join(
            prediction_dict.keys()
        ),
        "top_k": int(top_k),
    }

    return (
        baseline_df,
        top10_df,
        summary,
    )


# ============================================================
# 15. MAIN
# ============================================================

def main() -> None:
    """
    Main baseline spatial prediction pipeline.
    """
    paths = setup_paths()

    log_file = (
        paths["log_file"]
        if SAVE_LOG
        else None
    )

    log_print("=" * 70, log_file)
    log_print(
        "PREDICT SPATIAL BASELINE",
        log_file,
    )
    log_print("=" * 70, log_file)

    log_print(
        f"PROJECT_ROOT: {paths['project_root']}",
        log_file,
    )

    log_print(
        f"DATASETS: {DATASETS}",
        log_file,
    )

    log_print(
        f"PREDICT_MODELS: {PREDICT_MODELS}",
        log_file,
    )

    log_print(
        f"BASELINE_METHOD: {BASELINE_METHOD}",
        log_file,
    )

    log_print(
        f"TOP_K_DRUGS: {TOP_K_DRUGS}",
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
        return

    selected_drugs = load_selected_drugs_file(
        paths=paths,
        log_file=log_file,
    )

    # --------------------------------------------------------
    # Infer PCA dimension from first dataset
    # --------------------------------------------------------

    first_dataset = DATASETS[0]

    first_x_file = (
        paths["spatial_processed_dir"]
        / first_dataset
        / "X_spot_pca.csv"
    )

    X_tmp = load_x_spot_pca(
        file_path=first_x_file,
        log_file=log_file,
    )

    input_dim = int(
        X_tmp.shape[1]
    )

    device = get_torch_device(
        DEVICE
    )

    models = load_prediction_objects(
        paths=paths,
        selected_drugs=selected_drugs,
        input_dim=input_dim,
        log_file=log_file,
    )

    report_rows = []

    for dataset_name in DATASETS:

        summary = process_one_dataset(
            dataset_name=dataset_name,
            paths=paths,
            selected_drugs=selected_drugs,
            models=models,
            device=device,
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
        / "spatial_baseline_prediction_report.csv"
    )

    save_output_dataframe(
        df=report_df,
        output_file=report_file,
        log_file=log_file,
        index=False,
    )

    log_print("=" * 70, log_file)
    log_print(
        "SPATIAL BASELINE REPORT",
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
        "FINISHED PREDICT SPATIAL BASELINE",
        log_file,
    )
    log_print("=" * 70, log_file)


if __name__ == "__main__":
    main()