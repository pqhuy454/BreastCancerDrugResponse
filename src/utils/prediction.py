# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_HIDDEN_DIMS = [128, 64]
DEFAULT_DROPOUT = 0.2
DEFAULT_DEVICE = "auto"

DEFAULT_ENSEMBLE_METHOD = "mean"
DEFAULT_TOP_K_DRUGS = 10


# ============================================================
# 2. PROJECT ROOT AND LIST FILES
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


def read_list_file(file_path: Path) -> list[str]:
    """
    Read a text file where each line is one item.

    Parameters
    ----------
    file_path : Path
        Path to text file.

    Returns
    -------
    list[str]
        Unique values in original order.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    values: list[str] = []
    seen_values: set[str] = set()

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            value = str(line).strip()

            if not value:
                continue

            if value not in seen_values:
                values.append(value)
                seen_values.add(value)

    return values


def read_selected_genes(file_path: Path) -> list[str]:
    """
    Read selected genes from text file.

    Parameters
    ----------
    file_path : Path
        Path to selected_genes.txt.

    Returns
    -------
    list[str]
        Selected genes.
    """
    return read_list_file(file_path)


def read_selected_drugs(file_path: Path) -> list[str]:
    """
    Read selected drugs from text file.

    Parameters
    ----------
    file_path : Path
        Path to selected_drugs.txt.

    Returns
    -------
    list[str]
        Selected drugs.
    """
    return read_list_file(file_path)


# ============================================================
# 3. LOAD TRANSFORMERS AND MODELS
# ============================================================

def load_joblib_object(file_path: Path) -> Any:
    """
    Load object using joblib.

    Parameters
    ----------
    file_path : Path
        Object file path.

    Returns
    -------
    Any
        Loaded object.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    return joblib.load(file_path)


def load_scaler(file_path: Path) -> Any:
    """
    Load scaler object.

    Parameters
    ----------
    file_path : Path
        Path to scaler.pkl.

    Returns
    -------
    Any
        Loaded scaler.
    """
    return load_joblib_object(file_path)


def load_pca(file_path: Path) -> Any:
    """
    Load PCA object.

    Parameters
    ----------
    file_path : Path
        Path to pca.pkl.

    Returns
    -------
    Any
        Loaded PCA.
    """
    return load_joblib_object(file_path)


def load_sklearn_model(file_path: Path) -> Any:
    """
    Load sklearn-like model.

    Parameters
    ----------
    file_path : Path
        Path to model file.

    Returns
    -------
    Any
        Loaded model.
    """
    return load_joblib_object(file_path)


# ============================================================
# 4. TORCH DEVICE
# ============================================================

def get_torch_device(device: str = DEFAULT_DEVICE) -> torch.device:
    """
    Get PyTorch device.

    Parameters
    ----------
    device : str
        Device name or 'auto'.

    Returns
    -------
    torch.device
        Torch device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


# ============================================================
# 5. DNN MODEL
# ============================================================

class DNNMultiOutputRegressor(nn.Module):
    """
    DNN multi-output regressor compatible with train_dnn.py.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        """
        Initialize DNN model.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        output_dim : int
            Output dimension.
        hidden_dims : list[int]
            Hidden layer dimensions.
        dropout : float
            Dropout rate.
        """
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.network(x)


def build_dnn_model(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | None = None,
    dropout: float = DEFAULT_DROPOUT,
) -> DNNMultiOutputRegressor:
    """
    Build DNN model.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output dimension.
    hidden_dims : list[int] | None
        Hidden layer dimensions.
    dropout : float
        Dropout rate.

    Returns
    -------
    DNNMultiOutputRegressor
        DNN model.
    """
    if hidden_dims is None:
        hidden_dims = DEFAULT_HIDDEN_DIMS

    return DNNMultiOutputRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


def _prepare_loaded_dnn_object(
    loaded_object: Any,
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> Any:
    """
    Prepare loaded DNN object from joblib or torch checkpoint.

    Parameters
    ----------
    loaded_object : Any
        Loaded object.
    input_dim : int
        Default input dimension.
    output_dim : int
        Default output dimension.
    device : torch.device
        Torch device.

    Returns
    -------
    Any
        Prepared model object.
    """
    if isinstance(loaded_object, nn.Module):
        loaded_object.to(device)
        loaded_object.eval()
        return loaded_object

    if isinstance(loaded_object, dict) and "model_state_dict" in loaded_object:
        model_input_dim = int(loaded_object.get("input_dim", input_dim))
        model_output_dim = int(loaded_object.get("output_dim", output_dim))
        hidden_dims = loaded_object.get("hidden_dims", DEFAULT_HIDDEN_DIMS)
        dropout = float(loaded_object.get("dropout", DEFAULT_DROPOUT))

        model = build_dnn_model(
            input_dim=model_input_dim,
            output_dim=model_output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        model.load_state_dict(loaded_object["model_state_dict"])
        model.to(device)
        model.eval()

        return model

    if hasattr(loaded_object, "predict"):
        return loaded_object

    raise TypeError(
        "Không nhận diện được định dạng DNN model. "
        "Cần torch.nn.Module, dict có model_state_dict, hoặc object có .predict()."
    )


def load_dnn_model(
    model_file: Path,
    input_dim: int,
    output_dim: int,
    device: torch.device | None = None,
) -> Any:
    """
    Load DNN model flexibly.

    Parameters
    ----------
    model_file : Path
        Path to DNN model file.
    input_dim : int
        Default input dimension.
    output_dim : int
        Default output dimension.
    device : torch.device | None
        Torch device.

    Returns
    -------
    Any
        Loaded DNN model.
    """
    if not model_file.exists():
        raise FileNotFoundError(f"File không tồn tại: {model_file}")

    if device is None:
        device = get_torch_device()

    joblib_error: Exception | None = None
    torch_error: Exception | None = None

    try:
        loaded_object = joblib.load(model_file)
        return _prepare_loaded_dnn_object(
            loaded_object=loaded_object,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
        )
    except Exception as error:
        joblib_error = error

    try:
        loaded_object = torch.load(
            model_file,
            map_location=device,
            weights_only=False,
        )
        return _prepare_loaded_dnn_object(
            loaded_object=loaded_object,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
        )
    except Exception as error:
        torch_error = error

    raise RuntimeError(
        "Không load được DNN model bằng joblib hoặc torch.\n"
        f"Joblib error: {joblib_error}\n"
        f"Torch error: {torch_error}"
    )


# ============================================================
# 6. READ AND STANDARDIZE MATRIX
# ============================================================

def standardize_dataframe_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe index and columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Standardized dataframe.
    """
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def convert_matrix_to_numeric(
    df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Convert matrix values to numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fill_value : float
        Value used to fill missing values.

    Returns
    -------
    pd.DataFrame
        Numeric dataframe.
    """
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(fill_value)
    return df


def read_matrix_csv(
    file_path: Path,
    index_col: int | str | None = 0,
) -> pd.DataFrame:
    """
    Read matrix CSV file.

    Parameters
    ----------
    file_path : Path
        CSV file path.
    index_col : int | str | None
        Index column.

    Returns
    -------
    pd.DataFrame
        Matrix dataframe.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    df = pd.read_csv(file_path, index_col=index_col)
    df = standardize_dataframe_index_columns(df)
    df = convert_matrix_to_numeric(df)

    return df


# ============================================================
# 7. ALIGN GENE FEATURES
# ============================================================

def align_features_to_selected_genes(
    X: pd.DataFrame,
    selected_genes: list[str],
    fill_missing_value: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Align expression matrix to selected genes.

    Parameters
    ----------
    X : pd.DataFrame
        Expression matrix.
    selected_genes : list[str]
        Selected genes.
    fill_missing_value : float
        Fill value for missing genes.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Aligned matrix and overlap statistics.
    """
    X = standardize_dataframe_index_columns(X)

    input_features = list(X.columns)
    input_feature_set = set(input_features)

    matched_genes = [gene for gene in selected_genes if gene in input_feature_set]
    missing_genes = [gene for gene in selected_genes if gene not in input_feature_set]

    X_aligned = pd.DataFrame(
        fill_missing_value,
        index=X.index,
        columns=selected_genes,
    )

    if matched_genes:
        X_aligned.loc[:, matched_genes] = X.loc[:, matched_genes].to_numpy()

    n_input_features = len(input_features)
    n_selected_genes = len(selected_genes)
    n_matched_genes = len(matched_genes)
    n_missing_genes = len(missing_genes)

    coverage_percent = (
        n_matched_genes / n_selected_genes * 100
        if n_selected_genes > 0
        else 0.0
    )

    stats = {
        "n_input_features": n_input_features,
        "n_selected_genes": n_selected_genes,
        "n_matched_genes": n_matched_genes,
        "n_missing_genes": n_missing_genes,
        "coverage_percent": coverage_percent,
    }

    return X_aligned, stats


# ============================================================
# 8. TRANSFORM WITH SCALER AND PCA
# ============================================================

def transform_with_scaler(
    X: pd.DataFrame,
    scaler: Any,
) -> pd.DataFrame:
    """
    Transform features using trained scaler.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature dataframe.
    scaler : Any
        Trained scaler.

    Returns
    -------
    pd.DataFrame
        Scaled feature dataframe.
    """
    if hasattr(scaler, "n_features_in_"):
        expected_features = int(scaler.n_features_in_)
        actual_features = X.shape[1]

        if expected_features != actual_features:
            raise ValueError(
                f"Scaler feature mismatch: expected={expected_features}, "
                f"actual={actual_features}"
            )

    X_scaled_values = scaler.transform(X)

    X_scaled = pd.DataFrame(
        X_scaled_values,
        index=X.index,
        columns=X.columns,
    )

    X_scaled.index.name = X.index.name

    return X_scaled


def transform_with_pca(
    X: pd.DataFrame,
    pca: Any,
) -> pd.DataFrame:
    """
    Transform features using trained PCA.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature dataframe.
    pca : Any
        Trained PCA.

    Returns
    -------
    pd.DataFrame
        PCA feature dataframe.
    """
    if hasattr(pca, "n_features_in_"):
        expected_features = int(pca.n_features_in_)
        actual_features = X.shape[1]

        if expected_features != actual_features:
            raise ValueError(
                f"PCA feature mismatch: expected={expected_features}, "
                f"actual={actual_features}"
            )

    X_pca_values = pca.transform(X)

    n_components = X_pca_values.shape[1]
    pca_columns = [f"PC{i}" for i in range(1, n_components + 1)]

    X_pca = pd.DataFrame(
        X_pca_values,
        index=X.index,
        columns=pca_columns,
    )

    X_pca.index.name = X.index.name

    return X_pca


def preprocess_expression_to_pca(
    X_expression: pd.DataFrame,
    selected_genes: list[str],
    scaler: Any,
    pca: Any,
    fill_missing_gene_value: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Preprocess expression matrix to PCA features.

    Pipeline:
    align genes -> scaler transform -> PCA transform.

    Parameters
    ----------
    X_expression : pd.DataFrame
        Raw expression matrix.
    selected_genes : list[str]
        Selected training genes.
    scaler : Any
        Trained scaler.
    pca : Any
        Trained PCA.
    fill_missing_gene_value : float
        Fill value for missing genes.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        PCA feature dataframe and processing stats.
    """
    X_aligned, stats = align_features_to_selected_genes(
        X=X_expression,
        selected_genes=selected_genes,
        fill_missing_value=fill_missing_gene_value,
    )

    X_scaled = transform_with_scaler(
        X=X_aligned,
        scaler=scaler,
    )

    X_pca = transform_with_pca(
        X=X_scaled,
        pca=pca,
    )

    stats["aligned_shape"] = X_aligned.shape
    stats["scaled_shape"] = X_scaled.shape
    stats["pca_shape"] = X_pca.shape

    return X_pca, stats


# ============================================================
# 9. PREDICT IC50
# ============================================================

def adjust_prediction_output(
    y_pred: np.ndarray,
    index: pd.Index,
    selected_drugs: list[str],
) -> pd.DataFrame:
    """
    Convert prediction array to dataframe and align output columns.

    Parameters
    ----------
    y_pred : np.ndarray
        Prediction array.
    index : pd.Index
        Sample index.
    selected_drugs : list[str]
        Selected drugs.

    Returns
    -------
    pd.DataFrame
        Prediction dataframe.
    """
    y_pred = np.asarray(y_pred, dtype=float)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_outputs = y_pred.shape[1]
    n_drugs = len(selected_drugs)
    n_use = min(n_outputs, n_drugs)

    y_pred = y_pred[:, :n_use]
    drug_columns = selected_drugs[:n_use]

    pred_df = pd.DataFrame(
        y_pred,
        index=index,
        columns=drug_columns,
    )

    pred_df.index.name = index.name

    return pred_df


def _validate_model_input_features(model: Any, X: pd.DataFrame) -> None:
    """
    Validate model input feature count if available.

    Parameters
    ----------
    model : Any
        Model object.
    X : pd.DataFrame
        Input feature dataframe.
    """
    expected_features = None

    if hasattr(model, "n_features_in_"):
        expected_features = int(model.n_features_in_)

    elif hasattr(model, "estimators_"):
        estimators = getattr(model, "estimators_", [])

        if len(estimators) > 0 and hasattr(estimators[0], "n_features_in_"):
            expected_features = int(estimators[0].n_features_in_)

    if expected_features is not None and expected_features != X.shape[1]:
        raise ValueError(
            f"Model feature mismatch: expected={expected_features}, "
            f"actual={X.shape[1]}"
        )


def predict_with_sklearn_model(
    model: Any,
    X: pd.DataFrame,
    selected_drugs: list[str],
) -> pd.DataFrame:
    """
    Predict IC50 using sklearn-like model.

    Parameters
    ----------
    model : Any
        Sklearn-like model.
    X : pd.DataFrame
        Input feature dataframe.
    selected_drugs : list[str]
        Selected drugs.

    Returns
    -------
    pd.DataFrame
        Prediction dataframe.
    """
    _validate_model_input_features(model, X)

    y_pred = model.predict(X)

    return adjust_prediction_output(
        y_pred=y_pred,
        index=X.index,
        selected_drugs=selected_drugs,
    )


def _get_dnn_expected_input_dim(model: nn.Module) -> int | None:
    """
    Get DNN expected input dimension from first Linear layer.

    Parameters
    ----------
    model : nn.Module
        DNN model.

    Returns
    -------
    int | None
        Input dimension if found.
    """
    if not hasattr(model, "network"):
        return None

    for layer in model.network:
        if isinstance(layer, nn.Linear):
            return int(layer.in_features)

    return None


def predict_with_dnn_model(
    model: Any,
    X: pd.DataFrame,
    selected_drugs: list[str],
    device: torch.device | None = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Predict IC50 using DNN model.

    Parameters
    ----------
    model : Any
        DNN model or object with predict().
    X : pd.DataFrame
        Input feature dataframe.
    selected_drugs : list[str]
        Selected drugs.
    device : torch.device | None
        Torch device.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    pd.DataFrame
        Prediction dataframe.
    """
    if hasattr(model, "predict") and not isinstance(model, nn.Module):
        return predict_with_sklearn_model(
            model=model,
            X=X,
            selected_drugs=selected_drugs,
        )

    if not isinstance(model, nn.Module):
        raise TypeError("DNN model phải là torch.nn.Module hoặc object có .predict().")

    if device is None:
        device = get_torch_device()

    expected_input_dim = _get_dnn_expected_input_dim(model)

    if expected_input_dim is not None and expected_input_dim != X.shape[1]:
        raise ValueError(
            f"DNN feature mismatch: expected={expected_input_dim}, "
            f"actual={X.shape[1]}"
        )

    model.to(device)
    model.eval()

    X_values = X.to_numpy(dtype=np.float32)
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for start_idx in range(0, X_values.shape[0], batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_values[start_idx:end_idx]

            X_tensor = torch.tensor(
                X_batch,
                dtype=torch.float32,
                device=device,
            )

            batch_pred = model(X_tensor).detach().cpu().numpy()
            predictions.append(batch_pred)

    y_pred = np.vstack(predictions)

    return adjust_prediction_output(
        y_pred=y_pred,
        index=X.index,
        selected_drugs=selected_drugs,
    )


# ============================================================
# 10. ENSEMBLE PREDICTION
# ============================================================

def create_ensemble_prediction(
    prediction_dict: dict[str, pd.DataFrame],
    method: str = DEFAULT_ENSEMBLE_METHOD,
) -> pd.DataFrame:
    """
    Create ensemble prediction from multiple model predictions.

    Parameters
    ----------
    prediction_dict : dict[str, pd.DataFrame]
        Dictionary of prediction dataframes.
    method : str
        Ensemble method. Currently supports 'mean'.

    Returns
    -------
    pd.DataFrame
        Ensemble prediction dataframe.
    """
    valid_predictions = {
        model_name: pred_df
        for model_name, pred_df in prediction_dict.items()
        if pred_df is not None and isinstance(pred_df, pd.DataFrame) and not pred_df.empty
    }

    if len(valid_predictions) == 0:
        raise ValueError("Không có prediction hợp lệ để ensemble.")

    if len(valid_predictions) == 1:
        return next(iter(valid_predictions.values())).copy()

    common_index = None
    common_columns = None

    for pred_df in valid_predictions.values():
        if common_index is None:
            common_index = pred_df.index
        else:
            common_index = common_index.intersection(pred_df.index)

        if common_columns is None:
            common_columns = pd.Index(pred_df.columns)
        else:
            common_columns = common_columns.intersection(pd.Index(pred_df.columns))

    if common_index is None or common_columns is None:
        raise ValueError("Không xác định được index hoặc columns chung.")

    if len(common_index) == 0:
        raise ValueError("Không có sample_id chung giữa các prediction.")

    if len(common_columns) == 0:
        raise ValueError("Không có drug chung giữa các prediction.")

    arrays = []

    for pred_df in valid_predictions.values():
        aligned_df = pred_df.loc[common_index, common_columns]
        arrays.append(aligned_df.to_numpy(dtype=float))

    stacked = np.stack(arrays, axis=0)

    if method == "mean":
        ensemble_values = np.mean(stacked, axis=0)
    else:
        raise ValueError(f"Chưa hỗ trợ ensemble method: {method}")

    ensemble_df = pd.DataFrame(
        ensemble_values,
        index=common_index,
        columns=common_columns,
    )

    ensemble_df.index.name = common_index.name

    return ensemble_df


# ============================================================
# 11. DRUG RANKING
# ============================================================

def create_drug_ranking(
    pred_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K_DRUGS,
) -> pd.DataFrame:
    """
    Create top-k drug ranking for each sample.

    Lower predicted LN_IC50 means higher drug sensitivity.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction dataframe.
    top_k : int
        Number of top drugs.

    Returns
    -------
    pd.DataFrame
        Long-format ranking dataframe.
    """
    rows: list[dict[str, Any]] = []
    top_k = min(top_k, pred_df.shape[1])

    for sample_id, pred_series in pred_df.iterrows():
        sorted_series = pred_series.sort_values(ascending=True)

        for rank, (drug, predicted_value) in enumerate(
            sorted_series.iloc[:top_k].items(),
            start=1,
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "rank": rank,
                    "drug": drug,
                    "predicted_LN_IC50": float(predicted_value),
                }
            )

    return pd.DataFrame(rows)


def create_drug_ranking_for_sample(
    pred_series: pd.Series,
    top_k: int = DEFAULT_TOP_K_DRUGS,
) -> pd.DataFrame:
    """
    Create top-k drug ranking for one sample.

    Parameters
    ----------
    pred_series : pd.Series
        Prediction series for one sample.
    top_k : int
        Number of top drugs.

    Returns
    -------
    pd.DataFrame
        Ranking dataframe for one sample.
    """
    pred_series = pd.to_numeric(pred_series, errors="coerce")
    pred_series = pred_series.replace([np.inf, -np.inf], np.nan)
    pred_series = pred_series.dropna()

    top_k = min(top_k, pred_series.shape[0])
    sorted_series = pred_series.sort_values(ascending=True).iloc[:top_k]

    rows = []

    for rank, (drug, predicted_value) in enumerate(sorted_series.items(), start=1):
        rows.append(
            {
                "sample_id": pred_series.name,
                "rank": rank,
                "drug": drug,
                "predicted_LN_IC50": float(predicted_value),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 12. SAVE OUTPUT
# ============================================================

def save_dataframe(
    df: pd.DataFrame,
    output_file: Path,
    index: bool = True,
    overwrite: bool = True,
) -> bool:
    """
    Save dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    output_file : Path
        Output CSV file path.
    index : bool
        Whether to save index.
    overwrite : bool
        Whether to overwrite existing file.

    Returns
    -------
    bool
        True if saved, otherwise False.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        return False

    df.to_csv(output_file, index=index)

    return True