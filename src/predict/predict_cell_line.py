from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn


PREDICT_MODE = "all_test"  # "all_test" hoặc "single_cell_line"
MODEL_NAME = "all"  # "ridge", "xgboost", "dnn" hoặc "all"
CELL_LINE_NAME = "MCF7"  # Chỉ dùng khi PREDICT_MODE = "single_cell_line"
TOP_K = 10

SAVE_LOG = True
DEVICE = "auto"

ALLOWED_MODELS = ["ridge", "xgboost", "dnn"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    project_root = get_project_root()

    test_dir = project_root / "data" / "processed" / "test"
    models_dir = project_root / "models"
    metrics_dir = project_root / "results" / "metrics"
    predictions_dir = project_root / "results" / "predictions"
    logs_dir = project_root / "logs"

    if PREDICT_MODE == "all_test":
        metrics_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "log_file": logs_dir / f"predict_cell_line_{run_time}.log",

        "X_test": test_dir / "X_test_pca.csv",
        "y_test": test_dir / "y_test.csv",
        "mask_test": test_dir / "mask_test.csv",

        "ridge_model": models_dir / "ridge_model.pkl",
        "xgboost_model": models_dir / "xgboost_model.pkl",
        "dnn_model": models_dir / "dnn_model.pkl",

        "metrics_dir": metrics_dir,
        "predictions_dir": predictions_dir,
    }


def log_print(message: str = "", log_file: Path | None = None) -> None:
    print(message)

    if (
        SAVE_LOG
        and PREDICT_MODE == "all_test"
        and log_file is not None
    ):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def add_project_root_to_sys_path(project_root: Path) -> None:
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.append(project_root_str)


def read_matrix(file_path: Path, log_file: Path | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(file_path, index_col=0)

    except Exception as e:
        log_print("=" * 70, log_file)
        log_print("LỖI ĐỌC FILE", log_file)
        log_print("=" * 70, log_file)
        log_print(f"Tên file: {file_path.name}", log_file)
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)
        return None


def standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def check_required_files(paths: dict, model_names: list[str], log_file: Path | None = None) -> bool:
    required_files = [
        paths["X_test"],
        paths["y_test"],
        paths["mask_test"],
    ]

    for model_name in model_names:
        required_files.append(paths[f"{model_name}_model"])

    missing_files = [
        file_path.name
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        log_print("=" * 70, log_file)
        log_print("LỖI: THIẾU FILE INPUT", log_file)
        log_print("=" * 70, log_file)

        for file_name in missing_files:
            log_print(f"- {file_name}", log_file)

        return False

    return True


def validate_test_dataset(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    mask_test: pd.DataFrame,
    log_file: Path | None = None,
):
    if X_test.shape[0] != y_test.shape[0]:
        log_print("LỖI: X_test và y_test khác số dòng.", log_file)
        return False, X_test, y_test, mask_test

    if set(X_test.index) != set(y_test.index):
        log_print("LỖI: X_test và y_test khác cell line.", log_file)
        return False, X_test, y_test, mask_test

    if not X_test.index.equals(y_test.index):
        y_test = y_test.loc[X_test.index]
        log_print("TEST: đã sắp xếp y_test theo X_test.", log_file)

    if y_test.shape != mask_test.shape:
        log_print("LỖI: y_test và mask_test khác shape.", log_file)
        return False, X_test, y_test, mask_test

    if set(y_test.index) != set(mask_test.index):
        log_print("LỖI: y_test và mask_test khác index.", log_file)
        return False, X_test, y_test, mask_test

    if set(y_test.columns) != set(mask_test.columns):
        log_print("LỖI: y_test và mask_test khác columns.", log_file)
        return False, X_test, y_test, mask_test

    mask_test = mask_test.loc[y_test.index, y_test.columns]

    return True, X_test, y_test, mask_test


def clean_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X


def clean_y(y: pd.DataFrame) -> pd.DataFrame:
    return y.apply(pd.to_numeric, errors="coerce")


def clean_mask(mask: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    mask = mask.apply(pd.to_numeric, errors="coerce")

    values = set(pd.unique(mask.to_numpy().ravel()))

    if not values.issubset({0, 1, 0.0, 1.0}):
        mask = y.notna().astype(float)
    else:
        mask = mask.astype(float)

    return mask


def load_test_dataset(paths: dict, log_file: Path | None = None):
    X_test = read_matrix(paths["X_test"], log_file)
    y_test = read_matrix(paths["y_test"], log_file)
    mask_test = read_matrix(paths["mask_test"], log_file)

    if any(df is None for df in [X_test, y_test, mask_test]):
        return None

    X_test = standardize_index(X_test)
    y_test = standardize_index(y_test)
    mask_test = standardize_index(mask_test)

    test_ok, X_test, y_test, mask_test = validate_test_dataset(
        X_test=X_test,
        y_test=y_test,
        mask_test=mask_test,
        log_file=log_file,
    )

    if not test_ok:
        return None

    X_test = clean_X(X_test)
    y_test = clean_y(y_test)
    mask_test = clean_mask(mask_test, y_test)

    return X_test, y_test, mask_test


class DNNMultiOutputRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(DEVICE)


def load_sklearn_model(model_path: Path):
    return joblib.load(model_path)


def load_dnn_model(
    model_path: Path,
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device)

    hidden_dims = checkpoint.get("hidden_dims", [512, 256])
    dropout = checkpoint.get("dropout", 0.2)

    model = DNNMultiOutputRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def get_model_names() -> list[str]:
    if MODEL_NAME == "all":
        return ALLOWED_MODELS

    if MODEL_NAME not in ALLOWED_MODELS:
        raise ValueError(
            f"MODEL_NAME không hợp lệ: {MODEL_NAME}. "
            f"Chỉ nhận: {ALLOWED_MODELS} hoặc 'all'."
        )

    return [MODEL_NAME]


# 5. DỰ ĐOÁN VÀ ĐÁNH GIÁ

def predict_sklearn_model(
    model,
    X: pd.DataFrame,
    output_columns,
) -> pd.DataFrame:
    pred = model.predict(X)

    return pd.DataFrame(
        pred,
        index=X.index,
        columns=output_columns,
    )


def predict_dnn_model(
    model: nn.Module,
    X: pd.DataFrame,
    output_columns,
    device: torch.device,
) -> pd.DataFrame:
    X_tensor = torch.tensor(
        X.to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        pred = model(X_tensor).detach().cpu().numpy()

    return pd.DataFrame(
        pred,
        index=X.index,
        columns=output_columns,
    )


def predict_by_model_name(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    output_columns,
    device: torch.device,
) -> pd.DataFrame:
    if model_name in ["ridge", "xgboost"]:
        return predict_sklearn_model(
            model=model,
            X=X_test,
            output_columns=output_columns,
        )

    if model_name == "dnn":
        return predict_dnn_model(
            model=model,
            X=X_test,
            output_columns=output_columns,
            device=device,
        )

    raise ValueError(f"Model không hợp lệ: {model_name}")


def masked_regression_metrics(y_true, y_pred, mask) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=float)

    valid_mask = (
        (mask > 0)
        & np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    observed_values = int((mask > 0).sum())
    valid_values = int(valid_mask.sum())

    if valid_values == 0:
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "observed_values": observed_values,
            "valid_values": valid_values,
        }

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))

    r2 = np.nan if ss_tot == 0 else float(1 - ss_res / ss_tot)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "observed_values": observed_values,
        "valid_values": valid_values,
    }


def evaluate_dataset(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict:
    try:
        from src.utils.metrics import calculate_all_metrics

        return calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
        )

    except Exception:
        return masked_regression_metrics(
            y_true=y_true.to_numpy(),
            y_pred=y_pred.to_numpy(),
            mask=mask.to_numpy(),
        )


def save_predictions(
    prediction_df: pd.DataFrame,
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    if output_file.exists():
        log_print(f"File predictions đã tồn tại, sẽ ghi đè: {output_file}", log_file)

    prediction_df.to_csv(output_file)


def save_metrics(
    metrics_df: pd.DataFrame,
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    if output_file.exists():
        log_print(f"File metrics đã tồn tại, sẽ ghi đè: {output_file}", log_file)

    metrics_df.to_csv(output_file, index=False)


def predict_all_test(
    model_names: list[str],
    paths: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    mask_test: pd.DataFrame,
    device: torch.device,
    log_file: Path,
) -> None:
    metric_rows = []

    for model_name in model_names:
        log_print("=" * 70, log_file)
        log_print(f"DỰ ĐOÁN TEST VỚI MODEL: {model_name.upper()}", log_file)
        log_print("=" * 70, log_file)

        if model_name == "dnn":
            model = load_dnn_model(
                model_path=paths["dnn_model"],
                input_dim=X_test.shape[1],
                output_dim=y_test.shape[1],
                device=device,
            )
        else:
            model = load_sklearn_model(paths[f"{model_name}_model"])

        y_pred = predict_by_model_name(
            model_name=model_name,
            model=model,
            X_test=X_test,
            output_columns=y_test.columns,
            device=device,
        )

        metrics = evaluate_dataset(
            y_true=y_test,
            y_pred=y_pred,
            mask=mask_test,
        )

        prediction_output = paths["predictions_dir"] / f"{model_name}_test_predictions.csv"
        save_predictions(
            prediction_df=y_pred,
            output_file=prediction_output,
            log_file=log_file,
        )

        row = {
            "model": model_name,
            "dataset": "test",
            **metrics,
        }

        metric_rows.append(row)

        log_print(f"Test MAE: {metrics['MAE']:.6f}", log_file)
        log_print(f"Test RMSE: {metrics['RMSE']:.6f}", log_file)
        log_print(f"Test R2: {metrics['R2']:.6f}", log_file)
        log_print(f"Predictions: {prediction_output}", log_file)

    metrics_df = pd.DataFrame(metric_rows)

    if len(model_names) == 1:
        metrics_output = paths["metrics_dir"] / f"{model_names[0]}_test_metrics.csv"
    else:
        metrics_output = paths["metrics_dir"] / "cell_line_test_metrics_all_models.csv"

    save_metrics(
        metrics_df=metrics_df,
        output_file=metrics_output,
        log_file=log_file,
    )

    log_print("=" * 70, log_file)
    log_print("ĐÃ LƯU KẾT QUẢ TEST", log_file)
    log_print("=" * 70, log_file)
    log_print(f"Metrics: {metrics_output}", log_file)
    log_print(f"Log file: {log_file}", log_file)


def print_cell_line_prediction(
    model_name: str,
    y_pred_cell: pd.Series,
    y_true_cell: pd.Series,
    mask_cell: pd.Series,
) -> None:
    result_df = pd.DataFrame({
        "drug": y_pred_cell.index,
        "predicted_LN_IC50": y_pred_cell.values,
        "true_LN_IC50": y_true_cell.values,
        "mask": mask_cell.values,
    })

    result_df["abs_error"] = np.where(
        result_df["mask"] > 0,
        np.abs(result_df["predicted_LN_IC50"] - result_df["true_LN_IC50"]),
        np.nan,
    )

    result_df = result_df.sort_values(
        by="predicted_LN_IC50",
        ascending=True,
    ).reset_index(drop=True)

    log_print("=" * 70)
    log_print(f"MODEL: {model_name.upper()}")
    log_print("=" * 70)
    log_print("Top thuốc có LN(IC50) dự đoán thấp nhất:")
    log_print("")

    display_cols = [
        "drug",
        "predicted_LN_IC50",
        "true_LN_IC50",
        "mask",
        "abs_error",
    ]

    log_print(
        result_df[display_cols]
        .head(TOP_K)
        .to_string(index=False)
    )
    log_print("")


def predict_single_cell_line(
    model_names: list[str],
    paths: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    mask_test: pd.DataFrame,
    device: torch.device,
) -> None:
    cell_line = str(CELL_LINE_NAME).strip()

    if cell_line not in X_test.index:
        log_print("=" * 70)
        log_print("LỖI: CELL LINE KHÔNG TỒN TẠI TRONG TẬP TEST")
        log_print("=" * 70)
        log_print(f"Cell line đã nhập: {cell_line}")
        log_print("")
        log_print("Một số cell line có trong tập test:")
        log_print(", ".join(X_test.index[:20].tolist()))
        return

    X_cell = X_test.loc[[cell_line]]
    y_true_cell = y_test.loc[cell_line]
    mask_cell = mask_test.loc[cell_line]

    log_print("=" * 70)
    log_print("DỰ ĐOÁN 1 CELL LINE TRONG TẬP TEST")
    log_print("=" * 70)
    log_print(f"CELL_LINE_NAME: {cell_line}")
    log_print(f"Số thuốc: {y_test.shape[1]}")
    log_print(f"TOP_K: {TOP_K}")
    log_print("")

    for model_name in model_names:
        if model_name == "dnn":
            model = load_dnn_model(
                model_path=paths["dnn_model"],
                input_dim=X_test.shape[1],
                output_dim=y_test.shape[1],
                device=device,
            )
        else:
            model = load_sklearn_model(paths[f"{model_name}_model"])

        y_pred = predict_by_model_name(
            model_name=model_name,
            model=model,
            X_test=X_cell,
            output_columns=y_test.columns,
            device=device,
        )

        y_pred_cell = y_pred.loc[cell_line]

        print_cell_line_prediction(
            model_name=model_name,
            y_pred_cell=y_pred_cell,
            y_true_cell=y_true_cell,
            mask_cell=mask_cell,
        )


def main() -> None:
    paths = setup_paths()
    log_file = paths["log_file"]

    add_project_root_to_sys_path(paths["project_root"])

    model_names = get_model_names()
    device = get_device()

    if not check_required_files(
        paths=paths,
        model_names=model_names,
        log_file=log_file,
    ):
        log_print("CHƯƠNG TRÌNH DỪNG DO THIẾU FILE INPUT.", log_file)
        return

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU PREDICT CELL LINE", log_file)
    log_print("=" * 70, log_file)
    log_print(f"PROJECT_ROOT: {paths['project_root']}", log_file)
    log_print(f"PREDICT_MODE: {PREDICT_MODE}", log_file)
    log_print(f"MODEL_NAME: {MODEL_NAME}", log_file)
    log_print(f"DEVICE: {device}", log_file)

    data = load_test_dataset(paths, log_file)

    if data is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO LỖI DỮ LIỆU TEST.", log_file)
        return

    X_test, y_test, mask_test = data

    log_print("=" * 70, log_file)
    log_print("THÔNG TIN TEST DATASET", log_file)
    log_print("=" * 70, log_file)
    log_print(f"X_test shape: {X_test.shape}", log_file)
    log_print(f"y_test shape: {y_test.shape}", log_file)
    log_print(f"mask_test shape: {mask_test.shape}", log_file)

    if PREDICT_MODE == "all_test":
        predict_all_test(
            model_names=model_names,
            paths=paths,
            X_test=X_test,
            y_test=y_test,
            mask_test=mask_test,
            device=device,
            log_file=log_file,
        )

    elif PREDICT_MODE == "single_cell_line":
        predict_single_cell_line(
            model_names=model_names,
            paths=paths,
            X_test=X_test,
            y_test=y_test,
            mask_test=mask_test,
            device=device,
        )

    else:
        log_print("=" * 70, log_file)
        log_print("LỖI: PREDICT_MODE KHÔNG HỢP LỆ", log_file)
        log_print("=" * 70, log_file)
        log_print("PREDICT_MODE chỉ nhận 'all_test' hoặc 'single_cell_line'.", log_file)


if __name__ == "__main__":
    main()
