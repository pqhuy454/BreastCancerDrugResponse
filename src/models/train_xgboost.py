from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


MODEL_NAME = "xgboost"

N_ESTIMATORS = 300
MAX_DEPTH = 4
LEARNING_RATE = 0.03
SUBSAMPLE = 0.7
COLSAMPLE_BYTREE = 0.9

OBJECTIVE = "reg:squarederror"
TREE_METHOD = "hist"

RANDOM_STATE = 42
N_JOBS = -1
SAVE_LOG = True


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    project_root = get_project_root()

    train_dir = project_root / "data" / "processed" / "train"
    val_dir = project_root / "data" / "processed" / "validation"

    models_dir = project_root / "models"
    metrics_dir = project_root / "results" / "metrics"
    logs_dir = project_root / "logs"
    predictions_dir = project_root / "results" / "predictions"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "log_file": logs_dir / f"train_xgboost_{run_time}.log",

        "X_train": train_dir / "X_train_pca.csv",
        "y_train": train_dir / "y_train.csv",
        "mask_train": train_dir / "mask_train.csv",

        "X_val": val_dir / "X_val_pca.csv",
        "y_val": val_dir / "y_val.csv",
        "mask_val": val_dir / "mask_val.csv",

        "model_output": models_dir / "xgboost_model.pkl",
        "metrics_output": metrics_dir / "xgboost_model_train_val_metrics.csv",
        "prediction_output": predictions_dir / "xgboost_validation_predictions.csv",
    }


def log_print(message: str = "", log_file: Path | None = None) -> None:
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def read_matrix(file_path: Path, log_file: Path) -> pd.DataFrame | None:
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


def validate_y_mask(
    y: pd.DataFrame,
    mask: pd.DataFrame,
    dataset_name: str,
    log_file: Path,
) -> tuple[bool, pd.DataFrame]:
    if y.shape != mask.shape:
        log_print(f"LỖI: {dataset_name} y và mask khác shape.", log_file)
        return False, mask

    if set(y.index) != set(mask.index):
        log_print(f"LỖI: {dataset_name} y và mask khác index.", log_file)
        return False, mask

    if set(y.columns) != set(mask.columns):
        log_print(f"LỖI: {dataset_name} y và mask khác columns.", log_file)
        return False, mask

    mask = mask.loc[y.index, y.columns]

    return True, mask


def load_dataset(paths: dict, log_file: Path):
    required_files = [
        paths["X_train"],
        paths["y_train"],
        paths["mask_train"],
        paths["X_val"],
        paths["y_val"],
        paths["mask_val"],
    ]

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

        return None

    X_train = read_matrix(paths["X_train"], log_file)
    y_train = read_matrix(paths["y_train"], log_file)
    mask_train = read_matrix(paths["mask_train"], log_file)

    X_val = read_matrix(paths["X_val"], log_file)
    y_val = read_matrix(paths["y_val"], log_file)
    mask_val = read_matrix(paths["mask_val"], log_file)

    if any(df is None for df in [X_train, y_train, mask_train, X_val, y_val, mask_val]):
        return None

    X_train = standardize_index(X_train)
    y_train = standardize_index(y_train)
    mask_train = standardize_index(mask_train)

    X_val = standardize_index(X_val)
    y_val = standardize_index(y_val)
    mask_val = standardize_index(mask_val)

    if X_train.shape[0] != y_train.shape[0]:
        log_print("LỖI: X_train và y_train khác số dòng.", log_file)
        return None

    if X_val.shape[0] != y_val.shape[0]:
        log_print("LỖI: X_val và y_val khác số dòng.", log_file)
        return None

    if set(X_train.index) != set(y_train.index):
        log_print("LỖI: X_train và y_train khác cell line.", log_file)
        return None

    if set(X_val.index) != set(y_val.index):
        log_print("LỖI: X_val và y_val khác cell line.", log_file)
        return None

    y_train = y_train.loc[X_train.index]
    y_val = y_val.loc[X_val.index]

    train_ok, mask_train = validate_y_mask(
        y=y_train,
        mask=mask_train,
        dataset_name="TRAIN",
        log_file=log_file,
    )

    val_ok, mask_val = validate_y_mask(
        y=y_val,
        mask=mask_val,
        dataset_name="VALIDATION",
        log_file=log_file,
    )

    if not train_ok or not val_ok:
        return None

    X_train = (
        X_train
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_val = (
        X_val
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_train = y_train.apply(pd.to_numeric, errors="coerce")
    y_val = y_val.apply(pd.to_numeric, errors="coerce")

    mask_train = mask_train.apply(pd.to_numeric, errors="coerce")
    mask_val = mask_val.apply(pd.to_numeric, errors="coerce")

    if not set(pd.unique(mask_train.to_numpy().ravel())).issubset({0, 1, 0.0, 1.0}):
        mask_train = y_train.notna().astype(int)

    if not set(pd.unique(mask_val.to_numpy().ravel())).issubset({0, 1, 0.0, 1.0}):
        mask_val = y_val.notna().astype(int)

    return X_train, y_train, mask_train, X_val, y_val, mask_val


def fill_y_train_missing(y_train: pd.DataFrame) -> pd.DataFrame:
    y_filled = y_train.copy()

    medians = y_filled.median(axis=0, skipna=True)
    medians = medians.fillna(0)

    y_filled = y_filled.fillna(medians)
    y_filled = y_filled.fillna(0)

    return y_filled


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


def build_xgboost_model(use_tree_method: bool = True):
    if use_tree_method:
        base_model = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            objective=OBJECTIVE,
            tree_method=TREE_METHOD,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
    else:
        base_model = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            objective=OBJECTIVE,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

    return MultiOutputRegressor(
        base_model,
        n_jobs=N_JOBS,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train_filled: pd.DataFrame,
    log_file: Path,
):
    try:
        model = build_xgboost_model(use_tree_method=True)
        model.fit(X_train, y_train_filled)
        return model

    except Exception as e:
        log_print(
            f"CẢNH BÁO: Train XGBoost với tree_method='{TREE_METHOD}' bị lỗi.",
            log_file,
        )
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)
        log_print("Thử train lại bằng cách bỏ tree_method.", log_file)

        model = build_xgboost_model(use_tree_method=False)
        model.fit(X_train, y_train_filled)
        return model


def save_model(model, output_file: Path, log_file: Path) -> None:
    if output_file.exists():
        log_print(f"File model đã tồn tại, sẽ ghi đè: {output_file}", log_file)

    joblib.dump(model, output_file)


def save_metrics(metrics_df: pd.DataFrame, output_file: Path, log_file: Path) -> None:
    if output_file.exists():
        log_print(f"File metrics đã tồn tại, sẽ ghi đè: {output_file}", log_file)

    metrics_df.to_csv(output_file, index=False)


def save_validation_predictions(
    prediction_df: pd.DataFrame,
    output_file: Path,
    log_file: Path,
) -> None:
    if output_file.exists():
        log_print(
            f"File validation prediction đã tồn tại, sẽ ghi đè: {output_file}",
            log_file,
        )

    prediction_df.to_csv(output_file)


def main() -> None:
    paths = setup_paths()
    log_file = paths["log_file"]

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU TRAIN XGBOOST", log_file)
    log_print("=" * 70, log_file)

    log_print(f"PROJECT_ROOT: {paths['project_root']}", log_file)
    log_print(f"MODEL_NAME: {MODEL_NAME}", log_file)
    log_print(f"N_ESTIMATORS: {N_ESTIMATORS}", log_file)
    log_print(f"MAX_DEPTH: {MAX_DEPTH}", log_file)
    log_print(f"LEARNING_RATE: {LEARNING_RATE}", log_file)
    log_print(f"SUBSAMPLE: {SUBSAMPLE}", log_file)
    log_print(f"COLSAMPLE_BYTREE: {COLSAMPLE_BYTREE}", log_file)
    log_print(f"OBJECTIVE: {OBJECTIVE}", log_file)
    log_print(f"TREE_METHOD: {TREE_METHOD}", log_file)

    data = load_dataset(paths, log_file)

    if data is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO LỖI DỮ LIỆU INPUT.", log_file)
        return

    X_train, y_train, mask_train, X_val, y_val, mask_val = data

    log_print("=" * 70, log_file)
    log_print("THÔNG TIN DATASET", log_file)
    log_print("=" * 70, log_file)
    log_print(f"X_train shape: {X_train.shape}", log_file)
    log_print(f"y_train shape: {y_train.shape}", log_file)
    log_print(f"mask_train shape: {mask_train.shape}", log_file)
    log_print(f"X_val shape: {X_val.shape}", log_file)
    log_print(f"y_val shape: {y_val.shape}", log_file)
    log_print(f"mask_val shape: {mask_val.shape}", log_file)
    log_print(f"input dimension: {X_train.shape[1]}", log_file)
    log_print(f"output dimension: {y_train.shape[1]}", log_file)

    y_train_filled = fill_y_train_missing(y_train)

    start_time = time.time()

    model = train_model(
        X_train=X_train,
        y_train_filled=y_train_filled,
        log_file=log_file,
    )

    training_time = time.time() - start_time

    y_train_pred = pd.DataFrame(
        model.predict(X_train),
        index=X_train.index,
        columns=y_train.columns,
    )

    y_val_pred = pd.DataFrame(
        model.predict(X_val),
        index=X_val.index,
        columns=y_val.columns,
    )

    prediction_df = pd.DataFrame(
        y_val_pred,
        index=y_val.index,
        columns=y_val.columns,
    )

    save_validation_predictions(
        prediction_df=prediction_df,
        output_file=paths["prediction_output"],
        log_file=log_file,
    )

    train_metrics = evaluate_dataset(
        y_true=y_train,
        y_pred=y_train_pred,
        mask=mask_train,
    )

    val_metrics = evaluate_dataset(
        y_true=y_val,
        y_pred=y_val_pred,
        mask=mask_val,
    )

    metrics_df = pd.DataFrame([
        {
            "model": MODEL_NAME,
            "dataset": "train",
            **train_metrics,
        },
        {
            "model": MODEL_NAME,
            "dataset": "validation",
            **val_metrics,
        },
    ])

    save_model(
        model=model,
        output_file=paths["model_output"],
        log_file=log_file,
    )

    save_metrics(
        metrics_df=metrics_df,
        output_file=paths["metrics_output"],
        log_file=log_file,
    )

    log_print("=" * 70, log_file)
    log_print("KẾT QUẢ TRAIN XGBOOST", log_file)
    log_print("=" * 70, log_file)
    log_print(f"training time seconds: {training_time:.2f}", log_file)

    log_print(f"Train MAE: {train_metrics['MAE']:.6f}", log_file)
    log_print(f"Train RMSE: {train_metrics['RMSE']:.6f}", log_file)
    log_print(f"Train R2: {train_metrics['R2']:.6f}", log_file)

    log_print(f"Validation MAE: {val_metrics['MAE']:.6f}", log_file)
    log_print(f"Validation RMSE: {val_metrics['RMSE']:.6f}", log_file)
    log_print(f"Validation R2: {val_metrics['R2']:.6f}", log_file)

    log_print("=" * 70, log_file)
    log_print("ĐÃ LƯU OUTPUT", log_file)
    log_print("=" * 70, log_file)
    log_print(f"Model output path: {paths['model_output']}", log_file)
    log_print(f"Metrics output path: {paths['metrics_output']}", log_file)
    log_print(f"Validation prediction output path: {paths['prediction_output']}", log_file)
    log_print(f"Log file path: {log_file}", log_file)


if __name__ == "__main__":
    main()