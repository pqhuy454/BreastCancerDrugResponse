from pathlib import Path
from datetime import datetime
import time
import random
import copy
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


MODEL_NAME = "dnn"

HIDDEN_DIMS = [512, 256]
DROPOUT = 0.2
LEARNING_RATE = 0.0015
WEIGHT_DECAY = 0.001
BATCH_SIZE = 64

MAX_EPOCHS = 300
PATIENCE = 20

RANDOM_STATE = 42
DEVICE = "auto"
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
    figures_dir = project_root / "results" / "figures"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "log_file": logs_dir / f"train_dnn_{run_time}.log",

        "X_train": train_dir / "X_train_pca.csv",
        "y_train": train_dir / "y_train.csv",
        "mask_train": train_dir / "mask_train.csv",

        "X_val": val_dir / "X_val_pca.csv",
        "y_val": val_dir / "y_val.csv",
        "mask_val": val_dir / "mask_val.csv",

        "model_output": models_dir / "dnn_model.pkl",
        "metrics_output": metrics_dir / "dnn_model_train_val_metrics.csv",
        "prediction_output": predictions_dir / "dnn_validation_predictions.csv",
        "history_output": metrics_dir / "dnn_training_history.csv",
        "training_curve_output": figures_dir / "dnn_training_curve.png",
    }


def log_print(message: str = "", log_file: Path | None = None) -> None:
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(DEVICE)


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

    train_ok, mask_train = validate_y_mask(y_train, mask_train, "TRAIN", log_file)
    val_ok, mask_val = validate_y_mask(y_val, mask_val, "VALIDATION", log_file)

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


def prepare_tensors(
    X: pd.DataFrame,
    y: pd.DataFrame,
    mask: pd.DataFrame,
    device: torch.device,
):
    X_values = X.to_numpy(dtype=np.float32)

    y_values = y.to_numpy(dtype=np.float32)
    y_values = np.nan_to_num(y_values, nan=0.0)

    mask_values = mask.to_numpy(dtype=np.float32)

    X_tensor = torch.tensor(X_values, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_values, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask_values, dtype=torch.float32, device=device)

    return X_tensor, y_tensor, mask_tensor


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid_count = torch.sum(mask)

    if valid_count.item() == 0:
        return pred.sum() * 0.0

    squared_error = (pred - target) ** 2
    loss = torch.sum(squared_error * mask) / valid_count

    return loss


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


def build_model(input_dim: int, output_dim: int) -> DNNMultiOutputRegressor:
    return DNNMultiOutputRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )


def train_model(
    model: nn.Module,
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    mask_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    mask_val_tensor: torch.Tensor,
    device: torch.device,
    log_file: Path,
):
    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
        mask_train_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state_dict = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        total_train_batches = 0

        for X_batch, y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()

            pred = model(X_batch)
            loss = masked_mse_loss(pred, y_batch, mask_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_batches += 1

        train_loss = total_train_loss / max(total_train_batches, 1)

        model.eval()

        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = masked_mse_loss(
                val_pred,
                y_val_tensor,
                mask_val_tensor,
            ).item()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if epoch == 1 or epoch % 20 == 0:
            log_print(
                f"epoch {epoch}/{MAX_EPOCHS} | "
                f"train_loss: {train_loss:.6f} | "
                f"val_loss: {val_loss:.6f}",
                log_file,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0

            log_print(
                f"new best validation loss at epoch {epoch}: {best_val_loss:.6f}",
                log_file,
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            log_print(
                f"early stopping at epoch {epoch}",
                log_file,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_epoch, best_val_loss, history


def predict_model(
    model: nn.Module,
    X_tensor: torch.Tensor,
    index,
    columns,
) -> pd.DataFrame:
    model.eval()

    with torch.no_grad():
        pred = model(X_tensor).detach().cpu().numpy()

    return pd.DataFrame(
        pred,
        index=index,
        columns=columns,
    )


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


def save_model(
    model: nn.Module,
    output_file: Path,
    input_dim: int,
    output_dim: int,
    best_epoch: int,
    best_val_loss: float,
    log_file: Path,
) -> None:
    if output_file.exists():
        log_print(f"File model đã tồn tại, sẽ ghi đè: {output_file}", log_file)

    checkpoint = {
        "model_name": MODEL_NAME,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
    }

    torch.save(checkpoint, output_file)


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


def save_training_curve(history: list[dict], history_output: Path, figure_output: Path) -> None:
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_output, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Đường cong huấn luyện của mô hình DNN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_output, dpi=300)
    plt.close()


def main() -> None:
    paths = setup_paths()
    log_file = paths["log_file"]

    set_seed(RANDOM_STATE)
    device = get_device()

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU TRAIN DNN", log_file)
    log_print("=" * 70, log_file)

    log_print(f"PROJECT_ROOT: {paths['project_root']}", log_file)
    log_print(f"MODEL_NAME: {MODEL_NAME}", log_file)
    log_print(f"DEVICE: {device}", log_file)

    log_print(f"HIDDEN_DIMS: {HIDDEN_DIMS}", log_file)
    log_print(f"DROPOUT: {DROPOUT}", log_file)
    log_print(f"LEARNING_RATE: {LEARNING_RATE}", log_file)
    log_print(f"WEIGHT_DECAY: {WEIGHT_DECAY}", log_file)
    log_print(f"BATCH_SIZE: {BATCH_SIZE}", log_file)
    log_print(f"MAX_EPOCHS: {MAX_EPOCHS}", log_file)
    log_print(f"PATIENCE: {PATIENCE}", log_file)

    data = load_dataset(paths, log_file)

    if data is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO LỖI DỮ LIỆU INPUT.", log_file)
        return

    X_train, y_train, mask_train, X_val, y_val, mask_val = data

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    log_print("=" * 70, log_file)
    log_print("THÔNG TIN DATASET", log_file)
    log_print("=" * 70, log_file)
    log_print(f"X_train shape: {X_train.shape}", log_file)
    log_print(f"y_train shape: {y_train.shape}", log_file)
    log_print(f"mask_train shape: {mask_train.shape}", log_file)
    log_print(f"X_val shape: {X_val.shape}", log_file)
    log_print(f"y_val shape: {y_val.shape}", log_file)
    log_print(f"mask_val shape: {mask_val.shape}", log_file)
    log_print(f"input dimension: {input_dim}", log_file)
    log_print(f"output dimension: {output_dim}", log_file)

    X_train_tensor, y_train_tensor, mask_train_tensor = prepare_tensors(
        X_train,
        y_train,
        mask_train,
        device,
    )

    X_val_tensor, y_val_tensor, mask_val_tensor = prepare_tensors(
        X_val,
        y_val,
        mask_val,
        device,
    )

    model = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)

    start_time = time.time()

    model, best_epoch, best_val_loss, history = train_model(
        model=model,
        X_train_tensor=X_train_tensor,
        y_train_tensor=y_train_tensor,
        mask_train_tensor=mask_train_tensor,
        X_val_tensor=X_val_tensor,
        y_val_tensor=y_val_tensor,
        mask_val_tensor=mask_val_tensor,
        device=device,
        log_file=log_file,
    )

    training_time = time.time() - start_time

    y_train_pred = predict_model(
        model=model,
        X_tensor=X_train_tensor,
        index=X_train.index,
        columns=y_train.columns,
    )

    y_val_pred = predict_model(
        model=model,
        X_tensor=X_val_tensor,
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
        input_dim=input_dim,
        output_dim=output_dim,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        log_file=log_file,
    )

    save_metrics(
        metrics_df=metrics_df,
        output_file=paths["metrics_output"],
        log_file=log_file,
    )
    
    save_training_curve(
        history=history,
        history_output=paths["history_output"],
        figure_output=paths["training_curve_output"],
)

    log_print("=" * 70, log_file)
    log_print("KẾT QUẢ TRAIN DNN", log_file)
    log_print("=" * 70, log_file)

    log_print(f"best epoch: {best_epoch}", log_file)
    log_print(f"best validation loss: {best_val_loss:.6f}", log_file)
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
    log_print(f"Validation prediction output path: {paths['prediction_output']}", log_file,)
    log_print(f"Log file path: {log_file}", log_file)


if __name__ == "__main__":
    main()