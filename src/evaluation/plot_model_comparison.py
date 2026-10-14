
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SAVE_LOG = True

MODELS = ["ridge", "xgboost", "dnn"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    project_root = get_project_root()

    metrics_dir = project_root / "results" / "metrics"
    predictions_dir = project_root / "results" / "predictions"
    figures_dir = project_root / "results" / "figures"
    logs_dir = project_root / "logs"

    test_dir = project_root / "data" / "processed" / "test"

    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "log_file": logs_dir / f"plot_model_comparison_{run_time}.log",
        "metrics_dir": metrics_dir,
        "predictions_dir": predictions_dir,
        "figures_dir": figures_dir,
        "y_test": test_dir / "y_test.csv",
        "mask_test": test_dir / "mask_test.csv",
        "test_metrics_file": metrics_dir / "cell_line_test_metrics_all_models.csv",
    }


def log_print(message="", log_file=None):
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def load_metrics(metrics_dir: Path):
    metrics = {}

    for model in MODELS:
        file_path = metrics_dir / f"{model}_model_test_metrics.csv"

        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)

        metrics[model] = df

    return metrics


def load_prediction(predictions_dir: Path, model_name: str):
    candidates = [
        predictions_dir / f"{model_name}_test_predictions.csv",
        predictions_dir / f"{model_name}_predictions.csv",
    ]

    for file_path in candidates:
        if file_path.exists():
            return pd.read_csv(file_path, index_col=0)

    raise FileNotFoundError(f"Không tìm thấy prediction file của {model_name}")


def load_test_metrics(metrics_file: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_file)

    required_cols = [
        "model",
        "MAE",
        "RMSE",
        "R2",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Thiếu cột trong file metrics: {missing_cols}"
        )

    return df


def add_value_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=8)


def get_valid_values(y_true, y_pred, mask):
    valid_mask = mask > 0

    y_true_flat = y_true.values[valid_mask.values]
    y_pred_flat = y_pred.values[valid_mask.values]

    return y_true_flat, y_pred_flat


def plot_metrics_comparison(
    metrics_df,
    dataset_name,
    output_file,
):
    metric_names = ["MAE", "RMSE", "R2"]

    models = metrics_df["model"].tolist()

    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (_, row) in enumerate(metrics_df.iterrows()):
        values = [
            row["MAE"],
            row["RMSE"],
            row["R2"],
        ]

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=row["model"].upper(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Value")
    ax.set_title(f"{dataset_name} Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    add_value_labels(ax)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_model_comparison_test(
    metrics_df: pd.DataFrame,
    output_file: Path,
) -> None:
    metric_names = ["MAE", "RMSE", "R2"]

    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    model_order = [
        "ridge",
        "xgboost",
        "dnn",
    ]

    for i, model_name in enumerate(model_order):
        row = metrics_df[
            metrics_df["model"].str.lower() == model_name
        ]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        values = [
            row["MAE"],
            row["RMSE"],
            row["R2"],
        ]

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=model_name.upper(),
        )

        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)

    ax.set_ylabel("Metric Value")

    ax.set_title(
        "Comparison of Ridge, XGBoost and DNN on Test Set"
    )

    ax.legend()

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_error_distribution(
    y_true,
    y_pred,
    mask,
    model_name,
    output_file,
):
    yt, yp = get_valid_values(y_true, y_pred, mask)

    errors = yp - yt

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50)
    plt.axvline(0, linestyle="--")

    plt.title(
        f"Prediction Error Distribution - {model_name.upper()}"
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_scatter(
    y_true,
    y_pred,
    mask,
    model_name,
    output_file,
):
    yt, yp = get_valid_values(y_true, y_pred, mask)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        yt,
        yp,
        alpha=0.3,
        s=10,
    )

    min_val = min(yt.min(), yp.min())
    max_val = max(yt.max(), yp.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
    )

    plt.xlabel("True LN(IC50)")
    plt.ylabel("Predicted LN(IC50)")

    plt.title(
        f"True vs Predicted - {model_name.upper()}"
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main():
    paths = setup_paths()
    log_file = paths["log_file"]

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU VẼ BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH", log_file)
    log_print("=" * 70, log_file)

    y_test = pd.read_csv(paths["y_test"], index_col=0)
    mask_test = pd.read_csv(paths["mask_test"], index_col=0)

    validation_rows = []
    test_rows = []

    for model in MODELS:
        metric_file = paths["metrics_dir"] / f"{model}_model_train_val_metrics.csv"

        if metric_file.exists():
            df = pd.read_csv(metric_file)

            val_row = df[df["dataset"].str.lower() == "validation"]

            if len(val_row) > 0:
                row = val_row.iloc[0].copy()
                row["model"] = model
                validation_rows.append(row)

        test_metric_file = paths["metrics_dir"] / f"{model}_model_test_metrics.csv"

        if test_metric_file.exists():
            df = pd.read_csv(test_metric_file)

            row = df.iloc[0].copy()
            row["model"] = model
            test_rows.append(row)

    if validation_rows:
        validation_df = pd.DataFrame(validation_rows)

        plot_metrics_comparison(
            validation_df,
            "Validation",
            paths["figures_dir"] / "model_comparison_validation.png",
        )

    test_metrics_file = paths["test_metrics_file"]

    if test_metrics_file.exists():
        test_metrics_df = load_test_metrics(
            test_metrics_file
        )

        plot_model_comparison_test(
            metrics_df=test_metrics_df,
            output_file=(
                paths["figures_dir"]
                / "model_comparison_test.png"
            ),
        )

        log_print(
            "Đã lưu model_comparison_test.png",
            log_file,
        )

    for model in MODELS:
        try:
            y_pred = load_prediction(
                paths["predictions_dir"],
                model,
            )

            plot_error_distribution(
                y_test,
                y_pred,
                mask_test,
                model,
                paths["figures_dir"] / f"{model}_error_distribution.png",
            )

            plot_scatter(
                y_test,
                y_pred,
                mask_test,
                model,
                paths["figures_dir"] / f"{model}_scatter.png",
            )

        except Exception as e:
            log_print(
                f"Lỗi xử lý {model}: {e}",
                log_file,
            )

    log_print("=" * 70, log_file)
    log_print("HOÀN THÀNH", log_file)
    log_print("=" * 70, log_file)


if __name__ == "__main__":
    main()
