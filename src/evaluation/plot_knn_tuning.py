from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CẤU HÌNH
# ============================================================

SAVE_LOG = True

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]

HISTORY_FILE_TEMPLATE = "gnn_training_history_K{k}.csv"

VALIDATION_LOSS_FIGURE_NAME = "knn_tuning_validation_loss_curve.png"
TRAIN_LOSS_FIGURE_NAME = "knn_tuning_train_loss_curve.png"
BEST_VAL_LOSS_FIGURE_NAME = "knn_tuning_best_validation_loss_by_k.png"
SUMMARY_OUTPUT_NAME = "knn_tuning_summary.csv"


# ============================================================
# 2. THIẾT LẬP ĐƯỜNG DẪN
# ============================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    project_root = get_project_root()

    spatial_results_dir = project_root / "results" / "spatial"
    figures_dir = project_root / "results" / "figures" / "spatial"
    logs_dir = project_root / "logs"

    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "project_root": project_root,
        "spatial_results_dir": spatial_results_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
        "log_file": logs_dir / f"plot_knn_tuning_{run_time}.log",
        "validation_loss_figure": figures_dir / VALIDATION_LOSS_FIGURE_NAME,
        "train_loss_figure": figures_dir / TRAIN_LOSS_FIGURE_NAME,
        "best_val_loss_figure": figures_dir / BEST_VAL_LOSS_FIGURE_NAME,
        "summary_output": spatial_results_dir / SUMMARY_OUTPUT_NAME,
    }


# ============================================================
# 3. LOGGING
# ============================================================

def log_print(message: str = "", log_file: Path | None = None) -> None:
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(str(message) + "\n")


# ============================================================
# 4. ĐỌC DỮ LIỆU TUNING
# ============================================================

def get_history_file(spatial_results_dir: Path, k_value: int) -> Path:
    return spatial_results_dir / HISTORY_FILE_TEMPLATE.format(k=k_value)


def validate_history_df(history_df: pd.DataFrame, k_value: int) -> None:
    required_columns = [
        "epoch",
        "train_loss",
        "val_loss",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in history_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"File history của K={k_value} thiếu cột: {missing_columns}"
        )


def load_single_history(file_path: Path, k_value: int) -> pd.DataFrame:
    history_df = pd.read_csv(file_path)

    validate_history_df(
        history_df=history_df,
        k_value=k_value,
    )

    history_df = history_df.copy()
    history_df["K"] = int(k_value)

    history_df["epoch"] = pd.to_numeric(
        history_df["epoch"],
        errors="coerce",
    )

    history_df["train_loss"] = pd.to_numeric(
        history_df["train_loss"],
        errors="coerce",
    )

    history_df["val_loss"] = pd.to_numeric(
        history_df["val_loss"],
        errors="coerce",
    )

    history_df = history_df.dropna(
        subset=[
            "epoch",
            "train_loss",
            "val_loss",
        ]
    )

    history_df = history_df.sort_values(
        by="epoch",
        ascending=True,
    ).reset_index(drop=True)

    return history_df


def load_all_histories(
    spatial_results_dir: Path,
    log_file: Path | None = None,
) -> dict[int, pd.DataFrame]:
    histories = {}

    for k_value in K_VALUES:
        file_path = get_history_file(
            spatial_results_dir=spatial_results_dir,
            k_value=k_value,
        )

        if not file_path.exists():
            log_print(
                f"CẢNH BÁO: Không tìm thấy file {file_path}",
                log_file,
            )
            continue

        history_df = load_single_history(
            file_path=file_path,
            k_value=k_value,
        )

        histories[k_value] = history_df

        log_print(
            f"Đã đọc K={k_value}: {file_path.name}, "
            f"số epoch={len(history_df)}",
            log_file,
        )

    if len(histories) == 0:
        raise FileNotFoundError(
            "Không đọc được file history nào. "
            "Cần có các file gnn_training_history_K1.csv đến gnn_training_history_K8.csv."
        )

    return histories


# ============================================================
# 5. TẠO BẢNG TÓM TẮT
# ============================================================

def build_summary_df(histories: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for k_value, history_df in histories.items():
        best_val_idx = history_df["val_loss"].idxmin()
        best_train_idx = history_df["train_loss"].idxmin()

        best_val_row = history_df.loc[best_val_idx]
        best_train_row = history_df.loc[best_train_idx]

        row = {
            "K": int(k_value),
            "best_val_epoch": int(best_val_row["epoch"]),
            "best_val_loss": float(best_val_row["val_loss"]),
            "best_train_epoch": int(best_train_row["epoch"]),
            "best_train_loss": float(best_train_row["train_loss"]),
            "last_epoch": int(history_df["epoch"].max()),
        }

        if "elapsed_seconds" in history_df.columns:
            row["total_elapsed_seconds"] = float(
                pd.to_numeric(
                    history_df["elapsed_seconds"],
                    errors="coerce",
                ).fillna(0).sum()
            )

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        by="K",
        ascending=True,
    ).reset_index(drop=True)

    return summary_df


def save_summary(
    summary_df: pd.DataFrame,
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_file,
        index=False,
    )

    log_print(
        f"Đã lưu bảng tóm tắt tuning K: {output_file}",
        log_file,
    )


# ============================================================
# 6. VẼ ĐƯỜNG CONG LOSS THEO EPOCH
# ============================================================

def plot_loss_curves(
    histories: dict[int, pd.DataFrame],
    loss_column: str,
    y_label: str,
    title: str,
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    plt.figure(figsize=(11, 7))

    for k_value in sorted(histories.keys()):
        history_df = histories[k_value]

        plt.plot(
            history_df["epoch"],
            history_df[loss_column],
            linewidth=1.8,
            label=f"K={k_value}",
        )

        best_idx = history_df[loss_column].idxmin()
        best_epoch = history_df.loc[best_idx, "epoch"]
        best_loss = history_df.loc[best_idx, loss_column]

        plt.scatter(
            best_epoch,
            best_loss,
            s=35,
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(
        title="K",
        ncol=2,
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    plt.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    log_print(
        f"Đã lưu hình: {output_file}",
        log_file,
    )


def plot_validation_loss_curves(
    histories: dict[int, pd.DataFrame],
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    plot_loss_curves(
        histories=histories,
        loss_column="val_loss",
        y_label="Validation Loss",
        title="Ảnh hưởng của giá trị K đến Validation Loss",
        output_file=output_file,
        log_file=log_file,
    )


def plot_train_loss_curves(
    histories: dict[int, pd.DataFrame],
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    plot_loss_curves(
        histories=histories,
        loss_column="train_loss",
        y_label="Train Loss",
        title="Ảnh hưởng của giá trị K đến Train Loss",
        output_file=output_file,
        log_file=log_file,
    )


def plot_best_validation_loss_by_k(
    summary_df: pd.DataFrame,
    output_file: Path,
    log_file: Path | None = None,
) -> None:
    plot_df = summary_df.sort_values(
        by="K",
        ascending=True,
    ).reset_index(drop=True)

    best_idx = plot_df["best_val_loss"].idxmin()
    best_k = int(plot_df.loc[best_idx, "K"])
    best_loss = float(plot_df.loc[best_idx, "best_val_loss"])

    plt.figure(figsize=(8, 5))

    plt.plot(
        plot_df["K"],
        plot_df["best_val_loss"],
        marker="o",
        linewidth=2,
    )

    plt.scatter(
        best_k,
        best_loss,
        s=120,
        marker="*",
        label=f"Best K = {best_k}",
    )

    for _, row in plot_df.iterrows():
        plt.text(
            row["K"],
            row["best_val_loss"] + 0.002,
            f"{row['best_val_loss']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(plot_df["K"])
    plt.xlabel("K")
    plt.ylabel("Best Validation Loss")
    plt.title("Ảnh hưởng của giá trị K đến Best Validation Loss")

    plt.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    plt.legend()
    plt.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    log_print(
        f"Đã lưu hình: {output_file}",
        log_file,
    )


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    paths = setup_paths()
    log_file = paths["log_file"]

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU VẼ HÌNH TUNING K CHO GNN", log_file)
    log_print("=" * 70, log_file)

    log_print(f"PROJECT_ROOT: {paths['project_root']}", log_file)
    log_print(f"SPATIAL_RESULTS_DIR: {paths['spatial_results_dir']}", log_file)
    log_print(f"FIGURES_DIR: {paths['figures_dir']}", log_file)
    log_print("", log_file)

    histories = load_all_histories(
        spatial_results_dir=paths["spatial_results_dir"],
        log_file=log_file,
    )

    summary_df = build_summary_df(
        histories=histories,
    )

    save_summary(
        summary_df=summary_df,
        output_file=paths["summary_output"],
        log_file=log_file,
    )

    plot_validation_loss_curves(
        histories=histories,
        output_file=paths["validation_loss_figure"],
        log_file=log_file,
    )

    plot_train_loss_curves(
        histories=histories,
        output_file=paths["train_loss_figure"],
        log_file=log_file,
    )

    plot_best_validation_loss_by_k(
        summary_df=summary_df,
        output_file=paths["best_val_loss_figure"],
        log_file=log_file,
    )

    log_print("", log_file)
    log_print("BẢNG TÓM TẮT BEST LOSS THEO K", log_file)
    log_print(summary_df.to_string(index=False), log_file)

    log_print("", log_file)
    log_print("=" * 70, log_file)
    log_print("HOÀN THÀNH VẼ HÌNH TUNING K", log_file)
    log_print("=" * 70, log_file)


if __name__ == "__main__":
    main()
