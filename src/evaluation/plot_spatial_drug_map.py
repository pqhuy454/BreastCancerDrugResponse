# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_NAME = "whole_transcriptome_analysis"
DRUG_NAME = "Daporinad"

BASELINE_PREDICTION_FILE = "predicted_ic50_mean_baseline.csv"
REFINED_PREDICTION_FILE = "refined_ic50_gnn.csv"
SPOT_POSITIONS_FILE = "spot_positions.csv"

OUTPUT_DIR_NAME = "results/figures/spatial_maps"
FIGURE_DPI = 300
POINT_SIZE = 18
FIGURE_SIZE_SINGLE = (9.5, 7.5)
FIGURE_SIZE_COMPARISON = (16, 7)
COLOR_MAP = "viridis"
SAVE_LOG = True


# ============================================================
# 2. PATH SETUP AND LOGGING
# ============================================================

def get_project_root() -> Path:
    """
    Tự động xác định thư mục gốc project.

    File dự kiến nằm tại:
        src/visualization/plot_spatial_drug_map.py

    PROJECT_ROOT:
        parents[2]
    """
    return Path(__file__).resolve().parents[2]


def safe_filename(text: str) -> str:
    """
    Chuẩn hóa tên thuốc hoặc dataset thành tên file an toàn.
    """
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def setup_paths() -> dict:
    """
    Thiết lập toàn bộ đường dẫn input/output/log.
    """
    project_root = get_project_root()

    dataset_dir = project_root / "data" / "spatial_processed" / DATASET_NAME
    output_dir = project_root / OUTPUT_DIR_NAME / DATASET_NAME
    logs_dir = project_root / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    drug_safe = safe_filename(DRUG_NAME)

    return {
        "project_root": project_root,
        "dataset_dir": dataset_dir,
        "positions_file": dataset_dir / SPOT_POSITIONS_FILE,
        "baseline_file": dataset_dir / BASELINE_PREDICTION_FILE,
        "refined_file": dataset_dir / REFINED_PREDICTION_FILE,
        "output_dir": output_dir,
        "baseline_output": output_dir / f"baseline_{drug_safe}.png",
        "refined_output": output_dir / f"refined_{drug_safe}.png",
        "comparison_output": output_dir / f"comparison_{drug_safe}.png",
        "log_file": logs_dir / f"plot_spatial_drug_map_{DATASET_NAME}_{drug_safe}_{run_time}.log",
    }


def log_print(message: str = "", log_file: Path | None = None) -> None:
    """
    Vừa in ra terminal, vừa ghi log file.
    """
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

def check_required_files(paths: dict, log_file: Path) -> bool:
    """
    Kiểm tra các file input bắt buộc.
    """
    required_files = {
        "spot positions": paths["positions_file"],
        "baseline prediction": paths["baseline_file"],
        "refined prediction": paths["refined_file"],
    }

    missing_files = []

    for label, file_path in required_files.items():
        log_print(f"{label}: {file_path}", log_file)

        if not file_path.exists():
            missing_files.append((label, file_path))

    if missing_files:
        log_print("=" * 70, log_file)
        log_print("LỖI: THIẾU FILE INPUT", log_file)
        log_print("=" * 70, log_file)

        for label, file_path in missing_files:
            log_print(f"- Thiếu {label}: {file_path}", log_file)

        return False

    return True


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_prediction_matrix(file_path: Path, drug_name: str, label: str, log_file: Path) -> pd.DataFrame:
    """
    Đọc ma trận dự đoán dạng spot × drug.
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
    except Exception as e:
        raise RuntimeError(
            f"Lỗi đọc {label}: {file_path.name} | {type(e).__name__}: {e}"
        )

    df.index = df.index.astype(str).str.strip()
    df.index.name = None
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")

    if drug_name not in df.columns:
        available_examples = list(df.columns[:10])
        raise ValueError(
            f"Không tìm thấy thuốc '{drug_name}' trong {label}. "
            f"Một số thuốc đầu tiên: {available_examples}"
        )

    log_print(f"Đọc {label} thành công: {df.shape}", log_file)
    log_print(f"Số spot trong {label}: {df.shape[0]}", log_file)
    log_print(f"Số thuốc trong {label}: {df.shape[1]}", log_file)

    return df


def detect_position_columns(positions_df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Tự động phát hiện cột spot_id, x, y trong spot_positions.csv.
    """
    columns = list(positions_df.columns)
    lower_map = {str(col).lower(): col for col in columns}

    spot_candidates = [
        "spot_id",
        "barcode",
        "barcodes",
        "cell_line",
        "sample_id",
        "unnamed: 0",
    ]

    x_candidates = [
        "x",
        "pxl_col_in_fullres",
        "imagecol",
        "array_col",
        "col",
    ]

    y_candidates = [
        "y",
        "pxl_row_in_fullres",
        "imagerow",
        "array_row",
        "row",
    ]

    spot_col = None
    x_col = None
    y_col = None

    for candidate in spot_candidates:
        if candidate in lower_map:
            spot_col = lower_map[candidate]
            break

    for candidate in x_candidates:
        if candidate in lower_map:
            x_col = lower_map[candidate]
            break

    for candidate in y_candidates:
        if candidate in lower_map:
            y_col = lower_map[candidate]
            break

    if spot_col is None:
        spot_col = positions_df.columns[0]

    if x_col is None or y_col is None:
        numeric_cols = []
        for col in columns:
            if col == spot_col:
                continue
            converted = pd.to_numeric(positions_df[col], errors="coerce")
            if converted.notna().sum() > 0:
                numeric_cols.append(col)

        if len(numeric_cols) < 2:
            raise ValueError(
                "Không phát hiện được cột tọa độ x/y trong spot_positions.csv."
            )

        if x_col is None:
            x_col = numeric_cols[-1]
        if y_col is None:
            y_col = numeric_cols[-2]

    return spot_col, x_col, y_col


def load_spot_positions(file_path: Path, log_file: Path) -> pd.DataFrame:
    """
    Đọc spot_positions.csv và chuẩn hóa thành spot_id, x, y.
    """
    try:
        positions_df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(
            f"Lỗi đọc spot positions: {file_path.name} | {type(e).__name__}: {e}"
        )

    spot_col, x_col, y_col = detect_position_columns(positions_df)

    output_df = positions_df[[spot_col, x_col, y_col]].copy()
    output_df = output_df.rename(
        columns={
            spot_col: "spot_id",
            x_col: "x",
            y_col: "y",
        }
    )

    output_df["spot_id"] = output_df["spot_id"].astype(str).str.strip()
    output_df["x"] = pd.to_numeric(output_df["x"], errors="coerce")
    output_df["y"] = pd.to_numeric(output_df["y"], errors="coerce")

    before_drop = len(output_df)
    output_df = output_df.dropna(subset=["spot_id", "x", "y"]).copy()
    after_drop = len(output_df)

    if after_drop < before_drop:
        log_print(
            f"CẢNH BÁO: Loại {before_drop - after_drop} dòng vị trí thiếu spot/x/y.",
            log_file,
        )

    output_df = output_df.drop_duplicates(subset=["spot_id"], keep="first")
    output_df = output_df.reset_index(drop=True)
    output_df.index.name = None

    log_print(f"Đọc spot_positions thành công: {output_df.shape}", log_file)
    log_print(f"Cột spot sử dụng: {spot_col}", log_file)
    log_print(f"Cột x sử dụng: {x_col}", log_file)
    log_print(f"Cột y sử dụng: {y_col}", log_file)

    return output_df


# ============================================================
# 5. PREPARE PLOT DATA
# ============================================================

def prepare_plot_dataframe(
    prediction_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    drug_name: str,
    value_col: str,
    log_file: Path,
) -> pd.DataFrame:
    """
    Ghép prediction của một thuốc với tọa độ spot.

    Lưu ý:
    Không dùng merge trực tiếp để tránh lỗi pandas khi "spot_id" vừa là
    index name vừa là column label. Cách xử lý này tham khảo logic trong
    app.py: set_index("spot_id") rồi align theo prediction index.
    """
    if drug_name not in prediction_df.columns:
        raise ValueError(f"Không tìm thấy thuốc '{drug_name}' trong prediction_df.")

    if "spot_id" not in positions_df.columns:
        raise ValueError("positions_df thiếu cột spot_id.")

    prediction_spot_ids = pd.Index(
        prediction_df.index.astype(str).str.strip(),
        name=None,
    )

    prediction_values = pd.to_numeric(
        prediction_df[drug_name],
        errors="coerce",
    ).to_numpy(dtype=float)

    positions_clean = positions_df[["spot_id", "x", "y"]].copy()
    positions_clean = positions_clean.reset_index(drop=True)
    positions_clean.index.name = None
    positions_clean["spot_id"] = positions_clean["spot_id"].astype(str).str.strip()
    positions_clean["x"] = pd.to_numeric(positions_clean["x"], errors="coerce")
    positions_clean["y"] = pd.to_numeric(positions_clean["y"], errors="coerce")

    duplicate_spots = int(positions_clean["spot_id"].duplicated().sum())
    if duplicate_spots > 0:
        log_print(
            f"CẢNH BÁO: Có {duplicate_spots} spot_id trùng trong positions. Giữ dòng đầu tiên.",
            log_file,
        )
        positions_clean = positions_clean.drop_duplicates(
            subset=["spot_id"],
            keep="first",
        )

    positions_indexed = positions_clean.set_index("spot_id", drop=True)
    positions_indexed.index = positions_indexed.index.astype(str).str.strip()
    positions_indexed.index.name = None

    aligned_positions = positions_indexed.reindex(prediction_spot_ids)

    plot_df = pd.DataFrame(
        {
            "spot_id": prediction_spot_ids.to_numpy(),
            "x": aligned_positions["x"].to_numpy(dtype=float),
            "y": aligned_positions["y"].to_numpy(dtype=float),
            value_col: prediction_values,
        }
    )

    before_drop = len(plot_df)
    plot_df = plot_df.dropna(subset=["spot_id", "x", "y", value_col]).copy()
    after_drop = len(plot_df)

    if after_drop < before_drop:
        log_print(
            f"CẢNH BÁO: Loại {before_drop - after_drop} spot do thiếu tọa độ hoặc prediction.",
            log_file,
        )

    log_print(f"Số spot dùng để vẽ {value_col}: {plot_df.shape[0]}", log_file)

    if plot_df.empty:
        raise ValueError(f"Không còn spot hợp lệ để vẽ {value_col}.")

    return plot_df


def get_shared_color_range(*dataframes: pd.DataFrame, value_cols: list[str]) -> tuple[float, float]:
    """
    Lấy vmin/vmax chung cho baseline và refined để so sánh trực quan.
    """
    values = []

    for df, col in zip(dataframes, value_cols):
        values.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))

    all_values = np.concatenate(values)
    all_values = all_values[np.isfinite(all_values)]

    if len(all_values) == 0:
        raise ValueError("Không có giá trị hợp lệ để xác định thang màu.")

    return float(np.nanmin(all_values)), float(np.nanmax(all_values))


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_single_spatial_map(
    plot_df: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    output_file: Path,
    vmin: float,
    vmax: float,
    log_file: Path,
) -> None:
    """
    Vẽ một bản đồ không gian và lưu PNG.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE)

    scatter = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        c=plot_df[value_col],
        s=POINT_SIZE,
        cmap=COLOR_MAP,
        vmin=vmin,
        vmax=vmax,
    )

    ax.invert_yaxis()
    ax.set_xlabel("x", fontsize=14, fontweight="bold")
    ax.set_ylabel("y", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=12)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.86)
    colorbar.set_label(colorbar_label, fontsize=13)
    colorbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    log_print(f"Đã lưu ảnh: {output_file}", log_file)


def plot_comparison_map(
    baseline_plot_df: pd.DataFrame,
    refined_plot_df: pd.DataFrame,
    baseline_col: str,
    refined_col: str,
    drug_name: str,
    output_file: Path,
    vmin: float,
    vmax: float,
    log_file: Path,
) -> None:
    """
    Vẽ ảnh so sánh trước và sau tinh chỉnh cạnh nhau.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_COMPARISON)

    plot_items = [
        (
            axes[0],
            baseline_plot_df,
            baseline_col,
            f"Trước tinh chỉnh: {drug_name}",
        ),
        (
            axes[1],
            refined_plot_df,
            refined_col,
            f"Sau tinh chỉnh GNN: {drug_name}",
        ),
    ]

    scatter = None

    for ax, df, value_col, title in plot_items:
        scatter = ax.scatter(
            df["x"],
            df["y"],
            c=df[value_col],
            s=POINT_SIZE,
            cmap=COLOR_MAP,
            vmin=vmin,
            vmax=vmax,
        )

        ax.invert_yaxis()
        ax.set_xlabel("x", fontsize=13, fontweight="bold")
        ax.set_ylabel("y", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.tick_params(axis="both", labelsize=11)

    colorbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.82)
    colorbar.set_label("LN_IC50", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)

    fig.suptitle(
        f"So sánh bản đồ không gian của thuốc: {drug_name}",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    fig.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    log_print(f"Đã lưu ảnh so sánh: {output_file}", log_file)


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    paths = setup_paths()
    log_file = paths["log_file"]

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU VẼ SPATIAL DRUG MAP", log_file)
    log_print("=" * 70, log_file)
    log_print(f"PROJECT_ROOT: {paths['project_root']}", log_file)
    log_print(f"DATASET_NAME: {DATASET_NAME}", log_file)
    log_print(f"DRUG_NAME: {DRUG_NAME}", log_file)
    log_print(f"Dataset dir: {paths['dataset_dir']}", log_file)

    if not check_required_files(paths, log_file):
        log_print("CHƯƠNG TRÌNH DỪNG DO THIẾU FILE INPUT.", log_file)
        return

    try:
        positions_df = load_spot_positions(
            file_path=paths["positions_file"],
            log_file=log_file,
        )

        baseline_df = load_prediction_matrix(
            file_path=paths["baseline_file"],
            drug_name=DRUG_NAME,
            label="baseline prediction trước GNN",
            log_file=log_file,
        )

        refined_df = load_prediction_matrix(
            file_path=paths["refined_file"],
            drug_name=DRUG_NAME,
            label="refined prediction sau GNN",
            log_file=log_file,
        )

        baseline_plot_df = prepare_plot_dataframe(
            prediction_df=baseline_df,
            positions_df=positions_df,
            drug_name=DRUG_NAME,
            value_col="baseline_LN_IC50",
            log_file=log_file,
        )

        refined_plot_df = prepare_plot_dataframe(
            prediction_df=refined_df,
            positions_df=positions_df,
            drug_name=DRUG_NAME,
            value_col="refined_LN_IC50",
            log_file=log_file,
        )

        vmin, vmax = get_shared_color_range(
            baseline_plot_df,
            refined_plot_df,
            value_cols=["baseline_LN_IC50", "refined_LN_IC50"],
        )

        log_print(f"Color range chung: vmin={vmin:.6f}, vmax={vmax:.6f}", log_file)

        plot_single_spatial_map(
            plot_df=baseline_plot_df,
            value_col="baseline_LN_IC50",
            title=f"Bản đồ không gian trước tinh chỉnh: {DRUG_NAME}",
            colorbar_label="Baseline LN_IC50",
            output_file=paths["baseline_output"],
            vmin=vmin,
            vmax=vmax,
            log_file=log_file,
        )

        plot_single_spatial_map(
            plot_df=refined_plot_df,
            value_col="refined_LN_IC50",
            title=f"Bản đồ không gian sau tinh chỉnh: {DRUG_NAME}",
            colorbar_label="Refined LN_IC50",
            output_file=paths["refined_output"],
            vmin=vmin,
            vmax=vmax,
            log_file=log_file,
        )

        plot_comparison_map(
            baseline_plot_df=baseline_plot_df,
            refined_plot_df=refined_plot_df,
            baseline_col="baseline_LN_IC50",
            refined_col="refined_LN_IC50",
            drug_name=DRUG_NAME,
            output_file=paths["comparison_output"],
            vmin=vmin,
            vmax=vmax,
            log_file=log_file,
        )

    except Exception as e:
        log_print("=" * 70, log_file)
        log_print("LỖI KHI VẼ SPATIAL DRUG MAP", log_file)
        log_print("=" * 70, log_file)
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)
        return

    log_print("=" * 70, log_file)
    log_print("HOÀN THÀNH VẼ SPATIAL DRUG MAP", log_file)
    log_print("=" * 70, log_file)
    log_print(f"Baseline image: {paths['baseline_output']}", log_file)
    log_print(f"Refined image: {paths['refined_output']}", log_file)
    log_print(f"Comparison image: {paths['comparison_output']}", log_file)
    log_print(f"Log file: {log_file}", log_file)


if __name__ == "__main__":
    main()
