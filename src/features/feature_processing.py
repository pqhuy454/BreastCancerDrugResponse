from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CẤU HÌNH Ở ĐẦU FILE
# ==========================================================

TOP_N_GENES = 20000

SAVE_TOP_GENE_FILES = False
SAVE_SCALED_FILES = True
SAVE_SCALER = True

OVERWRITE_OUTPUTS = True
SAVE_LOG = True

FILL_MISSING = True
FILL_METHOD = "median"

RANDOM_STATE = 42


# ==========================================================
# 2. PROJECT ROOT + PATHS
# ==========================================================

def get_project_root() -> Path:
    """
    Tự động xác định PROJECT_ROOT.

    File hiện tại:
        src/features/feature_processing.py

    PROJECT_ROOT:
        parents[2]
    """
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    """
    Thiết lập toàn bộ đường dẫn.
    Không dùng hard-code đường dẫn tuyệt đối.
    """
    project_root = get_project_root()

    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"

    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"

    logs_dir = project_root / "logs"
    models_dir = project_root / "models"

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = logs_dir / f"feature_processing_{run_time}.log"

    return {
        "project_root": project_root,
        "processed_dir": processed_dir,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "test_dir": test_dir,
        "logs_dir": logs_dir,
        "models_dir": models_dir,
        "log_file": log_file,

        # INPUT
        "X_train_input": train_dir / "X_train.csv",
        "X_val_input": val_dir / "X_val.csv",
        "X_test_input": test_dir / "X_test.csv",

        # TOP GENE OUTPUT
        "X_train_top_genes": train_dir / "X_train_top_genes.csv",
        "X_val_top_genes": val_dir / "X_val_top_genes.csv",
        "X_test_top_genes": test_dir / "X_test_top_genes.csv",

        # SCALED OUTPUT
        "X_train_scaled": train_dir / "X_train_scaled.csv",
        "X_val_scaled": val_dir / "X_val_scaled.csv",
        "X_test_scaled": test_dir / "X_test_scaled.csv",

        # OTHER OUTPUT
        "selected_genes": processed_dir / "selected_genes.txt",
        "top_gene_variance": processed_dir / "top_gene_variance.csv",
        "summary": processed_dir / "feature_processing_summary.csv",

        # MODEL
        "scaler_file": models_dir / "scaler.pkl",
    }


# ==========================================================
# 3. LOG
# ==========================================================

def log_print(message: str = "", log_file: Path | None = None) -> None:
    """
    Vừa print terminal vừa ghi log file.
    """
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


# ==========================================================
# 4. CONFIG
# ==========================================================

def validate_config(total_genes: int, log_file: Path) -> dict:
    """
    Validate và chuẩn hóa config.
    """
    top_n_genes = TOP_N_GENES

    if top_n_genes <= 0:
        log_print(
            "CẢNH BÁO: TOP_N_GENES <= 0. "
            "Sẽ lấy toàn bộ gene.",
            log_file,
        )
        top_n_genes = total_genes

    if top_n_genes > total_genes:
        log_print(
            f"CẢNH BÁO: TOP_N_GENES={top_n_genes} "
            f"lớn hơn tổng số gene={total_genes}. "
            "Sẽ lấy toàn bộ gene.",
            log_file,
        )
        top_n_genes = total_genes

    fill_method = FILL_METHOD

    if fill_method.lower() != "median":
        log_print(
            "CẢNH BÁO: Chỉ hỗ trợ FILL_METHOD='median'. "
            "Sẽ dùng median.",
            log_file,
        )
        fill_method = "median"

    return {
        "top_n_genes": top_n_genes,
        "fill_method": fill_method,
    }


def print_config(config: dict, log_file: Path) -> None:
    """
    In cấu hình đang dùng.
    """
    log_print("=" * 70, log_file)
    log_print("CẤU HÌNH FEATURE PROCESSING", log_file)
    log_print("=" * 70, log_file)

    log_print(f"TOP_N_GENES: {TOP_N_GENES}", log_file)
    log_print(f"TOP_N_GENES sau xử lý: {config['top_n_genes']}", log_file)

    log_print(f"FILL_MISSING: {FILL_MISSING}", log_file)
    log_print(f"FILL_METHOD: {config['fill_method']}", log_file)

    log_print(f"SAVE_TOP_GENE_FILES: {SAVE_TOP_GENE_FILES}", log_file)
    log_print(f"SAVE_SCALED_FILES: {SAVE_SCALED_FILES}", log_file)
    log_print(f"SAVE_SCALER: {SAVE_SCALER}", log_file)

    log_print(f"OVERWRITE_OUTPUTS: {OVERWRITE_OUTPUTS}", log_file)
    log_print(f"SAVE_LOG: {SAVE_LOG}", log_file)


# ==========================================================
# 5. FILE CHECK
# ==========================================================

def check_required_files(paths: dict, log_file: Path) -> bool:
    """
    Kiểm tra file input bắt buộc.
    """
    log_print("=" * 70, log_file)
    log_print("KIỂM TRA FILE INPUT", log_file)
    log_print("=" * 70, log_file)

    required_files = [
        paths["X_train_input"],
        paths["X_val_input"],
        paths["X_test_input"],
    ]

    all_ok = True

    for file_path in required_files:
        log_print(f"Kiểm tra: {file_path}", log_file)

        if file_path.exists():
            log_print("Trạng thái: tồn tại", log_file)
        else:
            log_print(f"LỖI: Không tìm thấy file: {file_path.name}", log_file)
            all_ok = False

    return all_ok


# ==========================================================
# 6. LOAD DATA
# ==========================================================

def read_feature_matrix(file_path: Path, log_file: Path) -> pd.DataFrame | None:
    """
    Đọc feature matrix bằng pandas.read_csv(index_col=0).
    """
    try:
        df = pd.read_csv(file_path, index_col=0)

        log_print(f"Đọc thành công: {file_path.name}", log_file)
        log_print(f"Shape: {df.shape}", log_file)

        return df

    except Exception as e:
        log_print("=" * 70, log_file)
        log_print("LỖI ĐỌC FILE FEATURE MATRIX", log_file)
        log_print("=" * 70, log_file)
        log_print(f"Tên file: {file_path.name}", log_file)
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)

        return None


def standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa index và columns.
    """
    df = df.copy()

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    return df


# ==========================================================
# 7. VALIDATE GENE COLUMNS
# ==========================================================

def validate_gene_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    log_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Kiểm tra train/val/test có cùng gene hay không.
    """
    log_print("=" * 70, log_file)
    log_print("VALIDATE GENE COLUMNS", log_file)
    log_print("=" * 70, log_file)

    train_genes = list(X_train.columns)

    if len(train_genes) == 0:
        log_print("LỖI: X_train không có gene nào.", log_file)
        return None

    val_missing = set(train_genes) - set(X_val.columns)
    test_missing = set(train_genes) - set(X_test.columns)

    if len(val_missing) > 0:
        log_print(
            f"LỖI: Validation thiếu {len(val_missing)} gene so với train.",
            log_file,
        )
        return None

    if len(test_missing) > 0:
        log_print(
            f"LỖI: Test thiếu {len(test_missing)} gene so với train.",
            log_file,
        )
        return None

    X_val = X_val.loc[:, train_genes]
    X_test = X_test.loc[:, train_genes]

    log_print("Validation/Test đã được sắp xếp cùng thứ tự gene với train.", log_file)

    return X_train, X_val, X_test


# ==========================================================
# 8. NUMERIC + MISSING
# ==========================================================

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép toàn bộ dữ liệu sang numeric.
    """
    return df.apply(lambda col: pd.to_numeric(col, errors="coerce"))


def count_missing(df: pd.DataFrame) -> int:
    """
    Đếm số missing.
    """
    return int(df.isna().sum().sum())


def fill_missing_using_train_median(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    log_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Fill missing bằng median tính trên train.
    Không dùng validation/test để fit median.
    """
    log_print("=" * 70, log_file)
    log_print("XỬ LÝ MISSING", log_file)
    log_print("=" * 70, log_file)

    train_missing_before = count_missing(X_train)
    val_missing_before = count_missing(X_val)
    test_missing_before = count_missing(X_test)

    log_print(f"Train missing trước fill: {train_missing_before}", log_file)
    log_print(f"Validation missing trước fill: {val_missing_before}", log_file)
    log_print(f"Test missing trước fill: {test_missing_before}", log_file)

    if FILL_MISSING:
        train_median = X_train.median(axis=0, skipna=True)

        X_train = X_train.fillna(train_median)
        X_val = X_val.fillna(train_median)
        X_test = X_test.fillna(train_median)

        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

    train_missing_after = count_missing(X_train)
    val_missing_after = count_missing(X_val)
    test_missing_after = count_missing(X_test)

    log_print(f"Train missing sau fill: {train_missing_after}", log_file)
    log_print(f"Validation missing sau fill: {val_missing_after}", log_file)
    log_print(f"Test missing sau fill: {test_missing_after}", log_file)

    stats = {
        "train_missing_before": train_missing_before,
        "val_missing_before": val_missing_before,
        "test_missing_before": test_missing_before,

        "train_missing_after_fill": train_missing_after,
        "val_missing_after_fill": val_missing_after,
        "test_missing_after_fill": test_missing_after,
    }

    return X_train, X_val, X_test, stats


# ==========================================================
# 9. SELECT TOP GENES
# ==========================================================

def calculate_train_variance(X_train: pd.DataFrame) -> pd.Series:
    """
    Tính variance trên train.
    """
    return X_train.var(axis=0)


def select_top_genes(
    X_train: pd.DataFrame,
    config: dict,
    log_file: Path,
) -> tuple[list[str], pd.DataFrame, dict]:
    """
    Chọn top gene variance cao nhất.
    """
    log_print("=" * 70, log_file)
    log_print("SELECT TOP GENES", log_file)
    log_print("=" * 70, log_file)

    variance_series = calculate_train_variance(X_train)

    variance_df = pd.DataFrame({
        "gene": variance_series.index,
        "variance": variance_series.values,
    })

    variance_df = variance_df.sort_values(
        by="variance",
        ascending=False,
    ).reset_index(drop=True)

    total_genes = len(variance_df)

    selected_variance_df = variance_df.head(
        config["top_n_genes"]
    ).copy()

    selected_genes = selected_variance_df["gene"].tolist()

    removed_genes = total_genes - len(selected_genes)

    zero_variance_total = int((variance_df["variance"] == 0).sum())
    zero_variance_selected = int((selected_variance_df["variance"] == 0).sum())

    max_variance = float(variance_df["variance"].max())
    min_selected_variance = float(selected_variance_df["variance"].min())

    log_print(f"Tổng số gene ban đầu: {total_genes}", log_file)
    log_print(f"Số gene được chọn: {len(selected_genes)}", log_file)
    log_print(f"Số gene bị loại: {removed_genes}", log_file)

    log_print(f"Variance lớn nhất: {max_variance}", log_file)
    log_print(
        f"Variance nhỏ nhất trong top gene: {min_selected_variance}",
        log_file,
    )

    log_print(
        f"Số gene variance = 0 toàn bộ train: {zero_variance_total}",
        log_file,
    )

    log_print(
        f"Số gene variance = 0 trong top gene: {zero_variance_selected}",
        log_file,
    )

    stats = {
        "selected_genes": len(selected_genes),
        "removed_genes": removed_genes,
        "max_variance": max_variance,
        "min_selected_variance": min_selected_variance,
        "zero_variance_genes_total": zero_variance_total,
        "zero_variance_genes_selected": zero_variance_selected,
    }

    return selected_genes, selected_variance_df, stats


def filter_selected_genes(
    df: pd.DataFrame,
    selected_genes: list[str],
) -> pd.DataFrame:
    """
    Lọc dataset theo selected genes.
    """
    return df.loc[:, selected_genes].copy()


# ==========================================================
# 10. STANDARD SCALER
# ==========================================================

def fit_scaler_on_train(
    X_train_top_genes: pd.DataFrame,
) -> StandardScaler:
    """
    Fit scaler chỉ trên train.
    """
    scaler = StandardScaler()

    scaler.fit(X_train_top_genes)

    return scaler


def transform_with_scaler(
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform dataset bằng scaler đã fit trên train.
    """
    scaled_array = scaler.transform(df)

    scaled_df = pd.DataFrame(
        scaled_array,
        index=df.index,
        columns=df.columns,
    )

    return scaled_df


def check_scaled_matrix(
    df: pd.DataFrame,
    dataset_name: str,
    log_file: Path,
) -> dict:
    """
    Kiểm tra missing/inf sau scale.
    """
    log_print(f"Kiểm tra scaled matrix: {dataset_name}", log_file)

    inf_count = int(np.isinf(df.to_numpy()).sum())

    if inf_count > 0:
        log_print(
            f"CẢNH BÁO: {dataset_name} có {inf_count} giá trị inf.",
            log_file,
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    missing_count = count_missing(df)

    return {
        "missing": missing_count,
        "inf": inf_count,
    }


# ==========================================================
# 11. SAVE
# ==========================================================

def save_dataframe(
    df: pd.DataFrame,
    output_file: Path,
    log_file: Path,
) -> bool:
    """
    Lưu DataFrame.
    """
    if output_file.exists():
        if OVERWRITE_OUTPUTS:
            log_print(f"Ghi đè file: {output_file}", log_file)
        else:
            log_print(
                f"CẢNH BÁO: File đã tồn tại và OVERWRITE_OUTPUTS=False: {output_file}",
                log_file,
            )
            return False

    df.to_csv(output_file)

    return True


def save_gene_lists(
    selected_genes: list[str],
    variance_df: pd.DataFrame,
    paths: dict,
    log_file: Path,
) -> None:
    """
    Lưu selected genes và variance.
    """
    with open(paths["selected_genes"], "w", encoding="utf-8") as f:
        for gene in selected_genes:
            f.write(gene + "\n")

    variance_df.to_csv(paths["top_gene_variance"], index=False)

    log_print("Đã lưu selected_genes.txt", log_file)
    log_print("Đã lưu top_gene_variance.csv", log_file)


def save_outputs(
    paths: dict,
    X_train_top: pd.DataFrame,
    X_val_top: pd.DataFrame,
    X_test_top: pd.DataFrame,
    X_train_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    scaler: StandardScaler,
    log_file: Path,
) -> None:
    """
    Lưu toàn bộ output.
    """
    if SAVE_TOP_GENE_FILES:
        save_dataframe(X_train_top, paths["X_train_top_genes"], log_file)
        save_dataframe(X_val_top, paths["X_val_top_genes"], log_file)
        save_dataframe(X_test_top, paths["X_test_top_genes"], log_file)

    if SAVE_SCALED_FILES:
        save_dataframe(X_train_scaled, paths["X_train_scaled"], log_file)
        save_dataframe(X_val_scaled, paths["X_val_scaled"], log_file)
        save_dataframe(X_test_scaled, paths["X_test_scaled"], log_file)

    if SAVE_SCALER:
        joblib.dump(scaler, paths["scaler_file"])

        log_print(f"Đã lưu scaler: {paths['scaler_file']}", log_file)


def save_summary(
    summary_data: dict,
    output_file: Path,
    log_file: Path,
) -> None:
    """
    Lưu summary CSV.
    """
    summary_df = pd.DataFrame([summary_data])

    summary_df.to_csv(output_file, index=False)

    log_print(f"Đã lưu summary: {output_file}", log_file)


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    paths = setup_paths()
    log_file = paths["log_file"]

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU FEATURE PROCESSING", log_file)
    log_print("=" * 70, log_file)

    if not check_required_files(paths, log_file):
        log_print("CHƯƠNG TRÌNH DỪNG DO THIẾU FILE INPUT.", log_file)
        return

    log_print("=" * 70, log_file)
    log_print("LOAD DATA", log_file)
    log_print("=" * 70, log_file)

    X_train = read_feature_matrix(paths["X_train_input"], log_file)
    X_val = read_feature_matrix(paths["X_val_input"], log_file)
    X_test = read_feature_matrix(paths["X_test_input"], log_file)

    if X_train is None or X_val is None or X_test is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO LỖI LOAD DATA.", log_file)
        return

    X_train = standardize_index(X_train)
    X_val = standardize_index(X_val)
    X_test = standardize_index(X_test)

    log_print(f"Số mẫu train: {X_train.shape[0]}", log_file)
    log_print(f"Số mẫu validation: {X_val.shape[0]}", log_file)
    log_print(f"Số mẫu test: {X_test.shape[0]}", log_file)

    log_print(f"Số gene ban đầu: {X_train.shape[1]}", log_file)

    config = validate_config(
        total_genes=X_train.shape[1],
        log_file=log_file,
    )

    print_config(config, log_file)

    validated = validate_gene_columns(
        X_train,
        X_val,
        X_test,
        log_file,
    )

    if validated is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO GENE COLUMNS KHÔNG HỢP LỆ.", log_file)
        return

    X_train, X_val, X_test = validated

    X_train = convert_to_numeric(X_train)
    X_val = convert_to_numeric(X_val)
    X_test = convert_to_numeric(X_test)

    X_train, X_val, X_test, missing_stats = fill_missing_using_train_median(
        X_train,
        X_val,
        X_test,
        log_file,
    )

    selected_genes, variance_df, variance_stats = select_top_genes(
        X_train,
        config,
        log_file,
    )

    log_print("=" * 70, log_file)
    log_print("FILTER THEO SELECTED GENES", log_file)
    log_print("=" * 70, log_file)

    X_train_top = filter_selected_genes(X_train, selected_genes)
    X_val_top = filter_selected_genes(X_val, selected_genes)
    X_test_top = filter_selected_genes(X_test, selected_genes)

    log_print(f"X_train_top shape: {X_train_top.shape}", log_file)
    log_print(f"X_val_top shape: {X_val_top.shape}", log_file)
    log_print(f"X_test_top shape: {X_test_top.shape}", log_file)

    log_print("=" * 70, log_file)
    log_print("STANDARD SCALER", log_file)
    log_print("=" * 70, log_file)

    scaler = fit_scaler_on_train(X_train_top)

    X_train_scaled = transform_with_scaler(scaler, X_train_top)
    X_val_scaled = transform_with_scaler(scaler, X_val_top)
    X_test_scaled = transform_with_scaler(scaler, X_test_top)

    train_scaled_check = check_scaled_matrix(
        X_train_scaled,
        "train",
        log_file,
    )

    val_scaled_check = check_scaled_matrix(
        X_val_scaled,
        "validation",
        log_file,
    )

    test_scaled_check = check_scaled_matrix(
        X_test_scaled,
        "test",
        log_file,
    )

    train_abs_mean_mean = float(
        np.abs(X_train_scaled.mean(axis=0)).mean()
    )

    train_std_mean = float(
        X_train_scaled.std(axis=0).mean()
    )

    log_print(
        f"Mean trung bình tuyệt đối của X_train_scaled: {train_abs_mean_mean}",
        log_file,
    )

    log_print(
        f"STD trung bình của X_train_scaled: {train_std_mean}",
        log_file,
    )

    save_gene_lists(
        selected_genes,
        variance_df,
        paths,
        log_file,
    )

    save_outputs(
        paths,
        X_train_top,
        X_val_top,
        X_test_top,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
        log_file,
    )

    summary_data = {
        "input_train_file": str(paths["X_train_input"]),
        "input_val_file": str(paths["X_val_input"]),
        "input_test_file": str(paths["X_test_input"]),

        "output_train_top_genes_file": str(paths["X_train_top_genes"]),
        "output_val_top_genes_file": str(paths["X_val_top_genes"]),
        "output_test_top_genes_file": str(paths["X_test_top_genes"]),

        "output_train_scaled_file": str(paths["X_train_scaled"]),
        "output_val_scaled_file": str(paths["X_val_scaled"]),
        "output_test_scaled_file": str(paths["X_test_scaled"]),

        "selected_genes_file": str(paths["selected_genes"]),
        "top_gene_variance_file": str(paths["top_gene_variance"]),
        "scaler_file": str(paths["scaler_file"]),

        "original_train_shape": str(X_train.shape),
        "original_val_shape": str(X_val.shape),
        "original_test_shape": str(X_test.shape),

        "top_n_genes_config": TOP_N_GENES,
        "selected_genes": variance_stats["selected_genes"],
        "removed_genes": variance_stats["removed_genes"],

        "train_missing_before": missing_stats["train_missing_before"],
        "val_missing_before": missing_stats["val_missing_before"],
        "test_missing_before": missing_stats["test_missing_before"],

        "train_missing_after_fill": missing_stats["train_missing_after_fill"],
        "val_missing_after_fill": missing_stats["val_missing_after_fill"],
        "test_missing_after_fill": missing_stats["test_missing_after_fill"],

        "max_variance": variance_stats["max_variance"],
        "min_selected_variance": variance_stats["min_selected_variance"],

        "zero_variance_genes_total": variance_stats["zero_variance_genes_total"],
        "zero_variance_genes_selected": variance_stats["zero_variance_genes_selected"],

        "final_top_gene_train_shape": str(X_train_top.shape),
        "final_top_gene_val_shape": str(X_val_top.shape),
        "final_top_gene_test_shape": str(X_test_top.shape),

        "final_scaled_train_shape": str(X_train_scaled.shape),
        "final_scaled_val_shape": str(X_val_scaled.shape),
        "final_scaled_test_shape": str(X_test_scaled.shape),

        "train_scaled_abs_mean_mean": train_abs_mean_mean,
        "train_scaled_std_mean": train_std_mean,

        "train_missing_after_scale": train_scaled_check["missing"],
        "val_missing_after_scale": val_scaled_check["missing"],
        "test_missing_after_scale": test_scaled_check["missing"],

        "train_inf_after_scale": train_scaled_check["inf"],
        "val_inf_after_scale": val_scaled_check["inf"],
        "test_inf_after_scale": test_scaled_check["inf"],

        "processing_status": "success",
    }

    save_summary(
        summary_data,
        paths["summary"],
        log_file,
    )

    log_print("=" * 70, log_file)
    log_print("HOÀN THÀNH FEATURE PROCESSING", log_file)
    log_print("=" * 70, log_file)

    log_print(f"Log file: {log_file}", log_file)


if __name__ == "__main__":
    main()