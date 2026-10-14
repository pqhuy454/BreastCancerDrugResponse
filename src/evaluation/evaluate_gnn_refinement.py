from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATASETS = [
    "ductal_carcinoma_in_situ_invasive_carcinoma_FFPE",
    "whole_transcriptome_analysis",
]

TOP_K_LIST = [5, 10]

BASELINE_FILE_NAME = "predicted_ic50_mean_baseline.csv"
REFINED_FILE_NAME = "refined_ic50_gnn.csv"

DATASET_EVALUATION_FILE_NAME = "gnn_refinement_evaluation.csv"
TOP_CHANGED_DRUGS_FILE_NAME = "top_changed_drugs.csv"
TOP10_COMPARISON_FILE_NAME = "top10_before_after_comparison.csv"

REPORT_FILE_NAME = "gnn_refinement_evaluation_report.csv"
LOG_FILE_PREFIX = "evaluate_gnn_refinement"


# ============================================================
# 2. THIẾT LẬP ĐƯỜNG DẪN
# ============================================================

def get_project_root() -> Path:
    """Tự động xác định thư mục gốc project."""
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    """Thiết lập đường dẫn input/output/log."""
    project_root = get_project_root()

    spatial_processed_dir = project_root / "data" / "spatial_processed"
    results_spatial_dir = project_root / "results" / "spatial"
    logs_dir = project_root / "logs"

    results_spatial_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%d%m%y_%H%M")

    return {
        "project_root": project_root,
        "spatial_processed_dir": spatial_processed_dir,
        "results_spatial_dir": results_spatial_dir,
        "logs_dir": logs_dir,
        "log_file": logs_dir / f"{LOG_FILE_PREFIX}_{run_time}.log",
        "report_file": results_spatial_dir / REPORT_FILE_NAME,
    }


# ============================================================
# 3. LOGGING
# ============================================================

def setup_logging(log_file: Path) -> logging.Logger:
    """Thiết lập logging ra console và file."""
    logger = logging.getLogger(LOG_FILE_PREFIX)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ============================================================
# 4. ĐỌC VÀ CHUẨN HÓA DỮ LIỆU
# ============================================================

def should_use_first_column_as_index(raw_df: pd.DataFrame) -> bool:
    """Kiểm tra có nên dùng cột đầu tiên làm index hay không."""
    if raw_df.shape[1] == 0:
        return False

    first_col_name = str(raw_df.columns[0]).strip().lower()

    index_like_names = {
        "spot",
        "spots",
        "spot_id",
        "spot_ids",
        "barcode",
        "barcodes",
        "index",
        "unnamed: 0",
    }

    if first_col_name in index_like_names:
        return True

    first_col_values = raw_df.iloc[:, 0]
    numeric_values = pd.to_numeric(first_col_values, errors="coerce")

    if numeric_values.isna().mean() > 0.5:
        return True

    return False


def load_prediction_matrix(file_path: Path) -> pd.DataFrame:
    """Đọc ma trận dự đoán IC50 từ CSV."""
    raw_df = pd.read_csv(file_path)

    if should_use_first_column_as_index(raw_df):
        df = pd.read_csv(file_path, index_col=0)
    else:
        df = raw_df.copy()

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def align_prediction_matrices(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Căn chỉnh baseline và refined theo spot chung và thuốc chung."""
    common_spots = baseline_df.index.intersection(refined_df.index)
    common_drugs = baseline_df.columns.intersection(refined_df.columns)

    if len(common_spots) == 0:
        raise ValueError("Không có spot chung giữa baseline và refined.")

    if len(common_drugs) == 0:
        raise ValueError("Không có thuốc chung giữa baseline và refined.")

    baseline_aligned = baseline_df.loc[common_spots, common_drugs].copy()
    refined_aligned = refined_df.loc[common_spots, common_drugs].copy()

    refined_aligned = refined_aligned.loc[
        baseline_aligned.index,
        baseline_aligned.columns,
    ]

    if baseline_aligned.shape != refined_aligned.shape:
        raise ValueError("Baseline và refined không cùng shape sau khi căn chỉnh.")

    return baseline_aligned, refined_aligned


# ============================================================
# 5. CHỈ SỐ THAY ĐỔI GIÁ TRỊ
# ============================================================

def flatten_valid_pair(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Flatten hai ma trận và chỉ giữ các cặp giá trị hợp lệ."""
    baseline_values = baseline_df.to_numpy(dtype=float).ravel()
    refined_values = refined_df.to_numpy(dtype=float).ravel()

    valid_mask = (
        np.isfinite(baseline_values)
        & np.isfinite(refined_values)
    )

    baseline_series = pd.Series(baseline_values[valid_mask])
    refined_series = pd.Series(refined_values[valid_mask])

    return baseline_series, refined_series


def compute_value_change_metrics(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> dict:
    """Tính các chỉ số mức độ thay đổi LN(IC50)."""
    change_df = refined_df - baseline_df
    abs_change_df = change_df.abs()

    change_values = change_df.to_numpy(dtype=float).ravel()
    abs_change_values = abs_change_df.to_numpy(dtype=float).ravel()

    change_values = change_values[np.isfinite(change_values)]
    abs_change_values = abs_change_values[np.isfinite(abs_change_values)]

    if len(change_values) == 0 or len(abs_change_values) == 0:
        return {
            "mean_abs_change": np.nan,
            "median_abs_change": np.nan,
            "max_abs_change": np.nan,
            "mean_signed_change": np.nan,
            "std_abs_change": np.nan,
        }

    return {
        "mean_abs_change": float(np.mean(abs_change_values)),
        "median_abs_change": float(np.median(abs_change_values)),
        "max_abs_change": float(np.max(abs_change_values)),
        "mean_signed_change": float(np.mean(change_values)),
        "std_abs_change": float(np.std(abs_change_values)),
    }


def compute_correlation_metrics(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> dict:
    """Tính Pearson và Spearman correlation sau khi flatten."""
    baseline_series, refined_series = flatten_valid_pair(
        baseline_df=baseline_df,
        refined_df=refined_df,
    )

    if len(baseline_series) < 2:
        return {
            "pearson_corr": np.nan,
            "spearman_corr": np.nan,
        }

    pearson_corr = baseline_series.corr(refined_series, method="pearson")
    spearman_corr = baseline_series.corr(refined_series, method="spearman")

    return {
        "pearson_corr": float(pearson_corr),
        "spearman_corr": float(spearman_corr),
    }


# ============================================================
# 6. TOP-K VÀ RANK SHIFT
# ============================================================

def get_top_k_drugs(row: pd.Series, k: int) -> list:
    """Lấy Top-K thuốc có LN(IC50) thấp nhất, bỏ qua NaN."""
    valid_row = row.dropna()

    if len(valid_row) == 0:
        return []

    return list(valid_row.sort_values(ascending=True).head(k).index)


def compute_top1_changed_rate(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> float:
    """Tính tỷ lệ spot bị thay đổi thuốc Top-1."""
    changed_count = 0
    valid_spot_count = 0

    for spot_id in baseline_df.index:
        baseline_top1 = get_top_k_drugs(baseline_df.loc[spot_id], 1)
        refined_top1 = get_top_k_drugs(refined_df.loc[spot_id], 1)

        if len(baseline_top1) == 0 or len(refined_top1) == 0:
            continue

        valid_spot_count += 1

        if baseline_top1[0] != refined_top1[0]:
            changed_count += 1

    if valid_spot_count == 0:
        return np.nan

    return float(changed_count / valid_spot_count)


def compute_topk_overlap(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
    k: int,
) -> float:
    """Tính trung bình tỷ lệ giao nhau Top-K giữa before và after."""
    overlap_values = []

    for spot_id in baseline_df.index:
        baseline_top_k = get_top_k_drugs(baseline_df.loc[spot_id], k)
        refined_top_k = get_top_k_drugs(refined_df.loc[spot_id], k)

        if len(baseline_top_k) == 0 or len(refined_top_k) == 0:
            continue

        denominator = min(k, len(baseline_top_k), len(refined_top_k))

        if denominator == 0:
            continue

        overlap = len(set(baseline_top_k).intersection(set(refined_top_k))) / denominator
        overlap_values.append(overlap)

    if len(overlap_values) == 0:
        return np.nan

    return float(np.mean(overlap_values))


def compute_topk_metrics(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> dict:
    """Tính Top1 changed rate và Top-K overlap."""
    metrics = {
        "top1_changed_rate": compute_top1_changed_rate(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )
    }

    for k in TOP_K_LIST:
        metrics[f"top{k}_overlap_mean"] = compute_topk_overlap(
            baseline_df=baseline_df,
            refined_df=refined_df,
            k=k,
        )

    return metrics


def compute_mean_rank_shift(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> float:
    """Tính trung bình độ lệch thứ hạng tuyệt đối của thuốc."""
    rank_shift_values = []

    for spot_id in baseline_df.index:
        baseline_row = baseline_df.loc[spot_id]
        refined_row = refined_df.loc[spot_id]

        valid_drugs = baseline_row.dropna().index.intersection(
            refined_row.dropna().index
        )

        if len(valid_drugs) < 2:
            continue

        baseline_rank = baseline_row.loc[valid_drugs].rank(
            method="average",
            ascending=True,
        )

        refined_rank = refined_row.loc[valid_drugs].rank(
            method="average",
            ascending=True,
        )

        rank_shift = (refined_rank - baseline_rank).abs()
        rank_shift_values.extend(rank_shift.to_numpy(dtype=float).tolist())

    if len(rank_shift_values) == 0:
        return np.nan

    return float(np.mean(rank_shift_values))


# ============================================================
# 7. TOP THUỐC THAY ĐỔI NHIỀU NHẤT
# ============================================================

def compute_top_changed_drugs(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tính Top-20 thuốc thay đổi nhiều nhất."""
    change_df = refined_df - baseline_df
    abs_change_df = change_df.abs()

    rows = []

    for drug in baseline_df.columns:
        change_values = change_df[drug].dropna()
        abs_change_values = abs_change_df[drug].dropna()

        if len(abs_change_values) == 0:
            continue

        rows.append({
            "drug": drug,
            "mean_abs_change": float(abs_change_values.mean()),
            "mean_signed_change": float(change_values.mean()),
            "std_abs_change": float(abs_change_values.std(ddof=0)),
        })

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        return result_df

    return result_df.sort_values(
        by="mean_abs_change",
        ascending=False,
    ).head(20).reset_index(drop=True)


# ============================================================
# 8. SO SÁNH TOP-10 TỔNG THỂ
# ============================================================

def compute_top10_before_after(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
) -> pd.DataFrame:
    """So sánh Top-10 thuốc tổng thể trước và sau refine."""
    baseline_mean = baseline_df.mean(axis=0, skipna=True)
    refined_mean = refined_df.mean(axis=0, skipna=True)

    baseline_top10 = baseline_mean.sort_values(ascending=True).head(10)
    refined_top10 = refined_mean.sort_values(ascending=True).head(10)

    rows = []

    for rank in range(1, 11):
        baseline_drug = baseline_top10.index[rank - 1] if rank <= len(baseline_top10) else None
        refined_drug = refined_top10.index[rank - 1] if rank <= len(refined_top10) else None

        baseline_value = float(baseline_top10.iloc[rank - 1]) if rank <= len(baseline_top10) else np.nan
        refined_value = float(refined_top10.iloc[rank - 1]) if rank <= len(refined_top10) else np.nan

        rows.append({
            "rank": rank,
            "baseline_top10_drug": baseline_drug,
            "baseline_mean_LN_IC50": baseline_value,
            "refined_top10_drug": refined_drug,
            "refined_mean_LN_IC50": refined_value,
            "same_drug": baseline_drug == refined_drug,
        })

    return pd.DataFrame(rows)


def get_overall_top1(prediction_df: pd.DataFrame) -> tuple[str | None, float]:
    """Lấy thuốc Top-1 tổng thể theo mean LN(IC50) thấp nhất."""
    mean_values = prediction_df.mean(axis=0, skipna=True).dropna()

    if len(mean_values) == 0:
        return None, np.nan

    top1_drug = str(mean_values.idxmin())
    top1_value = float(mean_values.min())

    return top1_drug, top1_value


# ============================================================
# 9. ĐÁNH GIÁ MỘT DATASET
# ============================================================

def save_single_row_csv(row: dict, output_file: Path) -> None:
    """Lưu một dòng kết quả thành CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_file, index=False)


def evaluate_dataset(
    dataset_name: str,
    paths: dict,
    logger: logging.Logger,
) -> dict | None:
    """Đánh giá trước và sau GNN refine cho một dataset."""
    dataset_dir = paths["spatial_processed_dir"] / dataset_name

    baseline_file = dataset_dir / BASELINE_FILE_NAME
    refined_file = dataset_dir / REFINED_FILE_NAME

    logger.info("=" * 70)
    logger.info(f"DATASET: {dataset_name}")

    if not baseline_file.exists():
        logger.warning(f"WARNING: Thiếu file baseline: {baseline_file}")
        return None

    if not refined_file.exists():
        logger.warning(f"WARNING: Thiếu file refined: {refined_file}")
        return None

    try:
        baseline_df = load_prediction_matrix(baseline_file)
        refined_df = load_prediction_matrix(refined_file)

        baseline_shape = str(baseline_df.shape)
        refined_shape = str(refined_df.shape)

        baseline_df, refined_df = align_prediction_matrices(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )

        value_metrics = compute_value_change_metrics(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )

        corr_metrics = compute_correlation_metrics(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )

        topk_metrics = compute_topk_metrics(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )

        mean_rank_shift = compute_mean_rank_shift(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )

        baseline_top1_drug, baseline_top1_mean_ic50 = get_overall_top1(baseline_df)
        refined_top1_drug, refined_top1_mean_ic50 = get_overall_top1(refined_df)

        result_row = {
            "dataset_name": dataset_name,
            "n_spots": int(baseline_df.shape[0]),
            "n_drugs": int(baseline_df.shape[1]),
            "baseline_shape": baseline_shape,
            "refined_shape": refined_shape,
            **value_metrics,
            **corr_metrics,
            **topk_metrics,
            "mean_rank_shift": float(mean_rank_shift),
            "baseline_top1_drug": baseline_top1_drug,
            "baseline_top1_mean_ic50": baseline_top1_mean_ic50,
            "refined_top1_drug": refined_top1_drug,
            "refined_top1_mean_ic50": refined_top1_mean_ic50,
        }

        evaluation_file = dataset_dir / DATASET_EVALUATION_FILE_NAME
        save_single_row_csv(row=result_row, output_file=evaluation_file)

        top_changed_drugs_df = compute_top_changed_drugs(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )

        top_changed_drugs_file = dataset_dir / TOP_CHANGED_DRUGS_FILE_NAME
        top_changed_drugs_df.to_csv(top_changed_drugs_file, index=False)

        top10_comparison_df = compute_top10_before_after(
            baseline_df=baseline_df,
            refined_df=refined_df,
        )

        top10_comparison_file = dataset_dir / TOP10_COMPARISON_FILE_NAME
        top10_comparison_df.to_csv(top10_comparison_file, index=False)

        logger.info(f"Baseline shape: {baseline_shape}")
        logger.info(f"Refined shape: {refined_shape}")
        logger.info(f"Mean Absolute Change: {result_row['mean_abs_change']:.6f}")
        logger.info(f"Pearson: {result_row['pearson_corr']:.6f}")
        logger.info(f"Spearman: {result_row['spearman_corr']:.6f}")
        logger.info(f"Top1 Changed Rate: {result_row['top1_changed_rate']:.6f}")
        logger.info(f"Top5 Overlap: {result_row['top5_overlap_mean']:.6f}")
        logger.info(f"Top10 Overlap: {result_row['top10_overlap_mean']:.6f}")
        logger.info(f"Mean Rank Shift: {result_row['mean_rank_shift']:.6f}")
        logger.info(f"Baseline Top1: {baseline_top1_drug} ({baseline_top1_mean_ic50:.6f})")
        logger.info(f"Refined Top1: {refined_top1_drug} ({refined_top1_mean_ic50:.6f})")
        logger.info("Saved files:")
        logger.info(f"- {evaluation_file}")
        logger.info(f"- {top_changed_drugs_file}")
        logger.info(f"- {top10_comparison_file}")

        return result_row

    except Exception as error:
        logger.exception(f"ERROR: Lỗi khi xử lý dataset {dataset_name}: {error}")
        return None


# ============================================================
# 10. LƯU REPORT TỔNG HỢP
# ============================================================

def save_report(
    report_rows: list[dict],
    report_file: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Lưu report tổng hợp."""
    report_columns = [
        "dataset_name",
        "n_spots",
        "n_drugs",
        "mean_abs_change",
        "median_abs_change",
        "max_abs_change",
        "mean_signed_change",
        "std_abs_change",
        "pearson_corr",
        "spearman_corr",
        "top1_changed_rate",
        "top5_overlap_mean",
        "top10_overlap_mean",
        "mean_rank_shift",
        "baseline_top1_drug",
        "baseline_top1_mean_ic50",
        "refined_top1_drug",
        "refined_top1_mean_ic50",
    ]

    if len(report_rows) == 0:
        report_df = pd.DataFrame(columns=report_columns)
    else:
        report_df = pd.DataFrame(report_rows)
        existing_columns = [column for column in report_columns if column in report_df.columns]
        report_df = report_df[existing_columns]

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_file, index=False)

    logger.info("=" * 70)
    logger.info(f"Đã lưu report tổng hợp: {report_file}")
    logger.info("=" * 70)

    return report_df


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    """Hàm chính."""
    paths = setup_paths()
    logger = setup_logging(paths["log_file"])

    logger.info("=" * 70)
    logger.info("BẮT ĐẦU ĐÁNH GIÁ GNN REFINEMENT")
    logger.info("=" * 70)
    logger.info(f"PROJECT_ROOT: {paths['project_root']}")
    logger.info(f"SPATIAL_PROCESSED_DIR: {paths['spatial_processed_dir']}")
    logger.info(f"RESULTS_SPATIAL_DIR: {paths['results_spatial_dir']}")
    logger.info(f"LOG_FILE: {paths['log_file']}")
    logger.info("")

    report_rows = []

    for dataset_name in DATASETS:
        result_row = evaluate_dataset(
            dataset_name=dataset_name,
            paths=paths,
            logger=logger,
        )

        if result_row is not None:
            report_rows.append(result_row)

    report_df = save_report(
        report_rows=report_rows,
        report_file=paths["report_file"],
        logger=logger,
    )

    logger.info("")
    logger.info("BẢNG REPORT TỔNG HỢP")
    logger.info(report_df.to_string(index=False))
    logger.info("")
    logger.info("=" * 70)
    logger.info("HOÀN THÀNH ĐÁNH GIÁ GNN REFINEMENT")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
