from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ==========================================================
# 1. CẤU HÌNH Ở ĐẦU FILE
# ==========================================================

RAW_DIR_NAME = "raw"
PROCESSED_DIR_NAME = "processed"

EXPRESSION_FILE = "rnaseq_merged_rsem_tpm_20250922.csv"
DRUG_RESPONSE_FILE = "GDSC2_fitted_dose_response_27Oct23.xlsx"
DRUG_METADATA_FILE = "screened_compounds_rel_8.5.csv"

FILTER_BREAST_ONLY = False
TOP_N_DRUGS = 300

MIN_OBSERVED_DRUGS_PER_CELL_LINE = 1
MIN_OBSERVED_CELL_LINES_PER_DRUG = 1

USE_DRUG_METADATA = True

MISSING_GENE_THRESHOLD = 0.20
REMOVE_ZERO_VARIANCE_GENES = True
FILL_EXPRESSION_MISSING = True
EXPRESSION_FILL_METHOD = "median"

MERGE_DUPLICATE_GENES = True
DUPLICATE_GENE_AGG_METHOD = "mean"

CELL_LINE_COLUMN = "SANGER_MODEL_ID"
DRUG_NAME_COLUMN = "DRUG_NAME"
Y_VALUE_COLUMN = "LN_IC50"

KEEP_Y_NAN = True
CREATE_MASK = True
DUPLICATE_RESPONSE_AGG_METHOD = "mean"

DO_SPLIT = True
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

OVERWRITE_OUTPUTS = True
SAVE_INTERMEDIATE_FILES = False
SAVE_LOG = True


GENE_IDENTIFIER_CANDIDATES = [
    "gene_name",
    "gene_symbol",
    "symbol",
    "gene_id",
    "ensembl_gene_id",
    "model_id",
    "Unnamed: 0",
]

EXPRESSION_METADATA_COLUMNS = [
    "gene_id",
    "gene_name",
    "gene_symbol",
    "symbol",
    "ensembl_gene_id",
    "model_id",
    "SANGER_MODEL_ID",
    "Unnamed: 0",
]


# ==========================================================
# 2. SETUP PATH, LOG, CONFIG
# ==========================================================

def get_project_root() -> Path:
    """
    Tự động xác định thư mục gốc project.

    File hiện tại:
        src/data/data_processing.py

    PROJECT_ROOT:
        parents[2]
    """
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    """
    Thiết lập toàn bộ đường dẫn.
    Không dùng đường dẫn tuyệt đối.
    """
    project_root = get_project_root()

    data_dir = project_root / "data"
    raw_dir = data_dir / RAW_DIR_NAME
    processed_dir = data_dir / PROCESSED_DIR_NAME

    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"

    logs_dir = project_root / "logs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if DO_SPLIT:
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{Path(__file__).stem}_{run_time}.log"

    return {
        "project_root": project_root,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "logs_dir": logs_dir,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "test_dir": test_dir,
        "log_file": log_file,

        "expression_file": raw_dir / EXPRESSION_FILE,
        "drug_response_file": raw_dir / DRUG_RESPONSE_FILE,
        "drug_metadata_file": raw_dir / DRUG_METADATA_FILE,

        "X_expression_full": processed_dir / "X_expression_full.csv",
        "y_ic50_full": processed_dir / "y_ic50_full.csv",
        "y_mask_full": processed_dir / "y_mask_full.csv",

        "drug_metadata_clean": processed_dir / "drug_metadata_clean.csv",

        "X_expression": processed_dir / "X_expression.csv",
        "y_ic50": processed_dir / "y_ic50.csv",
        "y_mask": processed_dir / "y_mask.csv",
        "selected_cell_lines": processed_dir / "selected_cell_lines.txt",
        "selected_drugs": processed_dir / "selected_drugs.txt",

        "data_processing_summary": processed_dir / "data_processing_summary.csv",
        "split_summary": processed_dir / "split_summary.csv",

        "X_train": train_dir / "X_train.csv",
        "y_train": train_dir / "y_train.csv",
        "mask_train": train_dir / "mask_train.csv",

        "X_val": val_dir / "X_val.csv",
        "y_val": val_dir / "y_val.csv",
        "mask_val": val_dir / "mask_val.csv",

        "X_test": test_dir / "X_test.csv",
        "y_test": test_dir / "y_test.csv",
        "mask_test": test_dir / "mask_test.csv",
    }


def log_print(message: str = "", log_file: Path | None = None) -> None:
    """
    Vừa in ra terminal, vừa ghi vào file log.
    """
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def save_dataframe(
    df: pd.DataFrame,
    output_file: Path,
    log_file: Path,
    index: bool = True,
) -> bool:
    """
    Lưu DataFrame ra CSV.
    """
    if output_file.exists():
        if OVERWRITE_OUTPUTS:
            log_print(f"File đã tồn tại, sẽ ghi đè: {output_file}", log_file)
        else:
            log_print(f"LỖI: File đã tồn tại và OVERWRITE_OUTPUTS=False: {output_file}", log_file)
            return False

    df.to_csv(output_file, index=index)
    return True


def save_list_to_txt(values, output_file: Path, log_file: Path) -> bool:
    """
    Lưu danh sách ra file txt, mỗi dòng một giá trị.
    """
    if output_file.exists():
        if OVERWRITE_OUTPUTS:
            log_print(f"File đã tồn tại, sẽ ghi đè: {output_file}", log_file)
        else:
            log_print(f"LỖI: File đã tồn tại và OVERWRITE_OUTPUTS=False: {output_file}", log_file)
            return False

    with open(output_file, "w", encoding="utf-8") as f:
        for value in values:
            f.write(str(value) + "\n")

    return True


def save_summary(summary_data: dict, output_file: Path, log_file: Path) -> bool:
    """
    Lưu summary ra CSV.
    """
    summary_df = pd.DataFrame([summary_data])
    ok = save_dataframe(summary_df, output_file, log_file, index=False)

    if ok:
        log_print(f"Đã lưu summary: {output_file}", log_file)

    return ok


def validate_config(log_file: Path) -> dict:
    """
    Kiểm tra và chuẩn hóa cấu hình.
    """
    missing_gene_threshold = MISSING_GENE_THRESHOLD

    if not (0 <= missing_gene_threshold <= 1):
        log_print(
            "CẢNH BÁO: MISSING_GENE_THRESHOLD không nằm trong [0, 1]. "
            "Tự động đưa về 0.20.",
            log_file,
        )
        missing_gene_threshold = 0.20

    train_ratio = TRAIN_RATIO
    val_ratio = VAL_RATIO
    test_ratio = TEST_RATIO

    ratio_sum = train_ratio + val_ratio + test_ratio

    if not np.isclose(ratio_sum, 1.0):
        log_print(
            f"CẢNH BÁO: Tổng tỷ lệ split = {ratio_sum:.6f}, khác 1. "
            "Sẽ chuẩn hóa lại.",
            log_file,
        )

        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum

        log_print(
            f"Tỷ lệ sau chuẩn hóa: "
            f"train={train_ratio:.4f}, "
            f"validation={val_ratio:.4f}, "
            f"test={test_ratio:.4f}",
            log_file,
        )

    top_n_drugs = TOP_N_DRUGS

    if top_n_drugs <= 0:
        log_print(
            "CẢNH BÁO: TOP_N_DRUGS <= 0. Sẽ lấy toàn bộ thuốc sau lọc.",
            log_file,
        )

    return {
        "missing_gene_threshold": missing_gene_threshold,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "top_n_drugs": top_n_drugs,
    }


def print_config(config: dict, paths: dict, log_file: Path) -> None:
    """
    In cấu hình đang dùng.
    """
    log_print("=" * 70, log_file)
    log_print("CẤU HÌNH DATA PROCESSING", log_file)
    log_print("=" * 70, log_file)

    log_print(f"PROJECT_ROOT: {paths['project_root']}", log_file)
    log_print(f"RAW_DIR: {paths['raw_dir']}", log_file)
    log_print(f"PROCESSED_DIR: {paths['processed_dir']}", log_file)

    log_print(f"EXPRESSION_FILE: {EXPRESSION_FILE}", log_file)
    log_print(f"DRUG_RESPONSE_FILE: {DRUG_RESPONSE_FILE}", log_file)
    log_print(f"DRUG_METADATA_FILE: {DRUG_METADATA_FILE}", log_file)

    log_print(f"FILTER_BREAST_ONLY: {FILTER_BREAST_ONLY}", log_file)
    log_print(f"TOP_N_DRUGS: {TOP_N_DRUGS}", log_file)
    log_print(f"TOP_N_DRUGS sau xử lý: {config['top_n_drugs']}", log_file)
    log_print(f"MIN_OBSERVED_DRUGS_PER_CELL_LINE: {MIN_OBSERVED_DRUGS_PER_CELL_LINE}", log_file)
    log_print(f"MIN_OBSERVED_CELL_LINES_PER_DRUG: {MIN_OBSERVED_CELL_LINES_PER_DRUG}", log_file)

    log_print(f"MISSING_GENE_THRESHOLD: {config['missing_gene_threshold']}", log_file)
    log_print(f"REMOVE_ZERO_VARIANCE_GENES: {REMOVE_ZERO_VARIANCE_GENES}", log_file)
    log_print(f"FILL_EXPRESSION_MISSING: {FILL_EXPRESSION_MISSING}", log_file)
    log_print(f"EXPRESSION_FILL_METHOD: {EXPRESSION_FILL_METHOD}", log_file)
    log_print(f"MERGE_DUPLICATE_GENES: {MERGE_DUPLICATE_GENES}", log_file)
    log_print(f"DUPLICATE_GENE_AGG_METHOD: {DUPLICATE_GENE_AGG_METHOD}", log_file)

    log_print(f"KEEP_Y_NAN: {KEEP_Y_NAN}", log_file)
    log_print(f"CREATE_MASK: {CREATE_MASK}", log_file)
    log_print(f"DUPLICATE_RESPONSE_AGG_METHOD: {DUPLICATE_RESPONSE_AGG_METHOD}", log_file)

    log_print(f"DO_SPLIT: {DO_SPLIT}", log_file)
    log_print(
        f"SPLIT_RATIO: train={config['train_ratio']}, "
        f"validation={config['val_ratio']}, "
        f"test={config['test_ratio']}",
        log_file,
    )

    log_print(f"OVERWRITE_OUTPUTS: {OVERWRITE_OUTPUTS}", log_file)
    log_print(f"SAVE_INTERMEDIATE_FILES: {SAVE_INTERMEDIATE_FILES}", log_file)
    log_print(f"SAVE_LOG: {SAVE_LOG}", log_file)


def check_raw_files_for_pipeline(paths: dict, log_file: Path) -> bool:
    """
    Kiểm tra các file raw bắt buộc và optional.
    """
    log_print("=" * 70, log_file)
    log_print("KIỂM TRA FILE RAW CẦN DÙNG", log_file)
    log_print("=" * 70, log_file)

    required_files = {
        "Expression TPM": paths["expression_file"],
        "Drug response": paths["drug_response_file"],
    }

    optional_files = {
        "Drug metadata": paths["drug_metadata_file"],
    }

    all_required_ok = True

    for name, file_path in required_files.items():
        log_print(f"[BẮT BUỘC] {name}: {file_path}", log_file)

        if file_path.exists():
            log_print("Trạng thái: tồn tại", log_file)
        else:
            log_print(f"LỖI: Thiếu file bắt buộc: {file_path.name}", log_file)
            all_required_ok = False

    for name, file_path in optional_files.items():
        log_print(f"[OPTIONAL] {name}: {file_path}", log_file)

        if file_path.exists():
            log_print("Trạng thái: tồn tại", log_file)
        else:
            log_print(f"CẢNH BÁO: Thiếu file optional: {file_path.name}", log_file)

    return all_required_ok


# ==========================================================
# 4. XỬ LÝ RNA-seq TPM
# ==========================================================

def detect_gene_identifier_column(df: pd.DataFrame) -> str:
    """
    Phát hiện cột định danh gene theo thứ tự ưu tiên.
    """
    for candidate in GENE_IDENTIFIER_CANDIDATES:
        if candidate in df.columns:
            return candidate

    return df.columns[0]


def load_expression_data(file_path: Path, log_file: Path) -> pd.DataFrame | None:
    """
    Đọc file RNA-seq TPM bằng pandas.read_csv(low_memory=False).
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        log_print(f"Đọc expression TPM thành công: {file_path.name}", log_file)
        return df

    except Exception as e:
        log_print("=" * 70, log_file)
        log_print("LỖI ĐỌC FILE EXPRESSION TPM", log_file)
        log_print("=" * 70, log_file)
        log_print(f"Tên file: {file_path.name}", log_file)
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)
        return None


def prepare_expression_matrix(
    df: pd.DataFrame,
    log_file: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Chuẩn hóa expression TPM về dạng cell line × gene.
    """
    log_print("=" * 70, log_file)
    log_print("PHẦN 2: XỬ LÝ RNA-seq TPM", log_file)
    log_print("=" * 70, log_file)

    raw_shape = df.shape
    raw_missing = int(df.isna().sum().sum())
    raw_duplicate_rows = int(df.duplicated().sum())

    log_print(f"Shape ban đầu: {raw_shape}", log_file)
    log_print(f"Số missing ban đầu: {raw_missing}", log_file)
    log_print(f"Số dòng trùng lặp: {raw_duplicate_rows}", log_file)

    gene_id_col = detect_gene_identifier_column(df)
    log_print(f"Cột định danh gene sử dụng: {gene_id_col}", log_file)

    metadata_cols_found = [
        col for col in EXPRESSION_METADATA_COLUMNS
        if col in df.columns
    ]

    expression_cols = [
        col for col in df.columns
        if col not in metadata_cols_found
    ]

    if gene_id_col in expression_cols:
        expression_cols.remove(gene_id_col)

    if len(expression_cols) == 0:
        raise ValueError("Không tìm thấy cột expression sau khi loại metadata columns.")

    gene_names = df[gene_id_col].astype(str).str.strip()

    temp_df = df.loc[:, expression_cols].copy()
    temp_df.index = gene_names

    valid_gene_mask = (
        temp_df.index.notna()
        & (temp_df.index.astype(str).str.strip() != "")
        & (temp_df.index.astype(str).str.lower() != "nan")
    )

    removed_empty_gene_names = int((~valid_gene_mask).sum())
    temp_df = temp_df.loc[valid_gene_mask]

    log_print("Dạng dữ liệu xử lý: gene × cell line", log_file)

    expression_matrix = temp_df.T.copy()

    expression_matrix.index = expression_matrix.index.astype(str).str.strip()
    expression_matrix.columns = expression_matrix.columns.astype(str).str.strip()

    valid_cell_line_mask = (
        expression_matrix.index.notna()
        & (expression_matrix.index.astype(str).str.strip() != "")
        & (expression_matrix.index.astype(str).str.lower() != "nan")
    )

    expression_matrix = expression_matrix.loc[valid_cell_line_mask]

    log_print("Đã transpose thành: cell line × gene", log_file)
    log_print(f"Số gene name rỗng/nan bị loại: {removed_empty_gene_names}", log_file)
    log_print(f"Số cell line sau transpose: {expression_matrix.shape[0]}", log_file)
    log_print(f"Số gene trước xử lý: {expression_matrix.shape[1]}", log_file)

    stats = {
        "expression_raw_shape": str(raw_shape),
        "expression_raw_missing": raw_missing,
        "expression_raw_duplicate_rows": raw_duplicate_rows,
        "gene_identifier_column": gene_id_col,
        "removed_empty_gene_names": removed_empty_gene_names,
        "cell_lines_after_transpose": expression_matrix.shape[0],
        "genes_before_cleaning": expression_matrix.shape[1],
    }

    return expression_matrix, stats


def clean_expression_matrix(
    expression_matrix: pd.DataFrame,
    config: dict,
    log_file: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Làm sạch ma trận expression.
    """
    log_print("=" * 70, log_file)
    log_print("LÀM SẠCH EXPRESSION MATRIX", log_file)
    log_print("=" * 70, log_file)

    expression_matrix = expression_matrix.apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )

    duplicate_gene_count = int(expression_matrix.columns.duplicated().sum())
    log_print(f"Số gene trùng: {duplicate_gene_count}", log_file)

    if duplicate_gene_count > 0:
        if MERGE_DUPLICATE_GENES:
            if DUPLICATE_GENE_AGG_METHOD.lower() == "mean":
                log_print("Gộp gene trùng bằng mean.", log_file)
                expression_matrix = expression_matrix.T.groupby(level=0).mean().T
            else:
                log_print(
                    "CẢNH BÁO: Chỉ hỗ trợ mean cho gene trùng. Sẽ dùng mean.",
                    log_file,
                )
                expression_matrix = expression_matrix.T.groupby(level=0).mean().T
        else:
            log_print("Không gộp gene trùng. Giữ bản đầu tiên.", log_file)
            expression_matrix = expression_matrix.loc[
                :, ~expression_matrix.columns.duplicated(keep="first")
            ]

    missing_ratio = expression_matrix.isna().mean(axis=0)
    genes_missing_mask = missing_ratio > config["missing_gene_threshold"]
    genes_removed_missing = int(genes_missing_mask.sum())

    expression_matrix = expression_matrix.loc[:, ~genes_missing_mask]

    log_print(
        f"Số gene bị loại do missing > {config['missing_gene_threshold']}: "
        f"{genes_removed_missing}",
        log_file,
    )

    if FILL_EXPRESSION_MISSING:
        if EXPRESSION_FILL_METHOD.lower() == "median":
            gene_medians = expression_matrix.median(axis=0, skipna=True)
            expression_matrix = expression_matrix.fillna(gene_medians)
            expression_matrix = expression_matrix.fillna(0)
        else:
            log_print(
                "CẢNH BÁO: EXPRESSION_FILL_METHOD không hỗ trợ. Sẽ fill missing bằng 0.",
                log_file,
            )
            expression_matrix = expression_matrix.fillna(0)

    genes_removed_zero_variance = 0

    if REMOVE_ZERO_VARIANCE_GENES:
        gene_variance = expression_matrix.var(axis=0)
        zero_variance_mask = gene_variance == 0
        genes_removed_zero_variance = int(zero_variance_mask.sum())

        expression_matrix = expression_matrix.loc[:, ~zero_variance_mask]

    final_missing = int(expression_matrix.isna().sum().sum())

    log_print(f"Số gene bị loại do variance = 0: {genes_removed_zero_variance}", log_file)
    log_print(f"Shape cuối cùng: {expression_matrix.shape}", log_file)
    log_print(f"Missing cuối cùng: {final_missing}", log_file)

    stats = {
        "genes_removed_missing": genes_removed_missing,
        "genes_removed_zero_variance": genes_removed_zero_variance,
        "expression_after_clean_shape": str(expression_matrix.shape),
        "expression_final_missing": final_missing,
    }

    return expression_matrix, stats


# ==========================================================
# 5. XỬ LÝ DRUG RESPONSE GDSC2
# ==========================================================

def load_drug_response_data(file_path: Path, log_file: Path) -> pd.DataFrame | None:
    """
    Đọc file GDSC2 bằng pandas.read_excel().
    """
    try:
        df = pd.read_excel(file_path)
        log_print(f"Đọc drug response thành công: {file_path.name}", log_file)
        return df

    except Exception as e:
        log_print("=" * 70, log_file)
        log_print("LỖI ĐỌC FILE DRUG RESPONSE", log_file)
        log_print("=" * 70, log_file)
        log_print(f"Tên file: {file_path.name}", log_file)
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)
        return None


def create_mask_matrix(y_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo mask matrix:
    - 1 nếu có LN_IC50
    - 0 nếu missing
    """
    return y_matrix.notna().astype(int)


def prepare_drug_response_matrix(
    df: pd.DataFrame,
    log_file: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """
    Làm sạch drug response và pivot thành cell line × drug.
    """
    log_print("=" * 70, log_file)
    log_print("PHẦN 3: XỬ LÝ DRUG RESPONSE GDSC2", log_file)
    log_print("=" * 70, log_file)

    raw_shape = df.shape
    raw_rows = df.shape[0]

    log_print(f"Shape GDSC ban đầu: {raw_shape}", log_file)
    log_print(f"Số dòng ban đầu: {raw_rows}", log_file)

    required_columns = [
        CELL_LINE_COLUMN,
        DRUG_NAME_COLUMN,
        Y_VALUE_COLUMN,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        log_print("LỖI: Drug response thiếu cột bắt buộc:", log_file)

        for col in missing_columns:
            log_print(f"- {col}", log_file)

        return None, None, {
            "drug_response_raw_shape": str(raw_shape),
            "drug_response_status": "missing_required_columns",
        }

    data = df[required_columns].copy()

    data[CELL_LINE_COLUMN] = data[CELL_LINE_COLUMN].astype(str).str.strip()
    data[DRUG_NAME_COLUMN] = data[DRUG_NAME_COLUMN].astype(str).str.strip()
    data[Y_VALUE_COLUMN] = pd.to_numeric(data[Y_VALUE_COLUMN], errors="coerce")

    valid_mask = (
        data[CELL_LINE_COLUMN].notna()
        & (data[CELL_LINE_COLUMN] != "")
        & (data[CELL_LINE_COLUMN].str.lower() != "nan")
        & data[DRUG_NAME_COLUMN].notna()
        & (data[DRUG_NAME_COLUMN] != "")
        & (data[DRUG_NAME_COLUMN].str.lower() != "nan")
        & data[Y_VALUE_COLUMN].notna()
    )

    removed_rows = int((~valid_mask).sum())
    data = data.loc[valid_mask].copy()

    duplicated_pairs = int(
        data.duplicated(
            subset=[CELL_LINE_COLUMN, DRUG_NAME_COLUMN],
            keep=False,
        ).sum()
    )

    if DUPLICATE_RESPONSE_AGG_METHOD.lower() == "mean":
        grouped_df = (
            data.groupby(
                [CELL_LINE_COLUMN, DRUG_NAME_COLUMN],
                as_index=False,
            )[Y_VALUE_COLUMN]
            .mean()
        )
    else:
        log_print(
            "CẢNH BÁO: DUPLICATE_RESPONSE_AGG_METHOD chỉ hỗ trợ mean. Sẽ dùng mean.",
            log_file,
        )
        grouped_df = (
            data.groupby(
                [CELL_LINE_COLUMN, DRUG_NAME_COLUMN],
                as_index=False,
            )[Y_VALUE_COLUMN]
            .mean()
        )

    y_matrix = grouped_df.pivot(
        index=CELL_LINE_COLUMN,
        columns=DRUG_NAME_COLUMN,
        values=Y_VALUE_COLUMN,
    )

    y_matrix.index = y_matrix.index.astype(str).str.strip()
    y_matrix.columns = y_matrix.columns.astype(str).str.strip()

    y_matrix = y_matrix.sort_index(axis=0).sort_index(axis=1)

    mask_matrix = create_mask_matrix(y_matrix)

    if not KEEP_Y_NAN:
        y_matrix = y_matrix.fillna(0)

    observed_values = int(mask_matrix.sum().sum())
    total_values = int(mask_matrix.shape[0] * mask_matrix.shape[1])
    missing_rate = 1 - observed_values / total_values if total_values > 0 else np.nan

    log_print(f"Số dòng bị loại: {removed_rows}", log_file)
    log_print(f"Số cặp trùng: {duplicated_pairs}", log_file)
    log_print(f"Số cell line duy nhất: {y_matrix.shape[0]}", log_file)
    log_print(f"Số thuốc duy nhất: {y_matrix.shape[1]}", log_file)
    log_print(f"Shape y: {y_matrix.shape}", log_file)
    log_print(f"Observed values: {observed_values}", log_file)
    log_print(f"Missing rate: {missing_rate:.4f}", log_file)

    stats = {
        "drug_response_raw_shape": str(raw_shape),
        "drug_response_rows_removed": removed_rows,
        "duplicated_response_pairs": duplicated_pairs,
        "y_full_shape": str(y_matrix.shape),
        "y_full_observed_values": observed_values,
        "y_full_missing_rate": missing_rate,
    }

    return y_matrix, mask_matrix, stats


# ==========================================================
# 6. XỬ LÝ DRUG METADATA
# ==========================================================

def find_column_case_insensitive(columns, target_name: str):
    """
    Tìm cột không phân biệt hoa thường.
    """
    for col in columns:
        if str(col).lower() == target_name.lower():
            return col

    return None


def prepare_drug_metadata(file_path: Path, paths: dict, log_file: Path):
    """
    Đọc và làm sạch drug metadata.
    """
    if not USE_DRUG_METADATA:
        log_print("Bỏ qua drug metadata do USE_DRUG_METADATA=False.", log_file)
        return None, {}

    if not file_path.exists():
        log_print("CẢNH BÁO: Không tìm thấy drug metadata.", log_file)
        return None, {}

    log_print("=" * 70, log_file)
    log_print("PHẦN 4: XỬ LÝ DRUG METADATA", log_file)
    log_print("=" * 70, log_file)

    try:
        df = pd.read_csv(file_path, low_memory=False)

    except Exception as e:
        log_print("LỖI đọc drug metadata", log_file)
        log_print(f"Tên file: {file_path.name}", log_file)
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)
        return None, {}

    drug_name_col = find_column_case_insensitive(df.columns, "DRUG_NAME")

    if drug_name_col is None:
        for col in df.columns:
            if "drug" in str(col).lower():
                drug_name_col = col
                break

    if drug_name_col is None:
        log_print("CẢNH BÁO: Không tìm thấy cột DRUG_NAME trong drug metadata.", log_file)
        return None, {}

    df = df.copy()

    df[drug_name_col] = (
        df[drug_name_col]
        .astype(str)
        .str.strip()
    )

    valid_drug_mask = (
        df[drug_name_col].notna()
        & (df[drug_name_col] != "")
        & (df[drug_name_col].str.lower() != "nan")
    )

    df = df.loc[valid_drug_mask].copy()
    df = df.drop_duplicates(subset=[drug_name_col], keep="first")

    for col_name in ["DRUG_ID", "DRUG_NAME", "TARGET", "TARGET_PATHWAY", "SYNONYMS"]:
        real_col = find_column_case_insensitive(df.columns, col_name)

        if real_col is not None:
            df[real_col] = df[real_col].astype(str).str.strip()

            if col_name in ["TARGET", "TARGET_PATHWAY"]:
                df[real_col] = df[real_col].replace(
                    {
                        "": "Unknown",
                        "nan": "Unknown",
                        "NaN": "Unknown",
                    }
                )

    log_print(f"Số thuốc metadata sau xử lý: {len(df)}", log_file)

    if SAVE_INTERMEDIATE_FILES:
        save_dataframe(df, paths["drug_metadata_clean"], log_file, index=False)

    return df, {
        "drug_metadata_rows": len(df),
        "drug_metadata_drug_column": drug_name_col,
    }


# ==========================================================
# 7. TẠO DATASET CUỐI CÙNG
# ==========================================================

def validate_y_mask_alignment(
    y_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    log_file: Path,
) -> bool:
    """
    Kiểm tra y và mask có cùng shape, index, columns.
    """
    if y_df.shape != mask_df.shape:
        log_print("LỖI: y và mask không cùng shape.", log_file)
        return False

    if not y_df.index.equals(mask_df.index):
        log_print("LỖI: y và mask không cùng index.", log_file)
        return False

    if not y_df.columns.equals(mask_df.columns):
        log_print("LỖI: y và mask không cùng columns.", log_file)
        return False

    return True


def create_final_dataset(
    X_full: pd.DataFrame,
    y_full: pd.DataFrame,
    mask_full: pd.DataFrame,
    config: dict,
    log_file: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict]:
    """
    Tạo dataset cuối cùng:
    - lấy giao cell line giữa X và y
    - lọc thuốc theo observed
    - chọn top thuốc
    - lọc cell line theo observed
    """
    log_print("=" * 70, log_file)
    log_print("PHẦN 5: TẠO DATASET CUỐI CÙNG", log_file)
    log_print("=" * 70, log_file)

    X_full = X_full.copy()
    y_full = y_full.copy()
    mask_full = mask_full.copy()

    X_full.index = X_full.index.astype(str).str.strip()
    y_full.index = y_full.index.astype(str).str.strip()
    mask_full.index = mask_full.index.astype(str).str.strip()

    y_full.columns = y_full.columns.astype(str).str.strip()
    mask_full.columns = mask_full.columns.astype(str).str.strip()

    if not validate_y_mask_alignment(y_full, mask_full, log_file):
        return None, None, None, {}

    common_cell_lines = sorted(
        set(X_full.index).intersection(set(y_full.index))
    )

    log_print(f"Số cell line trong X: {X_full.shape[0]}", log_file)
    log_print(f"Số cell line trong y: {y_full.shape[0]}", log_file)
    log_print(f"Số cell line giao nhau: {len(common_cell_lines)}", log_file)

    if len(common_cell_lines) == 0:
        log_print("LỖI: Không có cell line giao nhau giữa X và y.", log_file)
        log_print(
            "Gợi ý: index của X_expression_full phải là mã cell line SIDM sau transpose.",
            log_file,
        )
        return None, None, None, {}

    X = X_full.loc[common_cell_lines].copy()
    y = y_full.loc[common_cell_lines].copy()
    mask = mask_full.loc[common_cell_lines].copy()

    if FILTER_BREAST_ONLY:
        log_print(
            "CẢNH BÁO: FILTER_BREAST_ONLY=True nhưng pipeline hiện tại không dùng metadata mô hình. "
            "Tiếp tục với toàn bộ cell line giao nhau.",
            log_file,
        )

    if X.shape[0] == 0:
        log_print("LỖI: Không còn cell line sau lọc.", log_file)
        return None, None, None, {}

    observed_per_drug = mask.sum(axis=0)

    valid_drugs = observed_per_drug[
        observed_per_drug >= MIN_OBSERVED_CELL_LINES_PER_DRUG
    ].index.tolist()

    y = y.loc[:, valid_drugs]
    mask = mask.loc[:, valid_drugs]

    if y.shape[1] == 0:
        log_print("LỖI: Không còn thuốc sau lọc observed cell lines.", log_file)
        return None, None, None, {}

    observed_per_drug = mask.sum(axis=0).sort_values(ascending=False)

    if config["top_n_drugs"] <= 0:
        selected_drugs = observed_per_drug.index.tolist()
    elif config["top_n_drugs"] > len(observed_per_drug):
        log_print(
            f"CẢNH BÁO: TOP_N_DRUGS={config['top_n_drugs']} lớn hơn số thuốc còn lại "
            f"({len(observed_per_drug)}). Lấy toàn bộ thuốc.",
            log_file,
        )
        selected_drugs = observed_per_drug.index.tolist()
    else:
        selected_drugs = observed_per_drug.head(config["top_n_drugs"]).index.tolist()

    y = y.loc[:, selected_drugs]
    mask = mask.loc[:, selected_drugs]

    observed_per_cell_line = mask.sum(axis=1)

    valid_cell_lines = observed_per_cell_line[
        observed_per_cell_line >= MIN_OBSERVED_DRUGS_PER_CELL_LINE
    ].index.tolist()

    X = X.loc[valid_cell_lines]
    y = y.loc[valid_cell_lines]
    mask = mask.loc[valid_cell_lines]

    if X.shape[0] == 0:
        log_print("LỖI: Không còn cell line sau lọc observed drugs.", log_file)
        return None, None, None, {}

    final_ids = sorted(X.index.tolist())

    X = X.loc[final_ids]
    y = y.loc[final_ids]
    mask = mask.loc[final_ids]

    X = X.apply(pd.to_numeric, errors="coerce")

    missing_X = int(X.isna().sum().sum())

    if missing_X > 0:
        log_print(f"CẢNH BÁO: X cuối còn {missing_X} missing. Fill median theo gene rồi fill 0.", log_file)
        medians = X.median(axis=0, skipna=True)
        X = X.fillna(medians).fillna(0)

    mask = mask.astype(int)

    mask_values = set(pd.unique(mask.to_numpy().ravel()))

    if not mask_values.issubset({0, 1}):
        log_print("CẢNH BÁO: mask có giá trị khác 0/1. Tạo lại mask từ y.", log_file)
        mask = y.notna().astype(int)

    final_observed_values = int(mask.sum().sum())
    final_total_values = int(mask.shape[0] * mask.shape[1])
    final_missing_rate = (
        1 - final_observed_values / final_total_values
        if final_total_values > 0
        else np.nan
    )

    log_print(f"Số cell line cuối cùng: {X.shape[0]}", log_file)
    log_print(f"Số gene cuối cùng: {X.shape[1]}", log_file)
    log_print(f"Số thuốc cuối cùng: {y.shape[1]}", log_file)
    log_print(f"Observed values cuối cùng: {final_observed_values}", log_file)
    log_print(f"Missing rate cuối cùng: {final_missing_rate:.4f}", log_file)

    stats = {
        "common_cell_lines": len(common_cell_lines),
        "final_cell_lines": X.shape[0],
        "final_genes": X.shape[1],
        "final_drugs": y.shape[1],
        "top_n_drugs_used": len(selected_drugs),
        "final_observed_values": final_observed_values,
        "final_missing_rate": final_missing_rate,
    }

    return X, y, mask, stats


# ==========================================================
# 8. SPLIT DATASET
# ==========================================================

def summarize_y_mask(y: pd.DataFrame, mask: pd.DataFrame) -> dict:
    """
    Tóm tắt observed/missing của y và mask.
    """
    observed_values = int(mask.sum().sum())
    total_values = int(mask.shape[0] * mask.shape[1])

    missing_rate = (
        1 - observed_values / total_values
        if total_values > 0
        else np.nan
    )

    return {
        "observed_values": observed_values,
        "missing_rate": missing_rate,
    }


def split_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    mask: pd.DataFrame,
    paths: dict,
    config: dict,
    log_file: Path,
) -> dict:
    """
    Chia dataset theo cell line thành train/validation/test.
    """
    if not DO_SPLIT:
        return {
            "total_samples": X.shape[0],
            "total_genes": X.shape[1],
            "total_drugs": y.shape[1],
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "split_status": "skipped",
        }

    log_print("=" * 70, log_file)
    log_print("PHẦN 6: SPLIT DATASET", log_file)
    log_print("=" * 70, log_file)

    cell_lines = X.index.tolist()

    if len(cell_lines) < 10:
        log_print("LỖI: Số mẫu quá ít để chia train/validation/test.", log_file)
        log_print(f"Tổng số mẫu hiện có: {len(cell_lines)}", log_file)

        return {
            "total_samples": len(cell_lines),
            "total_genes": X.shape[1],
            "total_drugs": y.shape[1],
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "split_status": "failed_too_few_samples",
        }

    train_ids, temp_ids = train_test_split(
        cell_lines,
        train_size=config["train_ratio"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    temp_ratio = config["val_ratio"] + config["test_ratio"]
    val_ratio_in_temp = config["val_ratio"] / temp_ratio if temp_ratio > 0 else 0.5

    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_ratio_in_temp,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    test_ids = sorted(test_ids)

    overlap_train_val = set(train_ids).intersection(val_ids)
    overlap_train_test = set(train_ids).intersection(test_ids)
    overlap_val_test = set(val_ids).intersection(test_ids)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        log_print("LỖI: Có cell line trùng giữa train/validation/test.", log_file)

        return {
            "total_samples": len(cell_lines),
            "total_genes": X.shape[1],
            "total_drugs": y.shape[1],
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "split_status": "failed_overlap",
        }

    total_after_split = len(train_ids) + len(val_ids) + len(test_ids)

    if total_after_split != len(cell_lines):
        log_print("LỖI: Tổng số mẫu sau split không bằng tổng ban đầu.", log_file)

        return {
            "total_samples": len(cell_lines),
            "total_genes": X.shape[1],
            "total_drugs": y.shape[1],
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "split_status": "failed_total_mismatch",
        }

    X_train = X.loc[train_ids]
    y_train = y.loc[train_ids]
    mask_train = mask.loc[train_ids]

    X_val = X.loc[val_ids]
    y_val = y.loc[val_ids]
    mask_val = mask.loc[val_ids]

    X_test = X.loc[test_ids]
    y_test = y.loc[test_ids]
    mask_test = mask.loc[test_ids]

    save_dataframe(X_train, paths["X_train"], log_file)
    save_dataframe(y_train, paths["y_train"], log_file)
    save_dataframe(mask_train, paths["mask_train"], log_file)

    save_dataframe(X_val, paths["X_val"], log_file)
    save_dataframe(y_val, paths["y_val"], log_file)
    save_dataframe(mask_val, paths["mask_val"], log_file)

    save_dataframe(X_test, paths["X_test"], log_file)
    save_dataframe(y_test, paths["y_test"], log_file)
    save_dataframe(mask_test, paths["mask_test"], log_file)

    train_stats = summarize_y_mask(y_train, mask_train)
    val_stats = summarize_y_mask(y_val, mask_val)
    test_stats = summarize_y_mask(y_test, mask_test)

    log_print(f"Số mẫu train: {len(train_ids)}", log_file)
    log_print(f"Số mẫu validation: {len(val_ids)}", log_file)
    log_print(f"Số mẫu test: {len(test_ids)}", log_file)

    log_print(f"Observed values train: {train_stats['observed_values']}", log_file)
    log_print(f"Observed values validation: {val_stats['observed_values']}", log_file)
    log_print(f"Observed values test: {test_stats['observed_values']}", log_file)

    log_print(f"Missing rate train: {train_stats['missing_rate']:.4f}", log_file)
    log_print(f"Missing rate validation: {val_stats['missing_rate']:.4f}", log_file)
    log_print(f"Missing rate test: {test_stats['missing_rate']:.4f}", log_file)

    split_summary = {
        "total_samples": len(cell_lines),
        "total_genes": X.shape[1],
        "total_drugs": y.shape[1],
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "train_observed_values": train_stats["observed_values"],
        "val_observed_values": val_stats["observed_values"],
        "test_observed_values": test_stats["observed_values"],
        "train_missing_rate": train_stats["missing_rate"],
        "val_missing_rate": val_stats["missing_rate"],
        "test_missing_rate": test_stats["missing_rate"],
        "split_status": "success",
    }

    save_dataframe(
        pd.DataFrame([split_summary]),
        paths["split_summary"],
        log_file,
        index=False,
    )

    log_print(f"Đã lưu split_summary: {paths['split_summary']}", log_file)

    return split_summary


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Hàm chính chạy toàn bộ data processing pipeline.
    """
    paths = setup_paths()
    log_file = paths["log_file"]
    start_time = datetime.now()

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU DATA PROCESSING", log_file)
    log_print("=" * 70, log_file)

    config = validate_config(log_file)
    print_config(config, paths, log_file)

    if not check_raw_files_for_pipeline(paths, log_file):
        log_print("CHƯƠNG TRÌNH DỪNG DO THIẾU FILE BẮT BUỘC.", log_file)
        return

    expression_df = load_expression_data(
        paths["expression_file"],
        log_file,
    )

    if expression_df is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO LỖI EXPRESSION.", log_file)
        return

    try:
        X_full, expression_stats_1 = prepare_expression_matrix(
            df=expression_df,
            log_file=log_file,
        )

        X_full, expression_stats_2 = clean_expression_matrix(
            expression_matrix=X_full,
            config=config,
            log_file=log_file,
        )

    except Exception as e:
        log_print("=" * 70, log_file)
        log_print("LỖI TRONG QUÁ TRÌNH XỬ LÝ EXPRESSION", log_file)
        log_print("=" * 70, log_file)
        log_print(f"Loại lỗi: {type(e).__name__}", log_file)
        log_print(f"Nội dung lỗi: {e}", log_file)
        return

    drug_response_df = load_drug_response_data(
        paths["drug_response_file"],
        log_file,
    )

    if drug_response_df is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO LỖI DRUG RESPONSE.", log_file)
        return

    y_full, mask_full, drug_response_stats = prepare_drug_response_matrix(
        df=drug_response_df,
        log_file=log_file,
    )

    if y_full is None or mask_full is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO LỖI TẠO y/mask.", log_file)
        return

    if SAVE_INTERMEDIATE_FILES:
        save_dataframe(X_full, paths["X_expression_full"], log_file)
        save_dataframe(y_full, paths["y_ic50_full"], log_file)
        save_dataframe(mask_full, paths["y_mask_full"], log_file)

    drug_metadata_df, drug_metadata_stats = prepare_drug_metadata(
        file_path=paths["drug_metadata_file"],
        paths=paths,
        log_file=log_file,
    )

    X, y, mask, dataset_stats = create_final_dataset(
        X_full=X_full,
        y_full=y_full,
        mask_full=mask_full,
        config=config,
        log_file=log_file,
    )

    if X is None or y is None or mask is None:
        log_print("CHƯƠNG TRÌNH DỪNG DO KHÔNG TẠO ĐƯỢC DATASET CUỐI.", log_file)
        return

    save_dataframe(X, paths["X_expression"], log_file)
    save_dataframe(y, paths["y_ic50"], log_file)
    save_dataframe(mask, paths["y_mask"], log_file)

    save_list_to_txt(X.index.tolist(), paths["selected_cell_lines"], log_file)
    save_list_to_txt(y.columns.tolist(), paths["selected_drugs"], log_file)

    split_stats = split_dataset(
        X=X,
        y=y,
        mask=mask,
        paths=paths,
        config=config,
        log_file=log_file,
    )

    if DO_SPLIT and split_stats.get("split_status") != "success":
        log_print("CHƯƠNG TRÌNH DỪNG DO SPLIT DATASET KHÔNG THÀNH CÔNG.", log_file)
        return

    end_time = datetime.now()
    elapsed_seconds = (end_time - start_time).total_seconds()

    summary_data = {
        "expression_file": str(paths["expression_file"]),
        "drug_response_file": str(paths["drug_response_file"]),
        "drug_metadata_file": str(paths["drug_metadata_file"]),
        "filter_breast_only": FILTER_BREAST_ONLY,
        "top_n_drugs_config": TOP_N_DRUGS,
        "top_n_drugs_used": dataset_stats.get("top_n_drugs_used", 0),
        "min_observed_drugs_per_cell_line": MIN_OBSERVED_DRUGS_PER_CELL_LINE,
        "min_observed_cell_lines_per_drug": MIN_OBSERVED_CELL_LINES_PER_DRUG,
        "expression_raw_shape": expression_stats_1.get("expression_raw_shape"),
        "expression_after_clean_shape": expression_stats_2.get("expression_after_clean_shape"),
        "drug_response_raw_shape": drug_response_stats.get("drug_response_raw_shape"),
        "y_full_shape": drug_response_stats.get("y_full_shape"),
        "common_cell_lines": dataset_stats.get("common_cell_lines"),
        "final_cell_lines": dataset_stats.get("final_cell_lines"),
        "final_genes": dataset_stats.get("final_genes"),
        "final_drugs": dataset_stats.get("final_drugs"),
        "final_observed_values": dataset_stats.get("final_observed_values"),
        "final_missing_rate": dataset_stats.get("final_missing_rate"),
        "train_ratio": config["train_ratio"],
        "val_ratio": config["val_ratio"],
        "test_ratio": config["test_ratio"],
        "train_samples": split_stats.get("train_samples", 0),
        "val_samples": split_stats.get("val_samples", 0),
        "test_samples": split_stats.get("test_samples", 0),
        "processing_status": "success",
    }

    save_summary(
        summary_data=summary_data,
        output_file=paths["data_processing_summary"],
        log_file=log_file,
    )

    log_print("=" * 70, log_file)
    log_print("HOÀN THÀNH DATA PROCESSING", log_file)
    log_print("=" * 70, log_file)

    log_print(f"X_expression: {paths['X_expression']}", log_file)
    log_print(f"y_ic50: {paths['y_ic50']}", log_file)
    log_print(f"y_mask: {paths['y_mask']}", log_file)
    log_print(f"selected_cell_lines: {paths['selected_cell_lines']}", log_file)
    log_print(f"selected_drugs: {paths['selected_drugs']}", log_file)
    log_print(f"summary: {paths['data_processing_summary']}", log_file)

    if DO_SPLIT:
        log_print(f"split_summary: {paths['split_summary']}", log_file)
        log_print(f"train dir: {paths['train_dir']}", log_file)
        log_print(f"validation dir: {paths['val_dir']}", log_file)
        log_print(f"test dir: {paths['test_dir']}", log_file)

    log_print(f"Log file: {log_file}", log_file)
    log_print(f"Thời gian hoàn thành: {elapsed_seconds:.2f} giây", log_file)


if __name__ == "__main__":
    main()