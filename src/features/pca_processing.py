from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


# ==========================================================
# 1. CẤU HÌNH Ở ĐẦU FILE
# ==========================================================

PCA_MODE = "fixed"

N_COMPONENTS = 128
PCA_VARIANCE_THRESHOLD = 0.95

RANDOM_STATE = 42

SAVE_PCA_FILES = True
SAVE_PCA_MODEL = True
SAVE_EXPLAINED_VARIANCE = True

OVERWRITE_OUTPUTS = True
SAVE_LOG = True


# ==========================================================
# 2. PROJECT ROOT + PATHS
# ==========================================================

def get_project_root() -> Path:
    """
    Tự động xác định PROJECT_ROOT.

    File hiện tại:
        src/features/pca_processing.py

    PROJECT_ROOT:
        parents[2]
    """
    return Path(__file__).resolve().parents[2]


def setup_paths() -> dict:
    """
    Thiết lập toàn bộ đường dẫn.
    Không hard-code đường dẫn tuyệt đối.
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

    log_file = logs_dir / f"pca_processing_{run_time}.log"

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
        "X_train_scaled": train_dir / "X_train_scaled.csv",
        "X_val_scaled": val_dir / "X_val_scaled.csv",
        "X_test_scaled": test_dir / "X_test_scaled.csv",

        # OUTPUT PCA FILES
        "X_train_pca": train_dir / "X_train_pca.csv",
        "X_val_pca": val_dir / "X_val_pca.csv",
        "X_test_pca": test_dir / "X_test_pca.csv",

        # MODEL
        "pca_model": models_dir / "pca.pkl",

        # EXPLAINED VARIANCE
        "explained_variance": processed_dir / "pca_explained_variance.csv",

        # SUMMARY
        "summary": processed_dir / "pca_processing_summary.csv",
    }


# ==========================================================
# 3. LOG
# ==========================================================

def log_print(message: str = "", log_file: Path | None = None) -> None:
    """
    Vừa print terminal vừa ghi log.
    """
    print(message)

    if SAVE_LOG and log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


# ==========================================================
# 4. CONFIG
# ==========================================================

def validate_config(
    max_components_allowed: int,
    log_file: Path,
) -> dict:
    """
    Validate cấu hình PCA.
    """
    pca_mode = PCA_MODE.lower()

    if pca_mode not in ["fixed", "variance"]:
        log_print(
            f"CẢNH BÁO: PCA_MODE='{PCA_MODE}' không hợp lệ. "
            "Sẽ dùng mode='fixed'.",
            log_file,
        )
        pca_mode = "fixed"

    n_components = N_COMPONENTS

    if n_components <= 0:
        log_print(
            "CẢNH BÁO: N_COMPONENTS <= 0. "
            "Sẽ dùng max_components_allowed.",
            log_file,
        )
        n_components = max_components_allowed

    if n_components > max_components_allowed:
        log_print(
            f"CẢNH BÁO: N_COMPONENTS={n_components} "
            f"lớn hơn max_components_allowed={max_components_allowed}. "
            "Sẽ dùng max_components_allowed.",
            log_file,
        )
        n_components = max_components_allowed

    variance_threshold = PCA_VARIANCE_THRESHOLD

    if not (0 < variance_threshold <= 1):
        log_print(
            "CẢNH BÁO: PCA_VARIANCE_THRESHOLD phải nằm trong khoảng (0, 1]. "
            "Sẽ dùng 0.95.",
            log_file,
        )
        variance_threshold = 0.95

    return {
        "pca_mode": pca_mode,
        "n_components": n_components,
        "variance_threshold": variance_threshold,
    }


def print_config(
    config: dict,
    paths: dict,
    log_file: Path,
) -> None:
    """
    In cấu hình PCA.
    """
    log_print("=" * 70, log_file)
    log_print("CẤU HÌNH PCA PROCESSING", log_file)
    log_print("=" * 70, log_file)

    log_print(f"PROJECT_ROOT: {paths['project_root']}", log_file)

    log_print(f"INPUT TRAIN: {paths['X_train_scaled']}", log_file)
    log_print(f"INPUT VALIDATION: {paths['X_val_scaled']}", log_file)
    log_print(f"INPUT TEST: {paths['X_test_scaled']}", log_file)

    log_print(f"PCA_MODE: {config['pca_mode']}", log_file)
    log_print(f"N_COMPONENTS: {config['n_components']}", log_file)
    log_print(
        f"PCA_VARIANCE_THRESHOLD: {config['variance_threshold']}",
        log_file,
    )

    log_print(f"SAVE_PCA_FILES: {SAVE_PCA_FILES}", log_file)
    log_print(f"SAVE_PCA_MODEL: {SAVE_PCA_MODEL}", log_file)
    log_print(
        f"SAVE_EXPLAINED_VARIANCE: {SAVE_EXPLAINED_VARIANCE}",
        log_file,
    )

    log_print(f"OVERWRITE_OUTPUTS: {OVERWRITE_OUTPUTS}", log_file)
    log_print(f"SAVE_LOG: {SAVE_LOG}", log_file)


# ==========================================================
# 5. FILE CHECK
# ==========================================================

def check_required_files(
    paths: dict,
    log_file: Path,
) -> bool:
    """
    Kiểm tra file input bắt buộc.
    """
    log_print("=" * 70, log_file)
    log_print("KIỂM TRA FILE INPUT", log_file)
    log_print("=" * 70, log_file)

    required_files = [
        paths["X_train_scaled"],
        paths["X_val_scaled"],
        paths["X_test_scaled"],
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

def read_feature_matrix(
    file_path: Path,
    log_file: Path,
) -> pd.DataFrame | None:
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
        log_print("LỖI ĐỌC FEATURE MATRIX", log_file)
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
# 7. VALIDATE DATA
# ==========================================================

def validate_feature_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    log_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Kiểm tra feature columns.
    """
    log_print("=" * 70, log_file)
    log_print("VALIDATE FEATURE COLUMNS", log_file)
    log_print("=" * 70, log_file)

    train_features = list(X_train.columns)

    val_missing = set(train_features) - set(X_val.columns)
    test_missing = set(train_features) - set(X_test.columns)

    if len(val_missing) > 0:
        log_print(
            f"LỖI: Validation thiếu {len(val_missing)} feature.",
            log_file,
        )
        return None

    if len(test_missing) > 0:
        log_print(
            f"LỖI: Test thiếu {len(test_missing)} feature.",
            log_file,
        )
        return None

    X_val = X_val.loc[:, train_features]
    X_test = X_test.loc[:, train_features]

    log_print(
        "Validation/Test đã được sắp xếp cùng thứ tự feature với train.",
        log_file,
    )

    return X_train, X_val, X_test


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép toàn bộ dữ liệu về numeric.
    """
    return df.apply(lambda col: pd.to_numeric(col, errors="coerce"))


def clean_missing_and_inf(
    df: pd.DataFrame,
    dataset_name: str,
    log_file: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Xử lý missing và inf trước PCA.
    """
    df = df.copy()

    inf_count = int(np.isinf(df.to_numpy()).sum())

    if inf_count > 0:
        log_print(
            f"CẢNH BÁO: {dataset_name} có {inf_count} giá trị inf.",
            log_file,
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing_before = int(df.isna().sum().sum())

    if missing_before > 0:
        log_print(
            f"CẢNH BÁO: {dataset_name} có {missing_before} missing values. "
            "Sẽ fill 0.",
            log_file,
        )

        df = df.fillna(0)

    missing_after = int(df.isna().sum().sum())

    stats = {
        "missing_before": missing_before,
        "inf_before": inf_count,
        "missing_after": missing_after,
    }

    return df, stats


# ==========================================================
# 8. PCA STRATEGY
# ==========================================================

def determine_pca_strategy(
    X_train: pd.DataFrame,
    config: dict,
    log_file: Path,
) -> dict:
    """
    Xác định chiến lược PCA.
    """
    n_train_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    max_components_allowed = min(
        n_train_samples - 1,
        n_features,
    )

    log_print("=" * 70, log_file)
    log_print("XÁC ĐỊNH PCA STRATEGY", log_file)
    log_print("=" * 70, log_file)

    log_print(f"n_train_samples: {n_train_samples}", log_file)
    log_print(f"n_features: {n_features}", log_file)
    log_print(
        f"max_components_allowed: {max_components_allowed}",
        log_file,
    )

    return {
        "max_components_allowed": max_components_allowed,
    }


def fit_pca_on_train(
    X_train: pd.DataFrame,
    config: dict,
    pca_strategy: dict,
    log_file: Path,
) -> tuple[PCA, dict]:
    """
    Fit PCA chỉ trên train.
    """
    log_print("=" * 70, log_file)
    log_print("FIT PCA TRÊN TRAIN", log_file)
    log_print("=" * 70, log_file)

    max_components_allowed = pca_strategy["max_components_allowed"]

    if config["pca_mode"] == "fixed":

        n_components_used = min(
            config["n_components"],
            max_components_allowed,
        )

        pca = PCA(
            n_components=n_components_used,
            random_state=RANDOM_STATE,
        )

        pca.fit(X_train)

        total_explained_variance_ratio = float(
            np.sum(pca.explained_variance_ratio_)
        )

        log_print(
            f"N_COMPONENTS cấu hình: {config['n_components']}",
            log_file,
        )

        log_print(
            f"n_components_used: {n_components_used}",
            log_file,
        )

        log_print(
            f"Fixed PCA giữ lại "
            f"{total_explained_variance_ratio:.6f} variance.",
            log_file,
        )

    else:

        pca = PCA(
            n_components=config["variance_threshold"],
            svd_solver="full",
            random_state=RANDOM_STATE,
        )

        pca.fit(X_train)

        n_components_used = int(pca.n_components_)

        total_explained_variance_ratio = float(
            np.sum(pca.explained_variance_ratio_)
        )

        log_print(
            f"PCA_VARIANCE_THRESHOLD: "
            f"{config['variance_threshold']}",
            log_file,
        )

        log_print(
            f"n_components_used: {n_components_used}",
            log_file,
        )

        log_print(
            f"Total explained variance ratio thực tế: "
            f"{total_explained_variance_ratio:.6f}",
            log_file,
        )

    stats = {
        "n_components_used": n_components_used,
        "total_explained_variance_ratio": total_explained_variance_ratio,
    }

    return pca, stats


# ==========================================================
# 9. TRANSFORM PCA
# ==========================================================

def transform_with_pca(
    pca: PCA,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Transform dataset bằng PCA đã fit trên train.
    """
    return pca.transform(df)


def make_pca_dataframe(
    pca_array: np.ndarray,
    index,
) -> pd.DataFrame:
    """
    Tạo DataFrame PCA.
    """
    n_components = pca_array.shape[1]

    columns = [
        f"PC{i+1}"
        for i in range(n_components)
    ]

    pca_df = pd.DataFrame(
        pca_array,
        index=index,
        columns=columns,
    )

    return pca_df


# ==========================================================
# 10. EXPLAINED VARIANCE
# ==========================================================

def compute_explained_variance_table(
    pca: PCA,
) -> pd.DataFrame:
    """
    Tạo bảng explained variance.
    """
    explained_variance = pca.explained_variance_

    explained_variance_ratio = pca.explained_variance_ratio_

    cumulative = np.cumsum(explained_variance_ratio)

    variance_df = pd.DataFrame({
        "component": [
            f"PC{i+1}"
            for i in range(len(explained_variance))
        ],
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": cumulative,
    })

    return variance_df


def find_components_for_threshold(
    cumulative_variance: np.ndarray,
    threshold: float,
):
    """
    Tìm số component cần để đạt threshold variance.
    """
    indices = np.where(cumulative_variance >= threshold)[0]

    if len(indices) == 0:
        return "not_reached"

    return int(indices[0] + 1)


# ==========================================================
# 11. CHECK OUTPUTS
# ==========================================================

def check_pca_outputs(
    df: pd.DataFrame,
    dataset_name: str,
    log_file: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Kiểm tra missing/inf sau PCA.
    """
    df = df.copy()

    inf_count = int(np.isinf(df.to_numpy()).sum())

    if inf_count > 0:
        log_print(
            f"CẢNH BÁO: {dataset_name} có {inf_count} giá trị inf sau PCA.",
            log_file,
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing_count = int(df.isna().sum().sum())

    if missing_count > 0:
        log_print(
            f"CẢNH BÁO: {dataset_name} có {missing_count} missing sau PCA. "
            "Sẽ fill 0.",
            log_file,
        )

        df = df.fillna(0)

    final_missing = int(df.isna().sum().sum())
    final_inf = int(np.isinf(df.to_numpy()).sum())

    stats = {
        "missing": final_missing,
        "inf": final_inf,
    }

    return df, stats


# ==========================================================
# 12. SAVE
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
            log_print(
                f"Ghi đè file: {output_file}",
                log_file,
            )

        else:
            log_print(
                f"CẢNH BÁO: File đã tồn tại và OVERWRITE_OUTPUTS=False: "
                f"{output_file}",
                log_file,
            )
            return False

    df.to_csv(output_file)

    return True


def save_outputs(
    paths: dict,
    X_train_pca: pd.DataFrame,
    X_val_pca: pd.DataFrame,
    X_test_pca: pd.DataFrame,
    pca: PCA,
    variance_df: pd.DataFrame,
    log_file: Path,
) -> None:
    """
    Lưu output PCA.
    """
    if SAVE_PCA_FILES:

        save_dataframe(
            X_train_pca,
            paths["X_train_pca"],
            log_file,
        )

        save_dataframe(
            X_val_pca,
            paths["X_val_pca"],
            log_file,
        )

        save_dataframe(
            X_test_pca,
            paths["X_test_pca"],
            log_file,
        )

    if SAVE_PCA_MODEL:

        joblib.dump(
            pca,
            paths["pca_model"],
        )

        log_print(
            f"Đã lưu PCA model: {paths['pca_model']}",
            log_file,
        )

    if SAVE_EXPLAINED_VARIANCE:

        variance_df.to_csv(
            paths["explained_variance"],
            index=False,
        )

        log_print(
            f"Đã lưu explained variance: "
            f"{paths['explained_variance']}",
            log_file,
        )


def save_summary(
    summary_data: dict,
    output_file: Path,
    log_file: Path,
) -> None:
    """
    Lưu summary CSV.
    """
    summary_df = pd.DataFrame([summary_data])

    summary_df.to_csv(
        output_file,
        index=False,
    )

    log_print(
        f"Đã lưu summary: {output_file}",
        log_file,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    paths = setup_paths()
    log_file = paths["log_file"]

    log_print("=" * 70, log_file)
    log_print("BẮT ĐẦU PCA PROCESSING", log_file)
    log_print("=" * 70, log_file)

    if not check_required_files(paths, log_file):
        log_print(
            "CHƯƠNG TRÌNH DỪNG DO THIẾU FILE INPUT.",
            log_file,
        )
        return

    log_print("=" * 70, log_file)
    log_print("LOAD DATA", log_file)
    log_print("=" * 70, log_file)

    X_train = read_feature_matrix(
        paths["X_train_scaled"],
        log_file,
    )

    X_val = read_feature_matrix(
        paths["X_val_scaled"],
        log_file,
    )

    X_test = read_feature_matrix(
        paths["X_test_scaled"],
        log_file,
    )

    if X_train is None or X_val is None or X_test is None:
        log_print(
            "CHƯƠNG TRÌNH DỪNG DO LỖI LOAD DATA.",
            log_file,
        )
        return

    X_train = standardize_index(X_train)
    X_val = standardize_index(X_val)
    X_test = standardize_index(X_test)

    log_print(f"Shape X_train_scaled: {X_train.shape}", log_file)
    log_print(f"Shape X_val_scaled: {X_val.shape}", log_file)
    log_print(f"Shape X_test_scaled: {X_test.shape}", log_file)

    log_print(f"Số samples train: {X_train.shape[0]}", log_file)
    log_print(f"Số samples validation: {X_val.shape[0]}", log_file)
    log_print(f"Số samples test: {X_test.shape[0]}", log_file)

    log_print(f"Số feature đầu vào: {X_train.shape[1]}", log_file)

    pca_strategy = determine_pca_strategy(
        X_train,
        {
            "pca_mode": PCA_MODE,
        },
        log_file,
    )

    config = validate_config(
        pca_strategy["max_components_allowed"],
        log_file,
    )

    print_config(
        config,
        paths,
        log_file,
    )

    validated = validate_feature_columns(
        X_train,
        X_val,
        X_test,
        log_file,
    )

    if validated is None:
        log_print(
            "CHƯƠNG TRÌNH DỪNG DO FEATURE COLUMNS KHÔNG HỢP LỆ.",
            log_file,
        )
        return

    X_train, X_val, X_test = validated

    X_train = convert_to_numeric(X_train)
    X_val = convert_to_numeric(X_val)
    X_test = convert_to_numeric(X_test)

    X_train, train_clean_stats = clean_missing_and_inf(
        X_train,
        "train",
        log_file,
    )

    X_val, val_clean_stats = clean_missing_and_inf(
        X_val,
        "validation",
        log_file,
    )

    X_test, test_clean_stats = clean_missing_and_inf(
        X_test,
        "test",
        log_file,
    )

    pca, pca_stats = fit_pca_on_train(
        X_train,
        config,
        pca_strategy,
        log_file,
    )

    log_print("=" * 70, log_file)
    log_print("TRANSFORM PCA", log_file)
    log_print("=" * 70, log_file)

    X_train_pca_array = transform_with_pca(
        pca,
        X_train,
    )

    X_val_pca_array = transform_with_pca(
        pca,
        X_val,
    )

    X_test_pca_array = transform_with_pca(
        pca,
        X_test,
    )

    X_train_pca = make_pca_dataframe(
        X_train_pca_array,
        X_train.index,
    )

    X_val_pca = make_pca_dataframe(
        X_val_pca_array,
        X_val.index,
    )

    X_test_pca = make_pca_dataframe(
        X_test_pca_array,
        X_test.index,
    )

    X_train_pca, train_pca_stats = check_pca_outputs(
        X_train_pca,
        "train",
        log_file,
    )

    X_val_pca, val_pca_stats = check_pca_outputs(
        X_val_pca,
        "validation",
        log_file,
    )

    X_test_pca, test_pca_stats = check_pca_outputs(
        X_test_pca,
        "test",
        log_file,
    )

    variance_df = compute_explained_variance_table(pca)

    cumulative = variance_df[
        "cumulative_explained_variance_ratio"
    ].to_numpy()

    components_50 = find_components_for_threshold(cumulative, 0.50)
    components_70 = find_components_for_threshold(cumulative, 0.70)
    components_80 = find_components_for_threshold(cumulative, 0.80)
    components_90 = find_components_for_threshold(cumulative, 0.90)
    components_95 = find_components_for_threshold(cumulative, 0.95)
    components_99 = find_components_for_threshold(cumulative, 0.99)

    pc1_ratio = float(
        variance_df.iloc[0]["explained_variance_ratio"]
    )

    total_variance_ratio = float(
        variance_df["explained_variance_ratio"].sum()
    )

    log_print("=" * 70, log_file)
    log_print("EXPLAINED VARIANCE", log_file)
    log_print("=" * 70, log_file)

    log_print(
        f"Explained variance ratio của PC1: "
        f"{pc1_ratio:.6f}",
        log_file,
    )

    log_print(
        f"Tổng explained variance ratio: "
        f"{total_variance_ratio:.6f}",
        log_file,
    )

    log_print(
        f"Components đạt 50% variance: {components_50}",
        log_file,
    )

    log_print(
        f"Components đạt 70% variance: {components_70}",
        log_file,
    )

    log_print(
        f"Components đạt 80% variance: {components_80}",
        log_file,
    )

    log_print(
        f"Components đạt 90% variance: {components_90}",
        log_file,
    )

    log_print(
        f"Components đạt 95% variance: {components_95}",
        log_file,
    )

    log_print(
        f"Components đạt 99% variance: {components_99}",
        log_file,
    )

    save_outputs(
        paths,
        X_train_pca,
        X_val_pca,
        X_test_pca,
        pca,
        variance_df,
        log_file,
    )

    summary_data = {
        "input_train_file": str(paths["X_train_scaled"]),
        "input_val_file": str(paths["X_val_scaled"]),
        "input_test_file": str(paths["X_test_scaled"]),

        "output_train_file": str(paths["X_train_pca"]),
        "output_val_file": str(paths["X_val_pca"]),
        "output_test_file": str(paths["X_test_pca"]),

        "pca_model_file": str(paths["pca_model"]),
        "explained_variance_file": str(paths["explained_variance"]),

        "pca_mode": config["pca_mode"],
        "n_components_config": config["n_components"],
        "pca_variance_threshold": config["variance_threshold"],

        "n_components_used": pca_stats["n_components_used"],
        "max_components_allowed": pca_strategy["max_components_allowed"],

        "train_rows": X_train.shape[0],
        "train_columns": X_train.shape[1],

        "val_rows": X_val.shape[0],
        "val_columns": X_val.shape[1],

        "test_rows": X_test.shape[0],
        "test_columns": X_test.shape[1],

        "total_explained_variance_ratio": total_variance_ratio,
        "pc1_explained_variance_ratio": pc1_ratio,

        "components_for_50_percent_variance": components_50,
        "components_for_70_percent_variance": components_70,
        "components_for_80_percent_variance": components_80,
        "components_for_90_percent_variance": components_90,
        "components_for_95_percent_variance": components_95,
        "components_for_99_percent_variance": components_99,

        "train_missing_before_pca": train_clean_stats["missing_before"],
        "val_missing_before_pca": val_clean_stats["missing_before"],
        "test_missing_before_pca": test_clean_stats["missing_before"],

        "train_inf_before_pca": train_clean_stats["inf_before"],
        "val_inf_before_pca": val_clean_stats["inf_before"],
        "test_inf_before_pca": test_clean_stats["inf_before"],

        "train_missing_after_pca": train_pca_stats["missing"],
        "val_missing_after_pca": val_pca_stats["missing"],
        "test_missing_after_pca": test_pca_stats["missing"],

        "train_inf_after_pca": train_pca_stats["inf"],
        "val_inf_after_pca": val_pca_stats["inf"],
        "test_inf_after_pca": test_pca_stats["inf"],

        "final_train_shape": str(X_train_pca.shape),
        "final_val_shape": str(X_val_pca.shape),
        "final_test_shape": str(X_test_pca.shape),

        "processing_status": "success",
    }

    save_summary(
        summary_data,
        paths["summary"],
        log_file,
    )

    log_print("=" * 70, log_file)
    log_print("HOÀN THÀNH PCA PROCESSING", log_file)
    log_print("=" * 70, log_file)

    log_print(
        f"Log file: {log_file}",
        log_file,
    )


if __name__ == "__main__":
    main()