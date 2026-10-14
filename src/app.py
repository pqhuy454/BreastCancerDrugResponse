# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
from datetime import datetime
from typing import Any
import sys

import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. PROJECT IMPORT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.prediction import (
    read_selected_drugs,
    read_selected_genes,
    load_sklearn_model,
    load_dnn_model,
    load_scaler,
    load_pca,
    get_torch_device,
    predict_with_sklearn_model,
    predict_with_dnn_model,
    create_drug_ranking,
)

from src.spatial.spatial_processing import (
    preprocess_uploaded_spatial_dataset,
)

from src.predict.predict_spatial_baseline import (
    predict_uploaded_spatial_baseline,
    predict_all_baseline_models,
    create_mean_baseline_prediction as create_spatial_mean_baseline,
)

from src.spatial.build_spatial_graph import (
    build_graph_for_processed_dataset,
)

from src.predict.predict_spatial_gnn import (
    get_device as get_gnn_device,
    load_gnn_checkpoint,
    predict_uploaded_spatial_gnn,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

APP_TITLE = "Dự đoán đáp ứng thuốc ung thư vú"
APP_SUBTITLE = (
    "Mô hình Cell Line và tinh chỉnh GraphSAGE trên dữ liệu sinh học không gian"
)

TOP_K_DRUGS = 10
PAGE_ICON = "🧬"

DISCLAIMER_TEXT = (
    "⚠️ Kết quả chỉ phục vụ nghiên cứu và minh họa. "
    "Không sử dụng cho quyết định điều trị lâm sàng."
)

MODEL_OPTIONS = [
    "DNN",
    "Ridge",
    "XGBoost",
    "Trung bình 3 mô hình",
]


# ============================================================
# 3. PATH SETUP
# ============================================================

def get_project_root() -> Path:
    """
    Get project root.

    Returns
    -------
    Path
        Project root path.
    """
    return Path(__file__).resolve().parents[1]


def setup_app_paths() -> dict[str, Path]:
    """
    Setup application paths.

    Returns
    -------
    dict[str, Path]
        Dictionary of application paths.
    """
    project_root = get_project_root()

    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"
    test_dir = processed_dir / "test"
    demo_uploads_dir = project_root / "data" / "demo_uploads"
    demo_results_dir = project_root / "results" / "demo"

    demo_uploads_dir.mkdir(parents=True, exist_ok=True)
    demo_results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "models_dir": models_dir,
        "processed_dir": processed_dir,
        "test_dir": test_dir,
        "demo_uploads_dir": demo_uploads_dir,
        "demo_results_dir": demo_results_dir,
        "selected_genes_file": processed_dir / "selected_genes.txt",
        "selected_drugs_file": processed_dir / "selected_drugs.txt",
        "test_pca_file": test_dir / "X_test_pca.csv",
        "ridge_model_file": models_dir / "ridge_model.pkl",
        "xgboost_model_file": models_dir / "xgboost_model.pkl",
        "dnn_model_file": models_dir / "dnn_model.pkl",
        "scaler_file": models_dir / "scaler.pkl",
        "pca_file": models_dir / "pca.pkl",
        "gnn_model_file": models_dir / "gnn_ic50_refiner.pt",
    }


# ============================================================
# 4. CUSTOM CSS
# ============================================================

def inject_custom_css() -> None:
    """
    Inject custom CSS for Streamlit interface.
    """
    st.markdown(
        """
        <style>
            .stApp {
                background: #f5f7fb;
                color: #1f2937;
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid rgba(148, 163, 184, 0.25);
                min-width: 400px !important;
                max-width: 400px !important;
                width: 400px !important;
            }

            section[data-testid="stSidebar"] > div {
                background: #ffffff;
                min-width: 400px !important;
                max-width: 400px !important;
                width: 400px !important;
                padding-top: 1.0rem;
            }

            div[data-testid="stSidebarResizer"] {
                display: none !important;
            }

            div[data-testid="collapsedControl"] {
                display: none !important;
            }

            button[data-testid="stSidebarCollapseButton"] {
                display: none !important;
            }

            [data-testid="stSidebarCollapseButton"] {
                display: none !important;
            }

            .main-header {
                padding: 1.8rem 2rem;
                border-radius: 22px;
                background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                margin-bottom: 1.5rem;
                box-shadow: 0 12px 28px rgba(79, 70, 229, 0.18);
            }

            .main-title {
                font-size: 2.15rem;
                font-weight: 800;
                margin-bottom: 0.35rem;
                letter-spacing: -0.02em;
            }

            .main-subtitle {
                font-size: 1.0rem;
                opacity: 0.95;
                line-height: 1.55;
            }

            .content-card {
                background: white;
                border-radius: 18px;
                padding: 1.25rem 1.35rem;
                border: 1px solid rgba(148, 163, 184, 0.22);
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
            }

            .metric-card {
                background: white;
                border-radius: 18px;
                padding: 1rem 1.1rem;
                border: 1px solid rgba(148, 163, 184, 0.22);
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
                min-height: 95px;
            }

            .metric-label {
                font-size: 0.78rem;
                color: #6b7280;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.035em;
            }

            .metric-value {
                font-size: 1.25rem;
                color: #1f2937;
                font-weight: 800;
                margin-top: 0.35rem;
                word-break: break-word;
            }

            .note-box {
                padding: 0.95rem 1.05rem;
                border-left: 5px solid #4f46e5;
                background: #eef2ff;
                color: #1f2937;
                border-radius: 12px;
                margin: 0.8rem 0 1.1rem 0;
                font-size: 0.95rem;
                line-height: 1.6;
            }

            .warning-box {
                padding: 0.95rem 1.05rem;
                border-left: 5px solid #f59e0b;
                background: #fffbeb;
                color: #78350f;
                border-radius: 12px;
                margin: 1rem 0;
                font-size: 0.95rem;
                line-height: 1.6;
            }

            .sidebar-guide-card {
                background: #ffffff;
                border: 1px solid rgba(148, 163, 184, 0.28);
                border-radius: 16px;
                padding: 1rem 1rem;
                margin: 0.85rem 0;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
                color: #374151;
                line-height: 1.65;
                font-size: 0.92rem;
            }

            .sidebar-warning-card {
                background: #fffbeb;
                border: 1px solid rgba(245, 158, 11, 0.35);
                border-left: 5px solid #f59e0b;
                border-radius: 16px;
                padding: 0.95rem 1rem;
                margin-top: 1.1rem;
                color: #78350f;
                line-height: 1.55;
                font-size: 0.92rem;
            }

            div.stButton > button {
                border-radius: 14px;
                border: 0;
                padding: 0.72rem 1.1rem;
                font-weight: 750;
                color: white;
                background: linear-gradient(120deg, #16a34a, #22c55e);
                box-shadow: 0 8px 18px rgba(22, 163, 74, 0.24);
            }

            div.stButton > button:hover {
                color: white;
                background: linear-gradient(120deg, #15803d, #16a34a);
            }

            div[data-testid="stTabs"] button {
                font-weight: 750;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 14px !important;
                overflow: hidden !important;
                border: 2px solid #111827 !important;
                background: #ffffff !important;
            }

            div[data-testid="stDataFrame"] div {
                border-color: rgba(17, 24, 39, 0.28) !important;
            }

            div[data-testid="stImage"],
            div[data-testid="stPyplot"] {
                border: 2px solid #111827 !important;
                border-radius: 14px !important;
                background: #ffffff !important;
                padding: 10px !important;
                overflow: hidden !important;
            }

            div[data-testid="stImage"] img {
                border-radius: 10px !important;
            }


            .custom-table-wrapper {
                border: 2px solid #111827;
                border-radius: 14px;
                overflow-x: hidden;
                overflow-y: auto;
                background: #ffffff;
                height: 430px;
            }

            .custom-table-wrapper table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.92rem;
            }

            .custom-table-wrapper th {
                background: #f9fafb;
                color: #111827;
                font-weight: 800;
                text-align: left;
                padding: 0.72rem 0.75rem;
                border-bottom: 1.5px solid #111827;
                border-right: 1px solid rgba(17, 24, 39, 0.18);
            }

            .custom-table-wrapper td {
                padding: 0.68rem 0.75rem;
                border-bottom: 1px solid rgba(17, 24, 39, 0.13);
                border-right: 1px solid rgba(17, 24, 39, 0.13);
            }

            .custom-table-wrapper tr:last-child td {
                border-bottom: 0;
            }

            .custom-table-wrapper th:last-child,
            .custom-table-wrapper td:last-child {
                border-right: 0;
            }

            .custom-table-wrapper td:nth-child(1),
            .custom-table-wrapper td:nth-child(3),
            .custom-table-wrapper td:nth-child(4),
            .custom-table-wrapper td:nth-child(5) {
                text-align: right;
            }

            .custom-table-wrapper.compact {
                height: auto;
            }

            .sidebar-title {
                font-size: 1.25rem;
                font-weight: 850;
                color: #1f2937;
                margin-bottom: 0.5rem;
            }

            .sidebar-text {
                color: #374151;
                line-height: 1.65;
                font-size: 0.95rem;
            }

            .small-caption {
                color: #6b7280;
                font-size: 0.86rem;
                line-height: 1.5;
            }


            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
                max-width: 1480px !important;
            }

            h1, h2, h3 {
                color: #0f172a !important;
                letter-spacing: -0.02em;
            }

            .stFileUploader {
                background: #ffffff;
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 16px;
                padding: 0.65rem 0.75rem;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
            }

            .section-title-card {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.8rem;
                background: #ffffff;
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 16px;
                padding: 0.9rem 1rem;
                margin: 0.9rem 0 0.75rem 0;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
            }

            .section-title-main {
                font-size: 1.08rem;
                font-weight: 850;
                color: #0f172a;
            }

            .section-title-sub {
                font-size: 0.86rem;
                color: #64748b;
                margin-top: 0.15rem;
            }

            .plot-card {
                border: 2px solid #111827;
                border-radius: 14px;
                background: #ffffff;
                padding: 0.8rem;
                overflow: hidden;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
            }

            .spatial-two-column-card {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 20px;
                padding: 1rem;
                margin: 0.4rem 0 1rem 0;
            }

            .metric-card {
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            }

            .stSuccess {
                border-radius: 14px !important;
            }



            /* ---------- Polished global layout ---------- */
            .block-container {
                padding-top: 1.4rem !important;
                padding-bottom: 2.2rem !important;
                max-width: 1500px !important;
            }

            .main-header {
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.28);
            }

            .main-header::after {
                content: "";
                position: absolute;
                width: 260px;
                height: 260px;
                right: -70px;
                top: -90px;
                background: rgba(255, 255, 255, 0.12);
                border-radius: 50%;
            }

            .main-title {
                text-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
            }

            /* ---------- Better tabs ---------- */
            div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
                gap: 0.5rem;
                border-bottom: 1px solid rgba(148, 163, 184, 0.28);
                padding-bottom: 0.35rem;
            }

            div[data-testid="stTabs"] button[role="tab"] {
                background: #ffffff !important;
                border: 1px solid rgba(148, 163, 184, 0.25) !important;
                border-radius: 999px !important;
                padding: 0.45rem 0.9rem !important;
                color: #334155 !important;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
            }

            div[data-testid="stTabs"] button[aria-selected="true"] {
                background: linear-gradient(120deg, #ecfdf5, #eef2ff) !important;
                color: #ef4444 !important;
                border-color: rgba(239, 68, 68, 0.35) !important;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            }

            div[data-testid="stTabs"] button[role="tab"] p {
                font-weight: 800 !important;
            }

            /* ---------- Button refinement ---------- */
            div.stButton > button {
                min-height: 3.1rem;
                font-size: 0.95rem;
                letter-spacing: 0.01em;
                transition: all 0.18s ease;
            }

            div.stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 12px 26px rgba(22, 163, 74, 0.28);
            }

            /* ---------- Upload cards ---------- */
            .stFileUploader {
                min-height: 118px;
            }

            .stFileUploader label {
                font-weight: 750 !important;
                color: #0f172a !important;
            }

            /* ---------- Plot and table balance ---------- */
            div[data-testid="stPyplot"] {
                margin-top: 0 !important;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05) !important;
            }

            div[data-testid="stPyplot"] > div {
                background: #ffffff !important;
            }

            .custom-table-wrapper {
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            }

            .custom-table-wrapper.compact {
                max-height: 430px;
                overflow-y: auto;
            }

            /* ---------- Select/radio polish ---------- */
            div[data-baseweb="select"] > div {
                border-radius: 14px !important;
                background: #ffffff !important;
                border-color: rgba(148, 163, 184, 0.28) !important;
            }

            div[role="radiogroup"] label {
                background: #ffffff;
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 999px;
                padding: 0.28rem 0.58rem;
                margin-right: 0.2rem;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 5. BASIC UI HELPERS
# ============================================================

def render_header() -> None:
    """
    Render main header.
    """
    st.markdown(
        f"""
        <div class="main-header">
            <div class="main-title">🧬 {APP_TITLE}</div>
            <div class="main-subtitle">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note(message: str) -> None:
    """
    Render note box.

    Parameters
    ----------
    message : str
        Message.
    """
    st.markdown(
        f"""
        <div class="note-box">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning(message: str) -> None:
    """
    Render warning box.

    Parameters
    ----------
    message : str
        Message.
    """
    st.markdown(
        f"""
        <div class="warning-box">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_success(message: str) -> None:
    st.markdown(
        f"""
        <div style="
            background-color: #ecfdf5;
            border-left: 5px solid #10b981;
            padding: 14px 16px;
            border-radius: 12px;
            margin: 10px 0;
            color: #065f46;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str | None = None) -> None:
    """
    Render a polished section title card.
    """
    subtitle_html = f'<div class="section-title-sub">{subtitle}</div>' if subtitle else ""

    st.markdown(
        f"""
        <div class="section-title-card">
            <div>
                <div class="section-title-main">{title}</div>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plot_card_start() -> None:
    """
    Start a bordered plot card.
    """
    st.markdown(
        '<div class="plot-card">',
        unsafe_allow_html=True,
    )


def render_plot_card_end() -> None:
    """
    End a bordered plot card.
    """
    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any) -> None:
    """
    Render metric card.

    Parameters
    ----------
    label : str
        Metric label.
    value : Any
        Metric value.
    """
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. MODEL AND DATA LOADING
# ============================================================

def check_required_file(file_path: Path, label: str) -> None:
    """
    Check if required file exists.

    Parameters
    ----------
    file_path : Path
        File path.
    label : str
        File label.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Thiếu {label}: {file_path}")


@st.cache_data(show_spinner=False)
def load_test_pca(test_pca_file: Path) -> pd.DataFrame:
    """
    Load test PCA matrix.

    Parameters
    ----------
    test_pca_file : Path
        Path to X_test_pca.csv.

    Returns
    -------
    pd.DataFrame
        Test PCA dataframe.
    """
    if not test_pca_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu test: {test_pca_file}"
        )

    X_test_pca = pd.read_csv(test_pca_file, index_col=0)
    X_test_pca.index = X_test_pca.index.astype(str).str.strip()
    X_test_pca.columns = X_test_pca.columns.astype(str).str.strip()
    X_test_pca.index.name = "cell_line"

    X_test_pca = X_test_pca.apply(pd.to_numeric, errors="coerce")
    X_test_pca = X_test_pca.replace([np.inf, -np.inf], np.nan)
    X_test_pca = X_test_pca.fillna(0.0)

    return X_test_pca


@st.cache_resource(show_spinner=True)
def load_all_models(project_root: Path) -> dict[str, Any]:
    """
    Load trained models and preprocessing objects.

    Parameters
    ----------
    project_root : Path
        Project root path.

    Returns
    -------
    dict[str, Any]
        Loaded objects.
    """
    paths = setup_app_paths()

    required_files = {
        "selected_genes.txt": paths["selected_genes_file"],
        "selected_drugs.txt": paths["selected_drugs_file"],
        "ridge_model.pkl": paths["ridge_model_file"],
        "xgboost_model.pkl": paths["xgboost_model_file"],
        "dnn_model.pkl": paths["dnn_model_file"],
        "scaler.pkl": paths["scaler_file"],
        "pca.pkl": paths["pca_file"],
        "gnn_ic50_refiner.pt": paths["gnn_model_file"],
    }

    for label, file_path in required_files.items():
        check_required_file(file_path=file_path, label=label)

    selected_genes = read_selected_genes(paths["selected_genes_file"])
    selected_drugs = read_selected_drugs(paths["selected_drugs_file"])

    ridge_model = load_sklearn_model(paths["ridge_model_file"])
    xgboost_model = load_sklearn_model(paths["xgboost_model_file"])
    scaler = load_scaler(paths["scaler_file"])
    pca = load_pca(paths["pca_file"])

    try:
        X_test_pca = load_test_pca(paths["test_pca_file"])
        dnn_input_dim = int(X_test_pca.shape[1])
    except Exception:
        dnn_input_dim = 128

    torch_device = get_torch_device("auto")

    dnn_model = load_dnn_model(
        model_file=paths["dnn_model_file"],
        input_dim=dnn_input_dim,
        output_dim=len(selected_drugs),
        device=torch_device,
    )

    gnn_device = get_gnn_device("auto")

    gnn_model, gnn_checkpoint = load_gnn_checkpoint(
        model_file=paths["gnn_model_file"],
        device=gnn_device,
        log_file=None,
    )

    baseline_models = {
        "ridge": ridge_model,
        "xgboost": xgboost_model,
        "dnn": dnn_model,
    }

    return {
        "paths": paths,
        "selected_genes": selected_genes,
        "selected_drugs": selected_drugs,
        "ridge_model": ridge_model,
        "xgboost_model": xgboost_model,
        "dnn_model": dnn_model,
        "baseline_models": baseline_models,
        "scaler": scaler,
        "pca": pca,
        "torch_device": torch_device,
        "gnn_device": gnn_device,
        "gnn_model": gnn_model,
        "gnn_checkpoint": gnn_checkpoint,
    }


# ============================================================
# 7. SIDEBAR
# ============================================================

def render_sidebar() -> None:
    """
    Render sidebar usage guide.
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-title">📘 Hướng dẫn sử dụng</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-guide-card">
                <b>Tab 1: Dự đoán Cell Line</b><br>
                1. Chọn Cell Line.<br>
                2. Chọn mô hình dự đoán.<br>
                3. Nhấn nút dự đoán.<br>
                4. Xem Top 10 thuốc nhạy nhất.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-guide-card">
                <b>Tab 2: Dự đoán dữ liệu sinh học không gian</b><br>
                1. Tải file .h5.<br>
                2. Tải file tissue_positions.csv.<br>
                3. Nhấn chạy dự đoán.<br>
                4. Xem kết quả GNN.<br>
                5. Xem Top 10 thuốc toàn bộ mẫu.<br>
                6. Xem Top 10 thuốc theo từng Spot.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-warning-card">
                ⚠️ Kết quả chỉ phục vụ nghiên cứu và minh họa.<br>
                Không sử dụng cho quyết định điều trị lâm sàng.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# 8. DATAFRAME FORMATTING
# ============================================================

def format_cellline_top10_table(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format top10 table for cell line tab.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Ranking dataframe.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe.
    """
    table_df = ranking_df.copy()

    rename_map = {
        "rank": "Hạng",
        "drug": "Tên thuốc",
        "predicted_LN_IC50": "LN_IC50 dự đoán",
    }

    table_df = table_df.rename(columns=rename_map)

    keep_columns = [
        "Hạng",
        "Tên thuốc",
        "LN_IC50 dự đoán",
    ]

    table_df = table_df[keep_columns].copy()

    return table_df


def format_overall_top10_table(overall_top10_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format overall top10 table.

    Parameters
    ----------
    overall_top10_df : pd.DataFrame
        Overall top10 dataframe.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe.
    """
    table_df = overall_top10_df.copy()

    table_df = table_df.rename(
        columns={
            "rank": "Hạng",
            "drug": "Tên thuốc",
            "mean_refined_LN_IC50": "Mean Refined LN_IC50",
            "median_refined_LN_IC50": "Median Refined LN_IC50",
            "min_refined_LN_IC50": "Min Refined LN_IC50",
            "max_refined_LN_IC50": "Max Refined LN_IC50",
            "std_refined_LN_IC50": "Std Refined LN_IC50",
        }
    )

    return table_df


def format_per_spot_top10_table(top10_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format per spot top10 table.

    Parameters
    ----------
    top10_df : pd.DataFrame
        Top10 dataframe.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe.
    """
    table_df = top10_df.copy()

    table_df = table_df.rename(
        columns={
            "spot_id": "Spot",
            "rank": "Hạng",
            "drug": "Tên thuốc",
            "predicted_LN_IC50": "LN_IC50 dự đoán",
        }
    )

    return table_df


# ============================================================
# 9. VISUALIZATION HELPERS
# ============================================================

def plot_top10_bar(
    df: pd.DataFrame,
    drug_col: str,
    value_col: str,
    title: str,
    figure_size: tuple[float, float] = (8.2, 4.6),
) -> None:
    """
    Plot vertical bar chart with bars rising from zero.

    Negative LN_IC50 values are displayed as positive bar heights for easier
    visual comparison, while the labels and y-axis still show the real
    negative LN_IC50 values.
    """
    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(
        plot_df[value_col],
        errors="coerce",
    )
    plot_df = plot_df.dropna(subset=[value_col])
    plot_df = plot_df.sort_values(value_col, ascending=True)

    actual_values = plot_df[value_col].to_numpy(dtype=float)

    if len(actual_values) == 0:
        st.info("Không có dữ liệu để vẽ biểu đồ.")
        return

    if np.nanmax(actual_values) <= 0:
        display_values = -actual_values
    else:
        display_values = actual_values

    fig, ax = plt.subplots(figsize=figure_size)

    bars = ax.bar(
        plot_df[drug_col].astype(str),
        display_values,
        width=0.72,
    )

    ax.axhline(0, linewidth=1.2, color="#111827")
    ax.set_xlabel("Tên thuốc", fontsize=11, fontweight="bold")
    ax.set_ylabel("LN_IC50 dự đoán", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    if np.nanmax(actual_values) <= 0:
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: "0" if abs(value) < 1e-9 else f"-{value:g}"
            )
        )

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)

    y_offset = max(display_values) * 0.018 if len(display_values) > 0 else 0.02

    for bar, actual_value in zip(bars, actual_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{actual_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#111827",
        )

    max_display_value = float(np.nanmax(display_values)) if len(display_values) > 0 else 1.0
    ax.set_ylim(
        bottom=0,
        top=max_display_value * 1.18,
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_spatial_map(
    positions_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    selected_drug: str,
) -> None:
    """
    Plot spatial map for selected drug.
    """
    plot_positions = positions_df.copy()

    if "spot_id" not in plot_positions.columns:
        raise ValueError("positions_df thiếu cột spot_id.")

    plot_positions["spot_id"] = plot_positions["spot_id"].astype(str)
    prediction_index = prediction_df.index.astype(str)

    plot_positions = (
        plot_positions
        .set_index("spot_id")
        .loc[prediction_index]
        .reset_index()
    )

    values = prediction_df[selected_drug].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))

    scatter = ax.scatter(
        plot_positions["x"],
        plot_positions["y"],
        c=values,
        s=7,
    )

    ax.invert_yaxis()
    ax.set_xlabel("x", fontsize=10, fontweight="bold")
    ax.set_ylabel("y", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Bản đồ không gian của thuốc: {selected_drug}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(axis="both", labelsize=9)

    colorbar = fig.colorbar(scatter, ax=ax, label="Refined LN_IC50", shrink=0.86)
    colorbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)





def render_custom_table(
    df: pd.DataFrame,
    compact: bool = False,
) -> None:
    """
    Render dataframe as a styled HTML table without Streamlit index column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to render.
    compact : bool
        Whether table height should fit content.
    """
    table_df = df.copy()

    html_table = table_df.to_html(
        index=False,
        escape=False,
        border=0,
    )

    wrapper_class = (
        "custom-table-wrapper compact"
        if compact
        else "custom-table-wrapper"
    )

    html = f"""
    <div class="{wrapper_class}">
        {html_table}
    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True,
    )

# ============================================================
# 10. CELL LINE PREDICTION

# ============================================================

def predict_one_cell_line(
    cell_line: str,
    X_test_pca: pd.DataFrame,
    model_objects: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict one cell line with Ridge, XGBoost, DNN and mean baseline.

    Parameters
    ----------
    cell_line : str
        Cell line ID.
    X_test_pca : pd.DataFrame
        Test PCA dataframe.
    model_objects : dict[str, Any]
        Loaded model objects.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Full prediction dataframe and comparison dataframe.
    """
    if cell_line not in X_test_pca.index:
        raise ValueError(f"Không tìm thấy Cell Line: {cell_line}")

    X_one = X_test_pca.loc[[cell_line]].copy()
    X_one.index.name = "cell_line"

    selected_drugs = model_objects["selected_drugs"]

    ridge_df = predict_with_sklearn_model(
        model=model_objects["ridge_model"],
        X=X_one,
        selected_drugs=selected_drugs,
    )

    xgboost_df = predict_with_sklearn_model(
        model=model_objects["xgboost_model"],
        X=X_one,
        selected_drugs=selected_drugs,
    )

    dnn_df = predict_with_dnn_model(
        model=model_objects["dnn_model"],
        X=X_one,
        selected_drugs=selected_drugs,
        device=model_objects["torch_device"],
    )

    common_drugs = (
        ridge_df.columns
        .intersection(xgboost_df.columns)
        .intersection(dnn_df.columns)
    )

    ridge_series = ridge_df.loc[cell_line, common_drugs]
    xgboost_series = xgboost_df.loc[cell_line, common_drugs]
    dnn_series = dnn_df.loc[cell_line, common_drugs]

    mean_series = (
        ridge_series
        + xgboost_series
        + dnn_series
    ) / 3.0

    comparison_df = pd.DataFrame(
        {
            "drug": common_drugs,
            "ridge_LN_IC50": ridge_series.to_numpy(dtype=float),
            "xgboost_LN_IC50": xgboost_series.to_numpy(dtype=float),
            "dnn_LN_IC50": dnn_series.to_numpy(dtype=float),
            "mean_LN_IC50": mean_series.to_numpy(dtype=float),
        }
    )

    prediction_df = pd.DataFrame(
        [mean_series.to_numpy(dtype=float)],
        index=[cell_line],
        columns=common_drugs,
    )
    prediction_df.index.name = "cell_line"

    return prediction_df, comparison_df


def get_selected_model_prediction(
    comparison_df: pd.DataFrame,
    model_choice: str,
) -> pd.DataFrame:
    """
    Get prediction dataframe according to selected model.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison dataframe.
    model_choice : str
        Selected model.

    Returns
    -------
    pd.DataFrame
        Drug and selected prediction value.
    """
    value_col_map = {
        "DNN": "dnn_LN_IC50",
        "Ridge": "ridge_LN_IC50",
        "XGBoost": "xgboost_LN_IC50",
        "Trung bình 3 mô hình": "mean_LN_IC50",
    }

    value_col = value_col_map[model_choice]

    selected_df = comparison_df[
        [
            "drug",
            value_col,
        ]
    ].copy()

    selected_df = selected_df.rename(
        columns={
            value_col: "predicted_LN_IC50",
        }
    )

    selected_df = selected_df.sort_values(
        "predicted_LN_IC50",
        ascending=True,
    ).reset_index(drop=True)

    selected_df.insert(
        0,
        "rank",
        range(1, len(selected_df) + 1),
    )

    return selected_df


def render_cell_line_tab(
    model_objects: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """
    Render cell line prediction tab.

    Parameters
    ----------
    model_objects : dict[str, Any]
        Loaded model objects.
    paths : dict[str, Path]
        App paths.
    """
    st.subheader("🧪 Dự đoán Cell Line")

    try:
        X_test_pca = load_test_pca(paths["test_pca_file"])
    except Exception as error:
        st.error(f"Lỗi tải dữ liệu test: {error}")
        return

    input_col, model_col, button_col = st.columns([1.0, 1.55, 1.0])

    with input_col:
        selected_cell_line = st.selectbox(
            "Chọn Cell Line",
            options=X_test_pca.index.astype(str).tolist(),
        )

    with model_col:
        model_choice = st.radio(
            "Chọn mô hình dự đoán",
            options=MODEL_OPTIONS,
            horizontal=True,
        )

    with button_col:
        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
        predict_button = st.button(
            "Dự đoán đáp ứng thuốc",
            use_container_width=True,
        )

    if not predict_button:
        return

    try:
        with st.spinner("Đang dự đoán..."):
            _, comparison_df = predict_one_cell_line(
                cell_line=selected_cell_line,
                X_test_pca=X_test_pca,
                model_objects=model_objects,
            )

            selected_prediction_df = get_selected_model_prediction(
                comparison_df=comparison_df,
                model_choice=model_choice,
            )

            top10_df = selected_prediction_df.head(TOP_K_DRUGS).copy()

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            render_metric_card("Cell Line", selected_cell_line)

        with metric_col2:
            render_metric_card("Số thuốc", comparison_df.shape[0])

        with metric_col3:
            render_metric_card("Top 1 drug", top10_df.iloc[0]["drug"])

        render_success(
            "Giá trị LN_IC50 càng thấp Cell Line càng nhạy với thuốc."
        )

        st.markdown("---")
        st.markdown("### Top 10 thuốc có độ nhạy cao nhất")

        table_col, chart_col = st.columns([1.0, 1.15])

        with table_col:
            render_section_title(
                "Bảng Top 10 thuốc",
                "Danh sách 10 thuốc có giá trị LN_IC50 dự đoán thấp nhất.",
            )
            render_custom_table(
                format_cellline_top10_table(top10_df),
                compact=False,
            )

        with chart_col:
            render_section_title(
                "Biểu đồ Top 10 thuốc",
                "Cột cao hơn tương ứng giá trị LN_IC50 thấp hơn.",
            )
            plot_top10_bar(
                df=top10_df,
                drug_col="drug",
                value_col="predicted_LN_IC50",
                title="Top 10 thuốc có giá trị LN_IC50 thấp nhất",
                figure_size=(8.2, 4.55),
)

        if model_choice == "Trung bình 3 mô hình":
            st.markdown("---")
            st.markdown("### So sánh kết quả dự đoán của 3 mô hình Ridge, XGBoost và DNN")

            compare_top10_df = comparison_df[
                comparison_df["drug"].isin(top10_df["drug"])
            ].copy()

            compare_top10_df = compare_top10_df.sort_values(
                "mean_LN_IC50",
                ascending=True,
            )

            compare_top10_df = compare_top10_df.rename(
                columns={
                    "drug": "Tên thuốc",
                    "ridge_LN_IC50": "So sánh Ridge",
                    "xgboost_LN_IC50": "So sánh XGBoost",
                    "dnn_LN_IC50": "So sánh DNN",
                    "mean_LN_IC50": "Trung bình 3 mô hình",
                }
            )

            compare_top10_df = compare_top10_df[
                [
                    "Tên thuốc",
                    "So sánh Ridge",
                    "So sánh XGBoost",
                    "So sánh DNN",
                    "Trung bình 3 mô hình",
                ]
            ].reset_index(drop=True)

            render_section_title(
                "Bảng so sánh 3 mô hình",
                "So sánh dự đoán giá trị LN_IC50 của Ridge, XGBoost, DNN và trung bình 3 mô hình.",
            )
            render_custom_table(
                compare_top10_df,
                compact=True,
            )

        st.markdown("---")
        render_warning(DISCLAIMER_TEXT)

    except Exception as error:
        st.error(f"Lỗi dự đoán: {error}")

# ============================================================
# 11. SPATIAL FILE HELPERS
# ============================================================

def create_run_id() -> str:
    """
    Create run id.

    Returns
    -------
    str
        Run id.
    """
    return datetime.now().strftime("demo_%Y%m%d_%H%M%S")


def save_uploaded_file(uploaded_file: Any, output_file: Path) -> Path:
    """
    Save uploaded file.

    Parameters
    ----------
    uploaded_file : Any
        Streamlit uploaded file.
    output_file : Path
        Output path.

    Returns
    -------
    Path
        Saved file path.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return output_file


def dataframe_to_csv_bytes(
    df: pd.DataFrame,
    index: bool = False,
) -> bytes:
    """
    Convert dataframe to CSV bytes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    index : bool
        Whether to include index.

    Returns
    -------
    bytes
        CSV bytes.
    """
    return df.to_csv(index=index).encode("utf-8")


# ============================================================
# 12. SPATIAL PIPELINE
# ============================================================

def run_spatial_demo_pipeline(
    h5_file: Path,
    tissue_file: Path,
    model_objects: dict[str, Any],
    run_output_dir: Path,
) -> dict[str, Any]:
    """
    Run complete uploaded spatial demo pipeline.

    API audited:
    - preprocess_uploaded_spatial_dataset(h5_file, tissue_position_file, selected_genes, scaler, pca)
      returns X_spot_pca, positions_df, summary
    - predict_uploaded_spatial_baseline(X_spot_pca, selected_drugs, models, device, top_k)
      returns baseline_prediction_df, top10_df, summary
    - build_graph_for_processed_dataset(positions_df, X_pca, prediction_dict, mean_baseline, dataset_name)
      returns graph
    - predict_uploaded_spatial_gnn(graph, model, checkpoint, device, top_k)
      returns prediction_df, top10_df, overall_top10_df, summary

    Parameters
    ----------
    h5_file : Path
        Uploaded H5 file path.
    tissue_file : Path
        Uploaded tissue positions file path.
    model_objects : dict[str, Any]
        Loaded model objects.
    run_output_dir : Path
        Output directory.

    Returns
    -------
    dict[str, Any]
        Pipeline outputs.
    """
    X_spot_pca, positions_df, preprocessing_summary = (
        preprocess_uploaded_spatial_dataset(
            h5_file=h5_file,
            tissue_position_file=tissue_file,
            selected_genes=model_objects["selected_genes"],
            scaler=model_objects["scaler"],
            pca=model_objects["pca"],
        )
    )

    baseline_prediction_df, baseline_top10_df, baseline_summary = (
        predict_uploaded_spatial_baseline(
            X_spot_pca=X_spot_pca,
            selected_drugs=model_objects["selected_drugs"],
            models=model_objects["baseline_models"],
            device=model_objects["torch_device"],
            top_k=TOP_K_DRUGS,
        )
    )

    prediction_dict = predict_all_baseline_models(
        X_spot_pca=X_spot_pca,
        selected_drugs=model_objects["selected_drugs"],
        models=model_objects["baseline_models"],
        device=model_objects["torch_device"],
        log_file=None,
    )

    mean_baseline_df = create_spatial_mean_baseline(
        prediction_dict=prediction_dict,
        log_file=None,
    )

    graph = build_graph_for_processed_dataset(
        positions_df=positions_df,
        X_pca=X_spot_pca,
        prediction_dict=prediction_dict,
        mean_baseline=mean_baseline_df,
        dataset_name="demo_upload",
    )

    prediction_df, top10_df, overall_top10_df, gnn_summary = (
        predict_uploaded_spatial_gnn(
            graph=graph,
            model=model_objects["gnn_model"],
            checkpoint=model_objects["gnn_checkpoint"],
            device=model_objects["gnn_device"],
            top_k=TOP_K_DRUGS,
        )
    )

    run_output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(
        run_output_dir / "refined_ic50_gnn.csv",
        index=True,
    )

    top10_df.to_csv(
        run_output_dir / "refined_top10_drugs_per_spot.csv",
        index=False,
    )

    overall_top10_df.to_csv(
        run_output_dir / "refined_top10_drugs_overall.csv",
        index=False,
    )

    summary = {
        **preprocessing_summary,
        **baseline_summary,
        **gnn_summary,
        "run_output_dir": str(run_output_dir),
    }

    pd.DataFrame([summary]).to_csv(
        run_output_dir / "demo_summary.csv",
        index=False,
    )

    return {
        "X_spot_pca": X_spot_pca,
        "positions_df": positions_df,
        "preprocessing_summary": preprocessing_summary,
        "baseline_prediction_df": baseline_prediction_df,
        "baseline_top10_df": baseline_top10_df,
        "baseline_summary": baseline_summary,
        "prediction_dict": prediction_dict,
        "mean_baseline_df": mean_baseline_df,
        "graph": graph,
        "prediction_df": prediction_df,
        "top10_df": top10_df,
        "overall_top10_df": overall_top10_df,
        "gnn_summary": gnn_summary,
        "summary": summary,
        "run_output_dir": run_output_dir,
    }


# ============================================================
# 13. SPATIAL TAB RENDERING
# ============================================================

def render_spatial_metrics(outputs: dict[str, Any]) -> None:
    """
    Render spatial summary metrics.
    """
    prediction_df = outputs["prediction_df"]
    overall_top10_df = outputs["overall_top10_df"]

    top1_drug = (
        overall_top10_df.iloc[0]["drug"]
        if not overall_top10_df.empty
        else "N/A"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Số spot", prediction_df.shape[0])

    with col2:
        render_metric_card("Số thuốc", prediction_df.shape[1])

    with col3:
        render_metric_card(
            "Input dim",
            model_input_dim_from_outputs(outputs),
        )

    with col4:
        render_metric_card("Top 1 overall drug", top1_drug)

    render_success(
        "Giá trị LN_IC50 càng thấp mẫu spatial càng nhạy với thuốc."
    )


def get_graph_attr_safe(graph: Any, attr_name: str) -> Any | None:
    """
    Get graph attribute from PyG Data or dictionary if available.
    """
    if graph is None:
        return None

    if isinstance(graph, dict):
        return graph.get(attr_name)

    return getattr(graph, attr_name, None)


def model_input_dim_from_outputs(outputs: dict[str, Any]) -> Any:
    """
    Get input dimension from GNN summary, graph.x, or reconstructed feature size.
    """
    summary = outputs.get("gnn_summary", {})

    for key in ["input_dim", "model_input_dim", "n_features"]:
        if key in summary and summary[key] not in [None, "", "N/A"]:
            return summary[key]

    graph_x = get_graph_attr_safe(
        outputs.get("graph"),
        "x",
    )

    if graph_x is not None and hasattr(graph_x, "shape"):
        return int(graph_x.shape[1])

    X_spot_pca = outputs.get("X_spot_pca")
    prediction_dict = outputs.get("prediction_dict", {})

    if isinstance(X_spot_pca, pd.DataFrame) and isinstance(prediction_dict, dict):
        extra_dims = sum(
            pred_df.shape[1]
            for pred_df in prediction_dict.values()
            if isinstance(pred_df, pd.DataFrame)
        )
        return int(X_spot_pca.shape[1] + extra_dims)

    return "N/A"


def render_overall_top10(outputs: dict[str, Any]) -> None:
    """
    Render overall top10 section.
    """
    overall_top10_df = outputs["overall_top10_df"]

    st.markdown("### Top 10 thuốc tiềm năng cho toàn bộ mẫu")

    overall_display_df = format_overall_top10_table(overall_top10_df)

    table_col, chart_col = st.columns([1.15, 1.0])

    with table_col:
        render_section_title(
            "Bảng Top 10 thuôc tiềm năng cho toàn bộ mẫu",
            "Xếp hạng thuốc theo giá trị Mean Refined LN_IC50 giảm dần.",
        )
        render_custom_table(
            overall_display_df,
            compact=True,
        )

    with chart_col:
        render_section_title(
            "Biểu đồ Top 10 thuốc tiềm năng cho toàn bộ mẫu",
            "Tổng hợp dự đoán trên toàn bộ spot của mẫu spatial.",
        )
        plot_top10_bar(
            df=overall_top10_df,
            drug_col="drug",
            value_col="mean_refined_LN_IC50",
            title="Top 10 thuốc tiềm năng cho toàn bộ mẫu",
            figure_size=(7.4, 4.55),
        )


def render_per_spot_top10(outputs: dict[str, Any]) -> None:
    """
    Render per-spot top10 section.
    """
    prediction_df = outputs["prediction_df"]
    top10_df = outputs["top10_df"]

    st.markdown("### Top 10 thuốc của Spot")

    selected_spot = st.selectbox(
        "Chọn Spot",
        options=prediction_df.index.astype(str).tolist(),
        key="selected_spatial_spot",
    )

    spot_top10_df = top10_df[
        top10_df["spot_id"].astype(str) == str(selected_spot)
    ].copy()

    table_col, chart_col = st.columns([1.0, 1.0])

    with table_col:
        render_section_title(
            "Bảng Top 10 thuốc tiềm năng theo Spot",
            f"Spot đang chọn: {selected_spot}",
        )
        render_custom_table(
            format_per_spot_top10_table(spot_top10_df),
            compact=True,
        )

    with chart_col:
        render_section_title(
            "Biểu đồ Top 10 thuốc tiềm năng theo Spot",
            "Cột cao hơn tương ứng giá trị LN_IC50 thấp hơn.",
        )
        plot_top10_bar(
            df=spot_top10_df,
            drug_col="drug",
            value_col="predicted_LN_IC50",
            title=f"Top 10 thuốc của Spot {selected_spot}",
            figure_size=(7.4, 4.55),
        )


def render_spatial_map(outputs: dict[str, Any]) -> None:
    """
    Render spatial map section.
    """
    prediction_df = outputs["prediction_df"]
    positions_df = outputs["positions_df"]

    st.markdown("### Bản đồ không gian của thuốc")

    control_col, map_col = st.columns([0.78, 1.42])

    with control_col:
        selected_drug = st.selectbox(
            "Chọn thuốc để hiển thị Spatial Map",
            options=prediction_df.columns.astype(str).tolist(),
            key="selected_spatial_drug",
        )

        render_section_title(
            "Thông tin bản đồ",
            "Màu sắc biểu thị giá trị Refined LN_IC50 của thuốc tại từng spot.",
        )
        st.markdown(
            f"""
            <div class="content-card">
                <b>Thuốc đang chọn:</b><br>
                <span style="font-size:1.15rem;font-weight:800;color:#0f172a;">{selected_drug}</span><br><br>
                <span style="color:#64748b;">Giá trị LN_IC50 càng thấp Spot càng nhạy với thuốc.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with map_col:
        render_section_title(
            "Spatial Map",
            "Phân bố đáp ứng thuốc đã tinh chỉnh bằng GraphSAGE.",
        )
        plot_spatial_map(
            positions_df=positions_df,
            prediction_df=prediction_df,
            selected_drug=selected_drug,
        )


def render_download_buttons(outputs: dict[str, Any]) -> None:
    """
    Render download buttons.

    Parameters
    ----------
    outputs : dict[str, Any]
        Spatial pipeline outputs.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Toàn bộ kết quả dự đoán",
            data=dataframe_to_csv_bytes(outputs["prediction_df"], index=True),
            file_name="refined_ic50_gnn.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "Top 10 thuốc tiềm năng theo Spot",
            data=dataframe_to_csv_bytes(outputs["top10_df"], index=False),
            file_name="refined_top10_drugs_per_spot.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            "Top 10 thuốc tiềm năng cho toàn bộ mẫu",
            data=dataframe_to_csv_bytes(outputs["overall_top10_df"], index=False),
            file_name="refined_top10_drugs_overall.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_spatial_results(outputs: dict[str, Any]) -> None:
    """
    Render saved spatial prediction results.
    """
    render_spatial_metrics(outputs)

    st.markdown("---")
    render_overall_top10(outputs)

    st.markdown("---")
    render_per_spot_top10(outputs)

    st.markdown("---")
    render_spatial_map(outputs)

    st.markdown("---")
    st.markdown("### Tải kết quả")
    render_download_buttons(outputs)

    render_warning(DISCLAIMER_TEXT)


def render_spatial_tab(
    model_objects: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """
    Render spatial prediction tab.

    Parameters
    ----------
    model_objects : dict[str, Any]
        Loaded model objects.
    paths : dict[str, Path]
        App paths.
    """
    st.subheader("🧬 Dự đoán dữ liệu sinh học không gian")

    upload_col1, upload_col2, button_col = st.columns([1.15, 1.15, 1.0])

    with upload_col1:
        h5_upload = st.file_uploader(
            "Tải lên file biểu hiện gen (.h5)",
            type=["h5"],
            accept_multiple_files=False,
            key="spatial_h5_upload",
        )

    with upload_col2:
        tissue_upload = st.file_uploader(
            "Tải lên file vị trí spot (.csv)",
            type=["csv"],
            accept_multiple_files=False,
            key="spatial_tissue_upload",
        )

    with button_col:
        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
        run_button = st.button(
            "Chạy dự đoán dữ liệu sinh học không gian",
            use_container_width=True,
        )

    if run_button:

        if h5_upload is None:
            st.error("Vui lòng tải lên file biểu hiện gen .h5.")
            return

        if tissue_upload is None:
            st.error("Vui lòng tải lên file vị trí spot .csv.")
            return

        try:
            run_id = create_run_id()

            upload_dir = paths["demo_uploads_dir"] / run_id
            output_dir = paths["demo_results_dir"] / run_id

            h5_file = save_uploaded_file(
                uploaded_file=h5_upload,
                output_file=upload_dir / "filtered_feature_bc_matrix.h5",
            )

            tissue_file = save_uploaded_file(
                uploaded_file=tissue_upload,
                output_file=upload_dir / tissue_upload.name,
            )

            with st.spinner(
                "Đang dự đoán..."
            ):
                outputs = run_spatial_demo_pipeline(
                    h5_file=h5_file,
                    tissue_file=tissue_file,
                    model_objects=model_objects,
                    run_output_dir=output_dir,
                )

            st.session_state["spatial_outputs"] = outputs

        except Exception as error:
            st.error(f"Lỗi khi chạy pipeline spatial: {error}")
            return

    if "spatial_outputs" in st.session_state:
        render_spatial_results(
            st.session_state["spatial_outputs"]
        )


# ============================================================
# 14. MAIN LAYOUT
# ============================================================

def render_main_layout(
    model_objects: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """
    Render main layout.

    Parameters
    ----------
    model_objects : dict[str, Any]
        Loaded model objects.
    paths : dict[str, Path]
        App paths.
    """
    tab1, tab2 = st.tabs(
        [
            "🧪 Dự đoán Cell Line",
            "🧬 Dự đoán dữ liệu sinh học không gian",
        ]
    )

    with tab1:
        render_cell_line_tab(
            model_objects=model_objects,
            paths=paths,
        )

    with tab2:
        render_spatial_tab(
            model_objects=model_objects,
            paths=paths,
        )


# ============================================================
# 15. MAIN
# ============================================================

def main() -> None:
    """
    Main Streamlit app.
    """
    st.set_page_config(
        page_title="Breast Cancer Drug Response Demo",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    render_header()

    try:
        with st.spinner("Đang tải mô hình đã huấn luyện..."):
            paths = setup_app_paths()
            model_objects = load_all_models(
                project_root=paths["project_root"]
            )

        render_sidebar()

        render_main_layout(
            model_objects=model_objects,
            paths=paths,
        )

    except Exception as error:
        st.error(f"Lỗi khởi động ứng dụng: {error}")


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
