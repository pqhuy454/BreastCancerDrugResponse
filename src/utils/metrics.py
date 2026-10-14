import numpy as np
import pandas as pd


def to_numpy_array(x) -> np.ndarray:
    """
    Chuyển dữ liệu đầu vào về numpy array.

    Hỗ trợ:
    - numpy.ndarray
    - pandas.DataFrame
    - pandas.Series
    - list
    """
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()

    return np.asarray(x)


def validate_shapes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
    Kiểm tra y_true, y_pred và mask có cùng shape hay không.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape không khớp: y_true {y_true.shape} != y_pred {y_pred.shape}"
        )

    if y_true.shape != mask.shape:
        raise ValueError(
            f"Shape không khớp: y_true {y_true.shape} != mask {mask.shape}"
        )


def get_valid_mask(y_true, y_pred, mask) -> np.ndarray:
    """
    Tạo mask hợp lệ cuối cùng.

    Điều kiện hợp lệ:
    - mask > 0
    - y_true không NaN
    - y_pred không NaN
    - y_true không inf
    - y_pred không inf
    """
    y_true_arr = to_numpy_array(y_true).astype(float)
    y_pred_arr = to_numpy_array(y_pred).astype(float)
    mask_arr = to_numpy_array(mask)

    validate_shapes(y_true_arr, y_pred_arr, mask_arr)

    mask_bool = mask_arr > 0

    valid_mask = (
        mask_bool
        & np.isfinite(y_true_arr)
        & np.isfinite(y_pred_arr)
    )

    return valid_mask


def masked_mae(y_true, y_pred, mask) -> float:
    """
    Tính Masked MAE.

    MAE = mean(abs(y_true - y_pred))
    chỉ tính trên các vị trí hợp lệ.
    """
    y_true_arr = to_numpy_array(y_true).astype(float)
    y_pred_arr = to_numpy_array(y_pred).astype(float)

    valid_mask = get_valid_mask(y_true_arr, y_pred_arr, mask)

    if valid_mask.sum() == 0:
        return np.nan

    error = np.abs(y_true_arr[valid_mask] - y_pred_arr[valid_mask])

    return float(np.mean(error))


def masked_mse(y_true, y_pred, mask) -> float:
    """
    Tính Masked MSE.

    MSE = mean((y_true - y_pred)^2)
    chỉ tính trên các vị trí hợp lệ.
    """
    y_true_arr = to_numpy_array(y_true).astype(float)
    y_pred_arr = to_numpy_array(y_pred).astype(float)

    valid_mask = get_valid_mask(y_true_arr, y_pred_arr, mask)

    if valid_mask.sum() == 0:
        return np.nan

    squared_error = (y_true_arr[valid_mask] - y_pred_arr[valid_mask]) ** 2

    return float(np.mean(squared_error))


def masked_rmse(y_true, y_pred, mask) -> float:
    """
    Tính Masked RMSE.

    RMSE = sqrt(MSE)
    """
    mse = masked_mse(y_true, y_pred, mask)

    if np.isnan(mse):
        return np.nan

    return float(np.sqrt(mse))


def masked_r2(y_true, y_pred, mask) -> float:
    """
    Tính Masked R2.

    R2 = 1 - SS_res / SS_tot
    """
    y_true_arr = to_numpy_array(y_true).astype(float)
    y_pred_arr = to_numpy_array(y_pred).astype(float)

    valid_mask = get_valid_mask(y_true_arr, y_pred_arr, mask)

    if valid_mask.sum() == 0:
        return np.nan

    y_true_valid = y_true_arr[valid_mask]
    y_pred_valid = y_pred_arr[valid_mask]

    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - ss_res / ss_tot)


def count_observed(mask) -> int:
    """
    Đếm số phần tử có mask > 0.
    """
    mask_arr = to_numpy_array(mask)

    return int((mask_arr > 0).sum())


def count_valid(y_true, y_pred, mask) -> int:
    """
    Đếm số phần tử hợp lệ thực sự được dùng để tính metric.
    """
    valid_mask = get_valid_mask(y_true, y_pred, mask)

    return int(valid_mask.sum())


def per_drug_mae(
    y_true,
    y_pred,
    mask,
    drug_names=None,
) -> pd.DataFrame:
    """
    Tính MAE cho từng thuốc/cột.

    Trả về DataFrame gồm:
    - drug
    - mae
    - n_valid
    """
    y_true_arr = to_numpy_array(y_true).astype(float)
    y_pred_arr = to_numpy_array(y_pred).astype(float)
    mask_arr = to_numpy_array(mask)

    validate_shapes(y_true_arr, y_pred_arr, mask_arr)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
        y_pred_arr = y_pred_arr.reshape(-1, 1)
        mask_arr = mask_arr.reshape(-1, 1)

    n_drugs = y_true_arr.shape[1]

    if drug_names is None:
        drug_names = [f"Drug_{i}" for i in range(n_drugs)]

    if len(drug_names) != n_drugs:
        raise ValueError(
            f"Số drug_names ({len(drug_names)}) không khớp số cột ({n_drugs})."
        )

    results = []

    for j in range(n_drugs):
        valid_mask = get_valid_mask(
            y_true_arr[:, j],
            y_pred_arr[:, j],
            mask_arr[:, j],
        )

        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            mae = np.nan
        else:
            mae = float(
                np.mean(
                    np.abs(
                        y_true_arr[:, j][valid_mask]
                        - y_pred_arr[:, j][valid_mask]
                    )
                )
            )

        results.append({
            "drug": drug_names[j],
            "mae": mae,
            "n_valid": n_valid,
        })

    return pd.DataFrame(results)


def per_drug_rmse(
    y_true,
    y_pred,
    mask,
    drug_names=None,
) -> pd.DataFrame:
    """
    Tính RMSE cho từng thuốc/cột.

    Trả về DataFrame gồm:
    - drug
    - rmse
    - n_valid
    """
    y_true_arr = to_numpy_array(y_true).astype(float)
    y_pred_arr = to_numpy_array(y_pred).astype(float)
    mask_arr = to_numpy_array(mask)

    validate_shapes(y_true_arr, y_pred_arr, mask_arr)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
        y_pred_arr = y_pred_arr.reshape(-1, 1)
        mask_arr = mask_arr.reshape(-1, 1)

    n_drugs = y_true_arr.shape[1]

    if drug_names is None:
        drug_names = [f"Drug_{i}" for i in range(n_drugs)]

    if len(drug_names) != n_drugs:
        raise ValueError(
            f"Số drug_names ({len(drug_names)}) không khớp số cột ({n_drugs})."
        )

    results = []

    for j in range(n_drugs):
        valid_mask = get_valid_mask(
            y_true_arr[:, j],
            y_pred_arr[:, j],
            mask_arr[:, j],
        )

        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            rmse = np.nan
        else:
            mse = np.mean(
                (
                    y_true_arr[:, j][valid_mask]
                    - y_pred_arr[:, j][valid_mask]
                ) ** 2
            )
            rmse = float(np.sqrt(mse))

        results.append({
            "drug": drug_names[j],
            "rmse": rmse,
            "n_valid": n_valid,
        })

    return pd.DataFrame(results)


def calculate_all_metrics(y_true, y_pred, mask) -> dict:
    """
    Tính toàn bộ metric chính và trả về dictionary.
    """
    return {
        "MAE": masked_mae(y_true, y_pred, mask),
        "MSE": masked_mse(y_true, y_pred, mask),
        "RMSE": masked_rmse(y_true, y_pred, mask),
        "R2": masked_r2(y_true, y_pred, mask),
        "observed_values": count_observed(mask),
        "valid_values": count_valid(y_true, y_pred, mask),
    }


def print_metrics(metrics_dict: dict, prefix: str = "") -> None:
    """
    In metric gọn gàng, dùng khi debug hoặc kiểm tra nhanh.
    """
    if prefix:
        print(f"{prefix} metrics:")

    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    # Test nhanh bằng dữ liệu giả, không đọc file ngoài
    y_true = np.array([
        [1.0, 2.0, np.nan],
        [2.0, 3.0, 4.0],
    ])

    y_pred = np.array([
        [1.1, 1.8, 0.0],
        [1.9, 3.2, 4.5],
    ])

    mask = np.array([
        [1, 1, 0],
        [1, 1, 1],
    ])

    print("Masked MAE:", masked_mae(y_true, y_pred, mask))
    print("Masked MSE:", masked_mse(y_true, y_pred, mask))
    print("Masked RMSE:", masked_rmse(y_true, y_pred, mask))
    print("Masked R2:", masked_r2(y_true, y_pred, mask))

    all_metrics = calculate_all_metrics(y_true, y_pred, mask)
    print_metrics(all_metrics, prefix="All")

    print("\nPer-drug MAE:")
    print(per_drug_mae(y_true, y_pred, mask))

    print("\nPer-drug RMSE:")
    print(per_drug_rmse(y_true, y_pred, mask))