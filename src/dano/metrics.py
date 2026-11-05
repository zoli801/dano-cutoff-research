"""All cutoff metrics are in the original 0--100 score scale."""
import numpy as np


def regression_metrics(y_true, y_pred):
    actual, predicted = np.asarray(y_true, float), np.asarray(y_pred, float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Nonempty aligned arrays are required")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions/labels are not allowed")
    error = predicted - actual
    return {"power_1_5": float(np.mean(np.abs(error) ** 1.5)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "max_abs_error": float(np.max(np.abs(error)))}
