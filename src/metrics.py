"""
모델 성능·안정성 진단 지표.
"""
import numpy as np
import pandas as pd


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error.

    Returns
    -------
    ece : float
    bins : pd.DataFrame
        bin별 (bin 구간, 표본 수, 평균 예측확률, 실제 부도율).
        reliability diagram을 그릴 때 사용 (04_calibration.ipynb).
        스칼라만 필요하면 `ece, _ = compute_ece(...)`로 받으면 됨 (06_monitoring.ipynb).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n = mask.sum()
        if n == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        weight = n / len(y_prob)
        ece += weight * abs(bin_acc - bin_conf)
        rows.append({'bin': f'{lo:.1f}-{hi:.1f}', 'n': n, 'confidence': bin_conf, 'actual_rate': bin_acc})
    return ece, pd.DataFrame(rows)


def compute_psi(baseline: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index. baseline 분위수 기준 bin을 만들어 current와 비교.
    """
    bin_edges = np.quantile(baseline.dropna(), np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return np.nan
    base_counts, _ = np.histogram(baseline.dropna(), bins=bin_edges)
    curr_counts, _ = np.histogram(current.dropna(), bins=bin_edges)
    base_pct = np.clip(base_counts / base_counts.sum(), 1e-4, None)
    curr_pct = np.clip(curr_counts / curr_counts.sum(), 1e-4, None)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
