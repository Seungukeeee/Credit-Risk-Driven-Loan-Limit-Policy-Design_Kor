"""
정책 tier 산정·기대손실(EL) 계산.
"""
import numpy as np
import pandas as pd

TIER_ORDER = ['저위험', '중위험', '고위험', '거절']
TIER_IDX = {t: i for i, t in enumerate(TIER_ORDER)}


def pd_base_tier(ratio: float) -> str:
    if ratio <= 0.5:
        return '저위험'
    elif ratio <= 2.0:
        return '중위험'
    elif ratio <= 4.0:
        return '고위험'
    else:
        return '거절'


def assign_risk_tier(policy_df: pd.DataFrame) -> pd.DataFrame:
    flag_ltv = policy_df['LTV'] > 1.2
    flag_cir_over = policy_df['CIR'] > 7.5
    elevated_pd = policy_df['base_tier'].isin(['중위험', '고위험'])
    flag_dti = elevated_pd & (policy_df['DTI'] > 0.40)

    downgrade_steps = flag_ltv.astype(int) + flag_cir_over.astype(int) + flag_dti.astype(int)
    base_pos = policy_df['base_tier'].map(TIER_IDX)
    final_pos = np.minimum(base_pos + downgrade_steps, len(TIER_ORDER) - 1)
    policy_df['risk_tier'] = final_pos.map(lambda i: TIER_ORDER[i])
    policy_df.loc[policy_df['CIR'] > 12, 'risk_tier'] = '거절'
    return policy_df


def compute_el(df: pd.DataFrame, exposure_col: str, lgd: float) -> float:
    """기대손실 = PD_calibrated * LGD * exposure. exposure_col은 상황에 따라
    'AMT_CREDIT'(정책 적용 전) 또는 'proposed_limit'(정책 적용 후)."""
    return (df['PD_calibrated'] * lgd * df[exposure_col]).sum()
