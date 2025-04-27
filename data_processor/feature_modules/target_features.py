# data_processor/feature_modules/target_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.target")

def generate_target_features(df: pd.DataFrame, target_periods: list, classification_threshold: float) -> pd.DataFrame:
    """
    予測対象（目標変数）を生成
    
    Args:
        df: 入力データフレーム
        target_periods: 予測期間のリスト
        classification_threshold: 分類の閾値
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("予測対象の目標変数を追加しています")

    features = {}
    threshold = classification_threshold

    # 各予測時間軸に対する目標変数を生成
    for period in target_periods:
        # 価格変動率（回帰用）
        target_change = df['close'].pct_change(periods=period).shift(-period)
        features[f'target_price_change_pct_{period}'] = target_change

        # 価格変動カテゴリ（3分類）
        features[f'target_price_direction_{period}'] = np.where(
            target_change > threshold,
            1,  # 上昇
            np.where(
                target_change < -threshold,
                -1,  # 下落
                0   # 横ばい
            )
        )

    return pd.DataFrame(features, index=df.index)
