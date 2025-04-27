# data_processor/feature_modules/volume_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.volume")

def generate_volume_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    出来高変動率に関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("出来高変動率の特徴量を追加しています")

    features = {}

    # 出来高変動率
    for i in range(1, 6):  # 過去1〜5期間に絞る
        features[f'volume_change_pct_{i}'] = df['volume'].pct_change(i)

    # 出来高の移動平均
    for period in [5, 10, 20]:
        features[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()

    # 出来高比率（現在の出来高 / 過去N期間の平均出来高）
    for period in [5, 10, 20]:
        vol_sma = df['volume'].rolling(window=period).mean()
        features[f'volume_ratio_{period}'] = df['volume'] / vol_sma

    # 出来高の標準偏差（ボラティリティ指標）
    for period in [5, 10, 20]:
        features[f'volume_std_{period}'] = df['volume'].rolling(window=period).std()

    # 出来高モメンタム
    features['volume_momentum_5'] = df['volume'] - df['volume'].rolling(window=5).mean()

    # On-Balance Volume (OBV) - 価格変動の方向に基づいて累積される出来高
    features['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # 価格変動と出来高の関連性
    features['price_volume_correlation'] = df['close'].rolling(window=10).corr(df['volume'])

    return pd.DataFrame(features, index=df.index)
