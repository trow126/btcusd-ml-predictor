# data_processor/feature_modules/price_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.price")

def generate_price_change_features(df: pd.DataFrame, vwap_period: int = 14) -> pd.DataFrame:
    """
    価格変動率と重要なローソク足パターンに関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        vwap_period: VWAP計算期間
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("価格変動率とローソク足パターンの特徴量を追加しています")

    features = {}

    # 価格変動率（現在の終値 / 過去の終値 - 1）
    for i in range(1, 11):  # 過去1〜10期間に絞る
        features[f'price_change_pct_{i}'] = df['close'].pct_change(i)

    # ローソク足の重要な形状指標
    features['candle_size'] = (df['high'] - df['low']) / df['open']  # ローソク足の大きさ
    features['body_size'] = abs(df['close'] - df['open']) / df['open']  # 実体部分の大きさ
    features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']  # 上ヒゲ
    features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']  # 下ヒゲ
    features['is_bullish'] = (df['close'] > df['open']).astype(int)  # 陽線=1, 陰線=0

    # ハイローの比率
    features['high_low_ratio'] = df['high'] / df['low']

    # 価格振れ幅（当日の変動率）
    features['price_range_pct'] = (df['high'] - df['low']) / df['open']

    # ギャップアップ/ダウン（5分足では重要度は低いが、一応）
    features['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).clip(lower=0)
    features['gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1)).clip(lower=0)

    # 出来高加重平均価格（VWAP）
    features['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).rolling(window=vwap_period).sum() / df['volume'].rolling(window=vwap_period).sum()

    # 価格モメンタム（現在値と過去平均値の差）
    for period in [5, 10, 20]:
        features[f'price_momentum_{period}'] = df['close'] - df['close'].rolling(window=period).mean()

    return pd.DataFrame(features, index=df.index)

def generate_high_low_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    高値/安値からの距離に関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("高値/安値からの距離の特徴量を追加しています")

    features = {}

    # 期間ごとの最高値/最安値からの距離
    for period in [5, 10, 20, 50]:
        # 過去N期間の最高値と最安値
        highest = df['high'].rolling(window=period).max()
        lowest = df['low'].rolling(window=period).min()

        # 現在値と最高値/最安値との距離（パーセンテージ）
        features[f'dist_from_high_{period}'] = (df['close'] - highest) / highest
        features[f'dist_from_low_{period}'] = (df['close'] - lowest) / lowest

        # 価格が期間の高値・安値圏にあるかどうか
        features[f'near_high_{period}'] = (features[f'dist_from_high_{period}'] > -0.005).astype(int)
        features[f'near_low_{period}'] = (features[f'dist_from_low_{period}'] < 0.005).astype(int)

    # 複数期間（5, 20, 50）の高値・安値範囲内での相対位置（0～1）
    for period in [5, 20, 50]:
        highest = df['high'].rolling(window=period).max()
        lowest = df['low'].rolling(window=period).min()
        features[f'price_position_{period}'] = (df['close'] - lowest) / (highest - lowest)

    return pd.DataFrame(features, index=df.index)
