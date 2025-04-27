# data_processor/feature_modules/volatility_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.volatility")

def generate_bollinger_bands_features(df: pd.DataFrame, bollinger_period: int, bollinger_std: int) -> pd.DataFrame:
    """
    ボリンジャーバンドに関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        bollinger_period: ボリンジャーバンドの期間
        bollinger_std: ボリンジャーバンドの標準偏差倍率
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("ボリンジャーバンドの特徴量を追加しています")

    features = {}
    period = bollinger_period
    std_dev = bollinger_std

    # 移動平均
    bb_ma = df['close'].rolling(window=period).mean()
    features['bb_ma'] = bb_ma

    # 標準偏差
    bb_std = df['close'].rolling(window=period).std()
    features['bb_std'] = bb_std

    # ボリンジャーバンド（上限、下限）
    bb_upper = bb_ma + (bb_std * std_dev)
    bb_lower = bb_ma - (bb_std * std_dev)

    features['bb_upper'] = bb_upper
    features['bb_lower'] = bb_lower

    # バンド幅（ボラティリティ指標）
    features['bb_width'] = (bb_upper - bb_lower) / bb_ma

    # バンド幅の変化率
    features['bb_width_change'] = features['bb_width'].pct_change(1)

    # バンド幅の拡大/縮小シグナル
    features['bb_width_expanding'] = (features['bb_width'] > features['bb_width'].shift(5)).astype(int)

    # 価格位置（上限と下限の間での相対位置、0〜1）
    features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

    # シグナル（バンドからの超過状態）
    features['bb_upper_exceed'] = (df['close'] > bb_upper).astype(int)
    features['bb_lower_exceed'] = (df['close'] < bb_lower).astype(int)

    # ボリンジャースクイーズ（バンド幅が狭まり、大きな動きの前兆）
    features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(window=20).quantile(0.2)).astype(int)

    return pd.DataFrame(features, index=df.index)

def generate_atr_features(df: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    """
    ATR（Average True Range）に関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        atr_period: ATRの計算期間
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("ATRの特徴量を追加しています")

    features = {}

    # True Range計算
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # ATR
    features['atr'] = tr.rolling(window=atr_period).mean()
    
    # ATRの変化率
    features['atr_change'] = features['atr'].pct_change(3)
    
    # ATRに対する価格変動の比率（ボラティリティ調整済み価格変動）
    features['price_change_to_atr'] = df['close'].pct_change(1) / features['atr']
    
    # ボラティリティの加速/減速
    features['volatility_change'] = features['atr'].diff(5) / features['atr'].shift(5)
    
    return pd.DataFrame(features, index=df.index)
