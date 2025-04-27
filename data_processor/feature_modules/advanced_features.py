# data_processor/feature_modules/advanced_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.advanced")

def generate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    高度な特徴量を生成
    
    Args:
        df: 入力データフレーム
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("高度な特徴量を追加しています")

    features = {}

    # 価格とボリュームの関係性指標
    features['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).rolling(window=10).sum()

    # 価格の短期勢い（モメンタム）
    features['price_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

    # 価格の加速度（モメンタムの変化率）
    features['price_acceleration'] = features['price_momentum'] - features['price_momentum'].shift(1)

    # 過去N期間の価格トレンド強度（1に近いほど強いトレンド）
    for period in [5, 10, 20]:
        x = np.arange(period)
        features[f'trend_strength_{period}'] = df['close'].rolling(window=period).apply(
            lambda y: np.abs(np.corrcoef(x, y)[0, 1]), raw=True
        )

    # 市場効率係数（Market Efficiency Coefficient）
    # 1に近いほど効率的な市場（ランダムウォーク）、0に近いほど非効率（トレンド/逆行）
    for period in [10, 20]:
        features[f'market_efficiency_{period}'] = abs(df['close'].diff(period)) / (
            df['close'].diff().abs().rolling(window=period).sum()
        )

    # ボラティリティに調整された価格変動指標
    for period in [5, 10]:
        returns = df['close'].pct_change(1)
        vol = returns.rolling(window=period).std()
        features[f'vol_adjusted_change_{period}'] = returns / vol

    # 価格と移動平均のゴールデンクロス/デッドクロスの距離
    ma_20 = df['close'].rolling(window=20).mean()
    features['price_ma_20_distance'] = (df['close'] - ma_20) / ma_20

    # 暴落/急騰検出
    returns = df['close'].pct_change()
    std_returns = returns.rolling(window=20).std()
    features['price_shock'] = returns / std_returns

    # 移動平均収束/発散指標
    features['ma_convergence'] = (
        df['close'].rolling(window=10).mean() -
        df['close'].rolling(window=20).mean()
    ) / df['close']

    # HL2, HLC3, OHLC4（様々な価格平均）
    features['hl2'] = (df['high'] + df['low']) / 2
    features['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    features['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # トレンド強度指標（ADX風）- 直近の動きとトレンドの一致度
    features['trend_intensity'] = abs(df['close'].diff(10)) / df['close'].diff().abs().rolling(window=10).sum()

    return pd.DataFrame(features, index=df.index)

def generate_fisher_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    フィッシャー変換RSIを生成
    
    Args:
        df: 入力データフレーム（RSI列を含む必要があります）
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("フィッシャー変換RSIの特徴量を追加しています")

    features = {}
    
    # フィッシャー変換RSI（RSIをより線形に変換）を計算
    if 'rsi_14' in df.columns:
        rsi = df['rsi_14']
        # 0.1〜0.9の範囲に変換（極端な値を避ける）
        rsi_scaled = 0.1 + 0.8 * (rsi / 100)
        features['fisher_transform_rsi'] = 0.5 * np.log((1 + rsi_scaled) / (1 - rsi_scaled))
    else:
        logger.warning("rsi_14 が存在しないため、フィッシャー変換RSIは生成されませんでした")

    return pd.DataFrame(features, index=df.index)
