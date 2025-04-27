# data_processor/feature_modules/moving_average_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.moving_average")

def generate_moving_average_features(df: pd.DataFrame, ma_periods: list) -> pd.DataFrame:
    """
    移動平均線と乖離率に関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        ma_periods: 移動平均線の期間リスト
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("移動平均線と乖離率の特徴量を追加しています")

    features = {}

    # 単純移動平均線
    sma_dict = {}
    for period in ma_periods:
        sma = df['close'].rolling(window=period).mean()
        features[f'sma_{period}'] = sma
        sma_dict[period] = sma

        # 乖離率（現在値が移動平均線からどれだけ離れているか）
        features[f'sma_diff_pct_{period}'] = (df['close'] - sma) / sma

    # 指数移動平均線
    ema_dict = {}
    for period in [5, 10, 20, 50]:
        ema = df['close'].ewm(span=period, adjust=False).mean()
        features[f'ema_{period}'] = ema
        ema_dict[period] = ema

    # 重み付け移動平均線（WMA）- 近い日付ほど重みが大きい
    for period in [5, 20]:
        weights = np.arange(1, period + 1)
        features[f'wma_{period}'] = df['close'].rolling(period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )

    # 移動平均線のクロスシグナル
    if 5 in sma_dict and 20 in sma_dict:
        features['sma_5_20_cross'] = np.where(
            (sma_dict[5] > sma_dict[20]) & (sma_dict[5].shift(1) <= sma_dict[20].shift(1)),
            1,  # ゴールデンクロス
            np.where(
                (sma_dict[5] < sma_dict[20]) & (sma_dict[5].shift(1) >= sma_dict[20].shift(1)),
                -1,  # デッドクロス
                0  # クロスなし
            )
        )

    # 移動平均線の傾き
    for period in [5, 20, 50]:
        if period in sma_dict:
            features[f'sma_{period}_slope'] = sma_dict[period].diff(5) / sma_dict[period].shift(5)

    # 複数の移動平均線の位置関係（トレンド指標）
    if all(p in sma_dict for p in [5, 20, 50]):
        features['ma_trend'] = np.where(
            (sma_dict[5] > sma_dict[20]) & (sma_dict[20] > sma_dict[50]),
            1,  # 強い上昇トレンド
            np.where(
                (sma_dict[5] < sma_dict[20]) & (sma_dict[20] < sma_dict[50]),
                -1,  # 強い下降トレンド
                0  # 明確なトレンドなし
            )
        )

    return pd.DataFrame(features, index=df.index)
