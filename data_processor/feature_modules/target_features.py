# data_processor/feature_modules/target_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.target")

def generate_target_features(df: pd.DataFrame, target_periods: list, classification_threshold: float) -> pd.DataFrame:
    """
    予測対象（目標変数）を生成（改良版）
    
    Args:
        df: 入力データフレーム
        target_periods: 予測期間のリスト
        classification_threshold: 分類の閾値
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("改良版予測対象の目標変数を追加しています")

    features = {}
    
    # ボラティリティベースの動的閾値を計算
    volatility = df['close'].pct_change().rolling(20).std().fillna(classification_threshold)
    
    # 価格の平滑化
    smoothed_close = df['close'].rolling(3).mean()
    
    # 各予測時間軸に対する目標変数を生成
    for period in target_periods:
        # 価格変動率（回帰用）- 元のまま
        target_change = df['close'].pct_change(periods=period).shift(-period)
        features[f'target_price_change_pct_{period}'] = target_change
        
        # 平滑化した価格変動率
        smoothed_change = smoothed_close.pct_change(periods=period).shift(-period)
        features[f'target_smoothed_change_{period}'] = smoothed_change
        
        # 動的閾値を使用した3分類
        dynamic_threshold = volatility * 0.8  # より適切な値に調整
        # 未来のボラティリティは予測時には不明なので、現在のボラティリティを使用
        
        features[f'target_price_direction_{period}'] = np.where(
            target_change > dynamic_threshold,
            1,  # 上昇
            np.where(
                target_change < -dynamic_threshold,
                -1,  # 下落
                0   # 横ばい
            )
        )
        
        # 2クラス分類（単純な上昇/下落）
        features[f'target_binary_{period}'] = np.where(
            target_change > 0,
            1,  # 上昇
            0   # 下落
        )

    return pd.DataFrame(features, index=df.index)
