# model_builder/utils/feature_utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> Dict[str, float]:
    """
    モデルの特徴量重要度を取得

    Args:
        model: LightGBMモデル
        feature_names: 特徴量名のリスト
        top_n: 上位N個の特徴量を返す

    Returns:
        Dict: 特徴量名と重要度のDict
    """
    importance = model.feature_importance(importance_type="gain")
    feature_importance = dict(zip(feature_names, importance))

    # 上位N個の特徴量を取得
    sorted_importance = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return dict(sorted_importance)
    
def select_features(df: pd.DataFrame, feature_groups: Dict[str, bool]) -> List[str]:
    """
    学習に使用する特徴量を選択

    Args:
        df: 入力データフレーム
        feature_groups: 特徴量グループの選択設定

    Returns:
        List[str]: 選択された特徴量名のリスト
    """
    # 特徴量グループごとの選択条件
    feature_cols = []

    # 価格関連特徴量
    if feature_groups.get("price", False):
        price_cols = [col for col in df.columns if (
            col.startswith("price_change") or
            col in ["open", "high", "low", "close"] or
            col in ["candle_size", "body_size", "upper_shadow", "lower_shadow", "is_bullish"]
        )]
        feature_cols.extend(price_cols)

    # 出来高関連特徴量
    if feature_groups.get("volume", False):
        volume_cols = [col for col in df.columns if (
            col.startswith("volume") or
            col == "turnover"
        )]
        feature_cols.extend(volume_cols)

    # テクニカル指標関連特徴量
    if feature_groups.get("technical", False):
        technical_cols = [col for col in df.columns if (
            col.startswith("sma_") or
            col.startswith("ema_") or
            col.startswith("rsi") or
            col.startswith("bb_") or
            col.startswith("macd") or
            col.startswith("stoch_") or
            col.startswith("dist_from_") or
            (col.startswith("highest_") or col.startswith("lowest_"))
        )]
        feature_cols.extend(technical_cols)

    # 目標変数を特徴量から除外
    feature_cols = [col for col in feature_cols if not col.startswith("target_")]

    # 重複を削除
    feature_cols = list(set(feature_cols))
    
    return feature_cols