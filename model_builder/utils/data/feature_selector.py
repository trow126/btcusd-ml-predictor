# model_builder/utils/data/feature_selector.py
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger("feature_selector")

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

    logger.info(f"選択された特徴量: {len(feature_cols)}個")

    return feature_cols

def prepare_features(df: pd.DataFrame, feature_groups: Dict[str, bool], target_periods: List[int]) -> tuple:
    """
    特徴量と目標変数を準備

    Args:
        df: 入力データフレーム
        feature_groups: 特徴量グループの選択設定
        target_periods: 予測対象期間のリスト

    Returns:
        Tuple: (特徴量のDict, 目標変数のDict)
    """
    logger.info("prepare_features: 特徴量と目標変数の準備を開始します")
    if df.empty:
        logger.warning("prepare_features: 入力データが空です")
        return {}, {}

    # 使用する特徴量を選択
    feature_cols = select_features(df, feature_groups)

    # 目標変数（各予測期間に対して）
    target_cols = {}
    for period in target_periods:
        # 回帰目標（価格変動率）
        target_cols[f"regression_{period}"] = f"target_price_change_pct_{period}"
        # 分類目標（価格変動方向）
        target_cols[f"classification_{period}"] = f"target_price_direction_{period}"

    # 特徴量と目標変数のDataFrameを準備
    X = df[feature_cols]
    y_dict = {}

    for target_name, target_col in target_cols.items():
        if target_col in df.columns:
            y_dict[target_name] = df[target_col]

    logger.info(f"prepare_features: 特徴量: {len(feature_cols)}個, 目標変数: {len(y_dict)}個")
    logger.info("prepare_features: 特徴量と目標変数の準備を終了します")

    return {"X": X}, y_dict