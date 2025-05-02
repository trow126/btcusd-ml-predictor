import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger("data_splitter")

def train_test_split(X: pd.DataFrame, y_dict: Dict[str, pd.Series], test_size: float) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, pd.Series]
]:
    """
    時系列を考慮してトレーニングデータとテストデータに分割

    Args:
        X: 特徴量DataFrame
        y_dict: 目標変数のDict
        test_size: テストデータの割合

    Returns:
        Tuple: (X_train, X_test, y_train, y_test) の辞書
    """
    logger.info("train_test_split: トレーニングデータとテストデータへの分割を開始します")
    if X.empty:
        logger.warning("train_test_split: 特徴量データが空です")
        return {}, {}, {}, {}

    # 時系列データなので、最後の一定割合をテストデータとする
    test_size_rows = int(len(X) * test_size)
    train_size = len(X) - test_size_rows

    X_train = {"X": X.iloc[:train_size].copy()}
    X_test = {"X": X.iloc[train_size:].copy()}

    y_train = {}
    y_test = {}

    for target_name, target_series in y_dict.items():
        y_train[target_name] = target_series.iloc[:train_size].copy()
        y_test[target_name] = target_series.iloc[train_size:].copy()

    logger.info(f"train_test_split: トレーニングデータ: {train_size}行, テストデータ: {test_size_rows}行")
    logger.info("train_test_split: トレーニングデータとテストデータへの分割を終了します")

    return X_train, X_test, y_train, y_test

def prepare_test_data(df: pd.DataFrame, test_size: float, target_periods: list) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    テストデータを準備

    Args:
        df: 入力データフレーム
        test_size: テストデータの割合
        target_periods: 予測対象期間のリスト

    Returns:
        Tuple: (特徴量DataFrame, 目標変数のDict)
    """
    logger.info("prepare_test_data: テストデータの準備を開始します")
    if df.empty:
        logger.warning("prepare_test_data: 入力データが空です")
        return pd.DataFrame(), {}

    # 時系列データなので、最後の一定割合をテストデータとする
    test_size_rows = int(len(df) * test_size)
    test_df = df.iloc[-test_size_rows:].copy()

    # 目標変数（各予測期間に対して）
    y_test = {}
    X_dict = {}  # 特殊な特徴量セット用

    # 特徴量（目標変数を除く）
    feature_cols = [col for col in test_df.columns if not col.startswith("target_")]
    X_test = test_df[feature_cols]
    X_dict["X"] = X_test  # 基本の特徴量セット

    for period in target_periods:
        # 回帰目標（価格変動率）
        if f"target_price_change_pct_{period}" in test_df.columns:
            y_test[f"regression_{period}"] = test_df[f"target_price_change_pct_{period}"]

        # 平滑化した回帰目標（平滑化した価格変動率）
        if f"target_smoothed_change_{period}" in test_df.columns:
            y_test[f"regression_smoothed_{period}"] = test_df[f"target_smoothed_change_{period}"]

        # 分類目標（価格変動方向）
        if f"target_price_direction_{period}" in test_df.columns:
            y_test[f"classification_{period}"] = test_df[f"target_price_direction_{period}"]

        # 二値分類目標（単純な上昇/下落）
        if f"target_binary_{period}" in test_df.columns:
            y_test[f"binary_classification_{period}"] = test_df[f"target_binary_{period}"]

        # 3分類目標（閾値ベース）
        if f"target_threshold_ternary_{period}" in test_df.columns:
            y_test[f"threshold_ternary_classification_{period}"] = test_df[f"target_threshold_ternary_{period}"]

        # 真の二値分類目標（横ばい除外）
        if f"target_threshold_binary_{period}" in test_df.columns:
            valid_mask = ~test_df[f"target_threshold_binary_{period}"].isna()
            if valid_mask.sum() > 0:
                # 有効なデータがある場合のみ
                y_test[f"threshold_binary_classification_{period}"] = test_df[f"target_threshold_binary_{period}"].loc[valid_mask]
                # 対応する特徴量も同じインデックスに絞る
                X_key = f"X_threshold_binary_{period}"
                X_dict[X_key] = X_test.loc[valid_mask]
                logger.info(f"prepare_test_data: 閾値ベース二値分類用の特徴量セット {X_key} を作成 ({len(X_dict[X_key])} 行)")

    # 高閾値シグナル変数の処理
    high_threshold_cols = [col for col in test_df.columns if 'high_threshold' in col]
    if high_threshold_cols:
        logger.info(f"prepare_test_data: {len(high_threshold_cols)}個の高閾値シグナル変数を対象に追加")
        for col in high_threshold_cols:
            # 元の変数名をそのまま使用
            target_name = col
            y_test[target_name] = test_df[col]
            logger.info(f"prepare_test_data: 高閾値シグナル変数 {target_name} を追加")
            
        # 存在確認のログ出力
        high_threshold_targets = [key for key in y_test.keys() if 'high_threshold' in key]
        logger.info(f"prepare_test_data: y_test内の高閾値シグナル変数: {len(high_threshold_targets)}個")
        for key in high_threshold_targets[:5]:
            logger.info(f"  - {key}")
        if len(high_threshold_targets) > 5:
            logger.info(f"  ...その他 {len(high_threshold_targets)-5}個")
    else:
        logger.warning("prepare_test_data: 高閾値シグナル変数が見つかりません")

    logger.info(f"prepare_test_data: テストデータ: {len(X_test)}行, 特徴量: {len(feature_cols)}個")
    logger.info(f"prepare_test_data: 特徴量セット: {list(X_dict.keys())}")
    logger.info("prepare_test_data: テストデータの準備を終了します")

    return X_dict, y_test