# model_builder/utils/data_utils.py
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger("data_utils")

def load_data(input_dir: str, input_filename: str) -> pd.DataFrame:
    """
    特徴量データを読み込む

    Args:
        input_dir: 入力ディレクトリパス
        input_filename: 入力ファイル名
        
    Returns:
        DataFrame: 読み込んだデータ
    """
    input_path = Path(input_dir) / input_filename
    logger.info(f"load_data: データを {input_path} から読み込みます")

    try:
        df = pd.read_csv(input_path, index_col="timestamp", parse_dates=True)
        logger.info(f"load_data: {len(df)} 行のデータを読み込みました")
        return df
    except Exception as e:
        logger.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

def prepare_features(df: pd.DataFrame, feature_groups: Dict[str, bool], target_periods: List[int]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
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