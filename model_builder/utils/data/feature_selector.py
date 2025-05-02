# model_builder/utils/data/feature_selector.py
import pandas as pd
import logging
from typing import Dict, List, Tuple

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
        
        # 平滑化した回帰目標（平滑化した価格変動率）
        target_cols[f"regression_smoothed_{period}"] = f"target_smoothed_change_{period}"
        
        # 分類目標（価格変動方向）
        target_cols[f"classification_{period}"] = f"target_price_direction_{period}"
        
        # 従来の二値分類目標（単純な上昇/下落）
        target_cols[f"binary_classification_{period}"] = f"target_binary_{period}"
        
        # 3分類目標（閾値ベース）
        target_cols[f"threshold_ternary_classification_{period}"] = f"target_threshold_ternary_{period}"
        
        # 真の二値分類目標（横ばい除外）
        target_cols[f"threshold_binary_classification_{period}"] = f"target_threshold_binary_{period}"

    # 特徴量と目標変数のDataFrameを準備
    X = df[feature_cols]
    y_dict = {}
    X_dict = {"X": X}  # 基本の特徴量セット

    for target_name, target_col in target_cols.items():
        if target_col in df.columns:
            # 横ばいが除外されているターゲットは、NaNを含む行を除外
            if "threshold_binary_classification" in target_name:
                valid_target = df[target_col].dropna()
                if len(valid_target) > 0:
                    # 対応する特徴量も同じインデックスに絞る
                    target_X = X.loc[valid_target.index]
                    y_dict[target_name] = valid_target
                    # 対応する特徴量セットを特別に保存
                    X_key = f"X_threshold_binary_{target_name.split('_')[-1]}"
                    X_dict[X_key] = target_X
                    logger.info(f"prepare_features: {target_name} 用の特徴量セット {X_key} を作成 ({len(target_X)} 行)")
                else:
                    logger.warning(f"prepare_features: {target_col} の有効なデータがありません")
            else:
                y_dict[target_name] = df[target_col]
        else:
            logger.warning(f"prepare_features: 目標変数 {target_col} がカラムに見つかりません")

    # 高閾値シグナル変数の追加
    high_threshold_cols = [col for col in df.columns if 'high_threshold' in col]
    if high_threshold_cols:
        logger.info(f"prepare_features: 高閾値シグナル変数 {len(high_threshold_cols)}個を目標変数に追加します")
        
        # パターン分析で閾値、方向、期間を抽出
        for col in high_threshold_cols:
            # 例: target_high_threshold_2p_long_3
            parts = col.split('_')
            if len(parts) >= 5:
                threshold_part = parts[3]  # '2p'
                direction = parts[4]        # 'long'
                period = parts[5]           # '3'
                
                # 目標変数名を生成
                # 元の変数名をそのまま使用
                target_name = col
                
                # 目標変数を追加
                y_dict[target_name] = df[col]
                logger.info(f"prepare_features: 高閾値シグナル目標変数 {target_name} を追加")
    
    logger.info(f"prepare_features: 特徴量: {len(feature_cols)}個, 目標変数: {len(y_dict)}個")
    logger.info(f"prepare_features: 特徴量セット: {list(X_dict.keys())}")
    logger.info(f"prepare_features: 目標変数名: {list(y_dict.keys())}")
    
    # 各目標変数のクラスバランスを確認
    for target_name, target_series in y_dict.items():
        if "classification" in target_name:
            value_counts = target_series.value_counts()
            logger.info(f"prepare_features: {target_name} のクラスバランス: {value_counts.to_dict()}")

    logger.info("prepare_features: 特徴量と目標変数の準備を終了します")

    return X_dict, y_dict