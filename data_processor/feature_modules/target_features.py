# data_processor/feature_modules/target_features.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("feature_engineering.target")

def generate_target_features(df: pd.DataFrame, target_periods: list, classification_threshold: float) -> pd.DataFrame:
    """
    予測対象（目標変数）を生成

    Args:
        df: 入力データフレーム
        target_periods: 予測期間のリスト
        classification_threshold: 分類の閾値

    Returns:
        DataFrame: 目標変数を含むデータフレーム
    """
    logger.info("=== 予測対象の目標変数を生成しています ===")
    logger.info(f"分類閾値: {classification_threshold}, 予測期間: {target_periods}")

    # 結果格納用の辞書
    features = {}

    # ボラティリティベースの動的閾値を計算
    volatility = df['close'].pct_change().rolling(20).std().fillna(classification_threshold)
    logger.info(f"ボラティリティの平均: {volatility.mean():.6f}")

    # 価格の平滑化
    smoothed_close = df['close'].rolling(3).mean()

    # 各予測時間軸に対する目標変数を生成
    for period in target_periods:
        logger.info(f"期間 {period} の目標変数を生成中...")

        # 価格変動率（回帰用）
        target_change = df['close'].pct_change(periods=period).shift(-period)
        features[f'target_price_change_pct_{period}'] = target_change

        # 平滑化した価格変動率
        smoothed_change = smoothed_close.pct_change(periods=period).shift(-period)
        features[f'target_smoothed_change_{period}'] = smoothed_change

        # 閾値ベースの3分類 (0: 下落, 1: 上昇, 2: 横ばい)
        features[f'target_threshold_ternary_{period}'] = np.where(
            target_change >= classification_threshold,
            1,  # 閾値以上の上昇
            np.where(
                target_change <= -classification_threshold,
                0,  # 閾値以上の下落
                2   # 閾値未満の変動（横ばい）
            )
        )

        # 閾値ベースの二値分類（上昇 vs 下落、横ばいは除外）
        features[f'target_threshold_binary_{period}'] = np.where(
            target_change >= classification_threshold,
            1,  # 上昇
            np.where(
                target_change <= -classification_threshold,
                0,  # 下落
                np.nan  # 横ばいはNaN (学習から除外)
            )
        )

        # 単純な二値分類（上昇/下落）
        features[f'target_binary_{period}'] = np.where(
            target_change > 0,
            1,  # 上昇
            0   # 下落
        )

    # データフレームに変換
    result_df = pd.DataFrame(features, index=df.index)

    # 生成された目標変数の統計情報をログに出力
    _log_target_statistics(result_df, target_periods)

    logger.info(f"目標変数を含むデータフレームを生成完了。サイズ: {result_df.shape}")
    return result_df

def _log_target_statistics(df: pd.DataFrame, target_periods: list) -> None:
    """
    生成された目標変数の統計情報をログに出力

    Args:
        df: 目標変数を含むデータフレーム
        target_periods: 予測期間のリスト
    """
    logger.info("=== 生成された目標変数の統計情報 ===")

    for period in target_periods:
        logger.info(f"---- 期間 {period} の統計情報 ----")

        # 回帰ターゲット
        price_change_col = f'target_price_change_pct_{period}'
        if price_change_col in df.columns:
            price_change_stats = df[price_change_col].describe()
            logger.info(f"{price_change_col} の統計: min={price_change_stats['min']:.6f}, "
                        f"max={price_change_stats['max']:.6f}, mean={price_change_stats['mean']:.6f}")

        # 3分類
        ternary_col = f'target_threshold_ternary_{period}'
        if ternary_col in df.columns:
            ternary_counts = df[ternary_col].value_counts()
            ternary_pcts = df[ternary_col].value_counts(normalize=True) * 100
            logger.info(f"{ternary_col} のクラス分布:")
            for cls, count in ternary_counts.items():
                pct = ternary_pcts[cls]
                logger.info(f"  クラス {cls}: {count} サンプル ({pct:.2f}%)")

        # 二値分類（NaNを含む）
        binary_col = f'target_threshold_binary_{period}'
        if binary_col in df.columns:
            # NaNを含む値をカウント
            binary_counts_with_nan = df[binary_col].value_counts(dropna=False)

            # NaNの数と割合
            nan_count = df[binary_col].isna().sum()
            nan_pct = (nan_count / len(df)) * 100
            logger.info(f"{binary_col} のNaN: {nan_count} サンプル ({nan_pct:.2f}%)")

            # NaNを除いた有効な値の統計
            valid_binary = df[binary_col].dropna()
            if len(valid_binary) > 0:
                binary_valid_counts = valid_binary.value_counts()
                binary_valid_pcts = valid_binary.value_counts(normalize=True) * 100

                logger.info(f"{binary_col} の有効値クラス分布:")
                for cls, count in binary_valid_counts.items():
                    pct = binary_valid_pcts[cls]
                    logger.info(f"  クラス {cls}: {count} サンプル ({pct:.2f}%)")

                logger.info(f"{binary_col} の有効値数: {len(valid_binary)} ({len(valid_binary)/len(df)*100:.2f}%)")

                # クラスバランス比率
                if len(binary_valid_counts) > 1:
                    class_ratio = binary_valid_counts.min() / binary_valid_counts.max()
                    logger.info(f"{binary_col} のクラスバランス比率: {class_ratio:.4f} (1に近いほど均等)")
            else:
                logger.warning(f"{binary_col} に有効な値がありません")

def verify_target_variables(df: pd.DataFrame, threshold: float = 0.0005) -> Dict[str, Any]:
    """
    目標変数を検証し、問題があれば修正する

    Args:
        df: 目標変数を含むデータフレーム
        threshold: 二値分類の閾値

    Returns:
        Dict: 検証結果のサマリー
    """
    logger.info("目標変数の検証を開始")

    # 結果のサマリー
    summary = {
        "issues_found": False,
        "fixed_issues": [],
        "warnings": []
    }

    # 目標変数の列を抽出
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if not target_cols:
        logger.warning("目標変数が見つかりません")
        summary["issues_found"] = True
        summary["warnings"].append("目標変数が見つかりません")
        return summary

    # 閾値ベースの二値分類ターゲットがあるか確認
    threshold_binary_cols = [col for col in target_cols if 'target_threshold_binary' in col]
    if not threshold_binary_cols:
        logger.warning("閾値ベースの二値分類ターゲットが見つかりません")
        summary["issues_found"] = True
        summary["warnings"].append("閾値ベースの二値分類ターゲットが見つかりません")

        # 価格変動率列から閾値ベースの二値分類ターゲットを生成
        price_change_cols = [col for col in target_cols if 'target_price_change_pct' in col]
        if price_change_cols:
            logger.info("価格変動率列から閾値ベースの二値分類ターゲットを生成します")

            for col in price_change_cols:
                period = col.split('_')[-1]
                target_name = f'target_threshold_binary_{period}'

                # 二値分類ターゲットを生成
                df[target_name] = np.where(
                    df[col] >= threshold,
                    1,  # 上昇
                    np.where(
                        df[col] <= -threshold,
                        0,  # 下落
                        np.nan  # 横ばい
                    )
                )

                # 統計情報
                valid_count = df[target_name].dropna().count()
                nan_count = df[target_name].isna().sum()

                logger.info(f"{target_name} を生成しました: 有効データ {valid_count}行, NaN {nan_count}行")

                # クラスバランス
                if valid_count > 0:
                    class_dist = df[target_name].value_counts(dropna=True)
                    logger.info(f"{target_name} のクラス分布: {class_dist.to_dict()}")

                summary["fixed_issues"].append(f"{target_name} を生成しました")

    # 各目標変数の整合性をチェック
    for col in threshold_binary_cols:
        # NaNチェック
        nan_count = df[col].isna().sum()
        if nan_count == len(df):
            logger.error(f"{col} はすべての値がNaNです")
            summary["issues_found"] = True
            summary["warnings"].append(f"{col} はすべての値がNaNです")
        elif nan_count > 0:
            nan_pct = (nan_count / len(df)) * 100
            if nan_pct > 50:
                logger.warning(f"{col} のNaN割合が高すぎます: {nan_pct:.2f}%")
                summary["warnings"].append(f"{col} のNaN割合が高すぎます: {nan_pct:.2f}%")

        # 有効値のクラスバランスチェック
        valid_values = df[col].dropna()
        if len(valid_values) > 0:
            class_dist = valid_values.value_counts()
            if len(class_dist) < 2:
                logger.error(f"{col} には単一クラスしかありません: {class_dist.to_dict()}")
                summary["issues_found"] = True
                summary["warnings"].append(f"{col} には単一クラスしかありません")
            elif len(class_dist) >= 2:
                class_ratio = class_dist.min() / class_dist.max()
                if class_ratio < 0.1:
                    logger.warning(f"{col} のクラスバランスが極端に偏っています: {class_ratio:.4f}")
                    summary["warnings"].append(f"{col} のクラスバランスが極端に偏っています: {class_ratio:.4f}")

    return summary