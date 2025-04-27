# data_processor/optimized_feature_generator.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

from .feature_modules.feature_generator_base import FeatureGeneratorBase
from .feature_modules.price_features import generate_price_change_features, generate_high_low_distance_features
from .feature_modules.volume_features import generate_volume_change_features
from .feature_modules.moving_average_features import generate_moving_average_features
from .feature_modules.oscillator_features import generate_rsi_features, generate_macd_features, generate_stochastic_features
from .feature_modules.volatility_features import generate_bollinger_bands_features, generate_atr_features
from .feature_modules.advanced_features import generate_advanced_features, generate_fisher_transform
from .feature_modules.target_features import generate_target_features
from .feature_modules.feature_selector import select_important_features

# ロガーの設定
logger = logging.getLogger("feature_engineering")

class OptimizedFeatureGenerator(FeatureGeneratorBase):
    """最適化された特徴量生成クラス"""

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        最適化された特徴量を生成

        Args:
            df: 入力データフレーム

        Returns:
            DataFrame: 特徴量を追加したデータフレーム
        """
        if df.empty:
            logger.warning("入力データが空です")
            return df

        # 処理前のデータをコピー
        result_df = df.copy()

        # 全ての特徴量を生成
        feature_dfs = []

        # 基本データは常に保持
        feature_dfs.append(result_df)

        # 価格変動率と重要なローソク足パターン
        if self.config["features"]["price_change"]:
            feature_dfs.append(generate_price_change_features(result_df, self.config["vwap_period"]))

        # 出来高変動率
        if self.config["features"]["volume_change"]:
            feature_dfs.append(generate_volume_change_features(result_df))

        # 移動平均線と乖離率
        if self.config["features"]["moving_averages"]:
            feature_dfs.append(generate_moving_average_features(result_df, self.config["ma_periods"]))

        # RSI（複数期間）
        if self.config["features"]["rsi"]:
            feature_dfs.append(generate_rsi_features(result_df, self.config["rsi_periods"]))

        # 高値/安値からの距離
        if self.config["features"]["high_low_distance"]:
            feature_dfs.append(generate_high_low_distance_features(result_df))

        # ボリンジャーバンドと関連指標
        if self.config["features"]["bollinger_bands"]:
            feature_dfs.append(generate_bollinger_bands_features(
                result_df, 
                self.config["bollinger_period"], 
                self.config["bollinger_std"]
            ))

        # MACD
        if self.config["features"]["macd"]:
            feature_dfs.append(generate_macd_features(result_df, self.config["macd_params"]))

        # ストキャスティクス
        if self.config["features"]["stochastic"]:
            feature_dfs.append(generate_stochastic_features(result_df, self.config["stochastic_params"]))

        # ATR関連特徴量
        if self.config["features"]["advanced_features"]:
            feature_dfs.append(generate_atr_features(result_df, self.config["atr_period"]))

        # 高度な特徴量
        if self.config["features"]["advanced_features"]:
            feature_dfs.append(generate_advanced_features(result_df))

        # 目標変数（ターゲット）の生成
        feature_dfs.append(generate_target_features(
            result_df, 
            self.config["target_periods"],
            self.config["classification_threshold"]
        ))

        # 全ての特徴量を結合
        result_df = pd.concat(feature_dfs, axis=1)

        # 重複列を削除
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        # NaNを含む行の処理
        result_df = result_df.dropna()

        # フィッシャー変換RSI（RSIをより線形に変換）を結合後に計算
        fisher_transform_df = generate_fisher_transform(result_df)
        if not fisher_transform_df.empty:
            result_df = pd.concat([result_df, fisher_transform_df], axis=1)

        # 特徴量の重要度に基づいて選別
        result_df = select_important_features(result_df)

        logger.info(f"特徴量を生成しました。カラム数: {len(result_df.columns)}, 行数: {len(result_df)}")

        return result_df


# 実行部分（外部から呼び出す場合）
def generate_optimized_features(config=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    最適化された特徴量を生成して保存する関数

    Args:
        config: 設定辞書またはNone（デフォルト設定を使用）

    Returns:
        Tuple[pd.DataFrame, Dict]: (特徴量DataFrame, レポート)
    """
    generator = OptimizedFeatureGenerator(config)

    # データ読み込み
    df = generator.load_data()

    # 特徴量生成
    feature_df = generator.generate_features(df)

    # 特徴量保存
    generator.save_features(feature_df)

    # レポート生成
    report = generator.generate_feature_report(feature_df)

    return feature_df, report
