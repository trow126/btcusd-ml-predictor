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

        logger.info(f"特徴量生成を開始。入力データサイズ: {df.shape}")
        logger.info(f"設定内容: 分類閾値={self.config['classification_threshold']}, 目標期間={self.config['target_periods']}")

        # 処理前のデータをコピー
        result_df = df.copy()

        # 全ての特徴量を生成
        feature_dfs = []

        # 基本データは常に保持
        feature_dfs.append(result_df)

        # 価格変動率と重要なローソク足パターン
        if self.config["features"]["price_change"]:
            logger.info("価格変動率特徴量を生成中...")
            price_features = generate_price_change_features(result_df, self.config["vwap_period"])
            feature_dfs.append(price_features)

        # 出来高変動率
        if self.config["features"]["volume_change"]:
            logger.info("出来高変動率特徴量を生成中...")
            volume_features = generate_volume_change_features(result_df)
            feature_dfs.append(volume_features)

        # 移動平均線と乖離率
        if self.config["features"]["moving_averages"]:
            logger.info("移動平均線特徴量を生成中...")
            ma_features = generate_moving_average_features(result_df, self.config["ma_periods"])
            feature_dfs.append(ma_features)

        # RSI（複数期間）
        if self.config["features"]["rsi"]:
            logger.info("RSI特徴量を生成中...")
            rsi_features = generate_rsi_features(result_df, self.config["rsi_periods"])
            feature_dfs.append(rsi_features)

        # 高値/安値からの距離
        if self.config["features"]["high_low_distance"]:
            logger.info("高値/安値特徴量を生成中...")
            hl_features = generate_high_low_distance_features(result_df)
            feature_dfs.append(hl_features)

        # ボリンジャーバンドと関連指標
        if self.config["features"]["bollinger_bands"]:
            logger.info("ボリンジャーバンド特徴量を生成中...")
            bb_features = generate_bollinger_bands_features(
                result_df, 
                self.config["bollinger_period"], 
                self.config["bollinger_std"]
            )
            feature_dfs.append(bb_features)

        # MACD
        if self.config["features"]["macd"]:
            logger.info("MACD特徴量を生成中...")
            macd_features = generate_macd_features(result_df, self.config["macd_params"])
            feature_dfs.append(macd_features)

        # ストキャスティクス
        if self.config["features"]["stochastic"]:
            logger.info("ストキャスティクス特徴量を生成中...")
            stoch_features = generate_stochastic_features(result_df, self.config["stochastic_params"])
            feature_dfs.append(stoch_features)

        # ATR関連特徴量
        if self.config["features"]["advanced_features"]:
            logger.info("ATR特徴量を生成中...")
            atr_features = generate_atr_features(result_df, self.config["atr_period"])
            feature_dfs.append(atr_features)

        # 高度な特徴量
        if self.config["features"]["advanced_features"]:
            logger.info("高度な特徴量を生成中...")
            adv_features = generate_advanced_features(result_df)
            feature_dfs.append(adv_features)

        # 目標変数（ターゲット）の生成
        logger.info("目標変数を生成中...")
        
        # 高閾値シグナルモデル用の設定をチェック
        high_threshold_config = self.config.get("high_threshold_models", {})
        if high_threshold_config:
            logger.info(f"高閾値シグナル設定を検出: {high_threshold_config}")
            
        target_features = generate_target_features(
            result_df, 
            self.config["target_periods"],
            self.config["classification_threshold"],
            high_threshold_config=high_threshold_config
        )
        feature_dfs.append(target_features)

        # 全ての特徴量を結合
        logger.info("すべての特徴量を結合中...")
        result_df = pd.concat(feature_dfs, axis=1)

        # 重複列を削除
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        logger.info(f"重複列削除後のカラム数: {len(result_df.columns)}")

        # NaNを含む行の処理
        nan_rows_before = len(result_df)
        result_df = result_df.dropna()
        nan_rows_removed = nan_rows_before - len(result_df)
        logger.info(f"NaNを含む行を削除: {nan_rows_removed}行 ({nan_rows_removed/nan_rows_before*100:.2f}%)")

        # フィッシャー変換RSI（RSIをより線形に変換）を結合後に計算
        fisher_transform_df = generate_fisher_transform(result_df)
        if not fisher_transform_df.empty:
            result_df = pd.concat([result_df, fisher_transform_df], axis=1)

        # 特徴量の重要度に基づいて選別
        result_df = select_important_features(result_df)

        # 目標変数のカラムを確認
        target_cols = [col for col in result_df.columns if col.startswith("target_")]
        logger.info(f"生成された目標変数列: {target_cols}")
        
        # 特に閾値ベースの二値分類目標変数を確認
        threshold_binary_cols = [col for col in target_cols if "threshold_binary" in col]
        for col in threshold_binary_cols:
            # 有効なサンプル数
            valid_samples = result_df[col].dropna().count()
            total_samples = len(result_df)
            logger.info(f"{col} の有効サンプル数: {valid_samples} ({valid_samples/total_samples*100:.2f}%)")
            
            # クラス分布
            if valid_samples > 0:
                class_dist = result_df[col].value_counts(dropna=True)
                logger.info(f"{col} のクラス分布: {class_dist.to_dict()}")

        logger.info(f"特徴量生成が完了しました。カラム数: {len(result_df.columns)}, 行数: {len(result_df)}")

        return result_df


# 実行部分（外部から呼び出す場合）
def generate_optimized_features(config=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # デバッグ情報
    logger.info("DEBUG: generate_optimized_features関数が呼び出されました")
    if config:
        logger.info(f"DEBUG: 渡された設定キー: {list(config.keys())}")
        if "high_threshold_models" in config:
            logger.info(f"DEBUG: high_threshold_modelsの内容: {config['high_threshold_models']}")
    
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
