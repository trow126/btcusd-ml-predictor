# model_builder/trainers/model_trainer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from ..config.default_config import get_default_trainer_config

from .base_trainer import BaseTrainer
from .regression_trainer import RegressionTrainer
from .classification_trainer import ClassificationTrainer
from .binary_classification_trainer import BinaryClassificationTrainer
from .threshold_binary_classification_trainer import ThresholdBinaryClassificationTrainer
from ..utils.data.data_loader import load_data
from ..utils.data.feature_selector import prepare_features
from ..utils.data.data_splitter import train_test_split
from ..utils.reporting.report_formatter import generate_training_report

class ModelTrainer(BaseTrainer):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        モデルトレーニングクラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)
        
        # 専用のトレーナーを初期化
        # グローバル設定と特定のモデル設定をマージして渡す
        regression_config = {
            "output_dir": self.config.get("output_dir", "models")
        }
        regression_config.update(self.config.get("regression", {}))
        self.regression_trainer = RegressionTrainer(regression_config)
        
        classification_config = {
            "output_dir": self.config.get("output_dir", "models")
        }
        classification_config.update(self.config.get("classification", {}))
        self.classification_trainer = ClassificationTrainer(classification_config)
        
        binary_classification_config = {
            "output_dir": self.config.get("output_dir", "models")
        }
        binary_classification_config.update(self.config.get("binary_classification", {}))
        self.binary_classification_trainer = BinaryClassificationTrainer(binary_classification_config)
        
        # 閾値ベースの二値分類トレーナーの初期化
        threshold_binary_config = {
            "output_dir": self.config.get("output_dir", "models")
        }
        threshold_binary_config.update(self.config.get("threshold_binary_classification", {}))
        self.threshold_binary_classification_trainer = ThresholdBinaryClassificationTrainer(threshold_binary_config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return get_default_trainer_config()

    def load_data(self) -> pd.DataFrame:
        """
        特徴量データを読み込む

        Returns:
            DataFrame: 読み込んだデータ
        """
        return load_data(
            self.config.get("input_dir", "data/processed"), 
            self.config.get("input_filename", "btcusd_5m_features.csv")
        )

    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        特徴量と目標変数を準備

        Args:
            df: 入力データフレーム

        Returns:
            Tuple: (特徴量のDict, 目標変数のDict)
        """
        return prepare_features(
            df, 
            self.config.get("feature_groups", {"price": True, "volume": True, "technical": True}), 
            self.config.get("target_periods", [1, 2, 3])
        )

    def train_test_split(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]) -> Tuple[
        Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, pd.Series]
    ]:
        """
        時系列を考慮してトレーニングデータとテストデータに分割

        Args:
            X: 特徴量DataFrame
            y_dict: 目標変数のDict

        Returns:
            Tuple: (X_train, X_test, y_train, y_test) の辞書
        """
        return train_test_split(X, y_dict, self.config.get("test_size", 0.2))

    def train_regression_models(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        回帰モデル（価格変動率予測）をトレーニング

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict

        Returns:
            Dict: トレーニング結果
        """
        regression_results = {}
        
        self.logger.info("train_regression_models: 回帰モデルのトレーニングを開始します")
        # 各予測期間に対してモデルをトレーニング
        for period in self.config.get("target_periods", [1, 2, 3]):
            target_name = f"regression_{period}"

            if target_name not in y_train or target_name not in y_test:
                self.logger.warning(f"目標変数 {target_name} が見つかりません")
                continue

            # 専用トレーナーを使ってモデルをトレーニング
            result = self.regression_trainer.train(
                X_train, X_test, y_train, y_test, period
            )
            
            regression_results[target_name] = result

        return regression_results

    def train_classification_models(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        分類モデル（価格変動方向予測）をトレーニング

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict

        Returns:
            Dict: トレーニング結果
        """
        classification_results = {}

        self.logger.info("train_classification_models: 分類モデルのトレーニングを開始します")
        # 各予測期間に対してモデルをトレーニング
        for period in self.config.get("target_periods", [1, 2, 3]):
            target_name = f"classification_{period}"

            if target_name not in y_train or target_name not in y_test:
                self.logger.warning(f"目標変数 {target_name} が見つかりません")
                continue

            # 専用トレーナーを使ってモデルをトレーニング
            result = self.classification_trainer.train(
                X_train, X_test, y_train, y_test, period
            )
            
            classification_results[target_name] = result

        return classification_results
        
    def train_binary_classification_models(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        二値分類モデル（上昇/下落予測）をトレーニング

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict

        Returns:
            Dict: トレーニング結果
        """
        binary_classification_results = {}

        self.logger.info("train_binary_classification_models: 二値分類モデルのトレーニングを開始します")
        # 各予測期間に対してモデルをトレーニング
        for period in self.config.get("target_periods", [1, 2, 3]):
            target_name = f"binary_classification_{period}"

            if target_name not in y_train or target_name not in y_test:
                self.logger.warning(f"目標変数 {target_name} が見つかりません")
                continue

            # 専用トレーナーを使ってモデルをトレーニング
            result = self.binary_classification_trainer.train(
                X_train, X_test, y_train, y_test, period
            )
            
            binary_classification_results[target_name] = result

        return binary_classification_results
        
    def train_threshold_binary_classification_models(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        閾値ベースの二値分類モデル（有意な上昇/下落予測）をトレーニング
        横ばいデータを除外して学習する改良版

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict

        Returns:
            Dict: トレーニング結果
        """
        threshold_binary_classification_results = {}

        self.logger.info("===== train_threshold_binary_classification_models: 閾値ベースの二値分類モデルのトレーニングを開始します =====")
        
        # 利用可能な目標変数を確認
        for key in y_train.keys():
            if "threshold" in key:
                self.logger.info(f"閾値関連の目標変数が見つかりました: {key}")
        
        # 各予測期間に対してモデルをトレーニング
        for period in self.config.get("target_periods", [1, 2, 3]):
            target_name = f"threshold_binary_classification_{period}"
            
            # 目標変数を確認
            if target_name not in y_train or target_name not in y_test:
                self.logger.warning(f"目標変数 {target_name} が見つかりません")
                threshold_binary_classification_results[target_name] = {
                    "error": "missing_target",
                    "message": f"目標変数 {target_name} が見つかりません"
                }
                continue
                
            # クラスバランスを確認（デバッグ情報）
            valid_samples = y_train[target_name].dropna()
            class_balance = valid_samples.value_counts()
            self.logger.info(f"{target_name} のクラスバランス: {class_balance.to_dict()}")
            
            # サンプル数が極端に少ない場合は警告
            if len(valid_samples) < 500:  # 少なくとも500サンプルは欲しい
                self.logger.warning(f"{target_name} の有効サンプル数が少なすぎます: {len(valid_samples)}")
                if len(valid_samples) < 100:  # 極端に少ない場合はスキップ
                    self.logger.error(f"{target_name} のサンプル数が極端に少ないため、モデルトレーニングをスキップします")
                    threshold_binary_classification_results[target_name] = {
                        "error": "insufficient_samples",
                        "message": f"有効サンプル数が極端に少なすぎます: {len(valid_samples)}",
                        "class_balance": class_balance.to_dict()
                    }
                    continue

            try:
                # 専用トレーナーを使ってモデルをトレーニング
                self.logger.info(f"閾値ベースの二値分類モデル（{period}期先）のトレーニングを開始します")
                result = self.threshold_binary_classification_trainer.train(
                    X_train, X_test, y_train, y_test, period
                )
                
                threshold_binary_classification_results[target_name] = result
                self.logger.info(f"閾値ベースの二値分類モデル（{period}期先）のトレーニングが完了しました")
            except Exception as e:
                self.logger.error(f"{target_name} のトレーニング中にエラーが発生しました: {str(e)}")
                import traceback
                self.logger.error(f"トレースバック: {traceback.format_exc()}")
                threshold_binary_classification_results[target_name] = {
                    "error": "training_error",
                    "message": str(e)
                }
                
        self.logger.info("===== train_threshold_binary_classification_models: 閾値ベースの二値分類モデルのトレーニングが完了しました =====")
        return threshold_binary_classification_results

    def generate_training_report(
        self, regression_results: Dict[str, Any], classification_results: Dict[str, Any],
        binary_classification_results: Dict[str, Any] = None,
        threshold_binary_classification_results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        トレーニング結果の要約レポートを生成

        Args:
            regression_results: 回帰モデルのトレーニング結果
            classification_results: 分類モデルのトレーニング結果
            binary_classification_results: 二値分類モデルのトレーニング結果
            threshold_binary_classification_results: 閾値ベースの二値分類モデルのトレーニング結果

        Returns:
            Dict: トレーニング結果のレポート
        """
        self.logger.info("generate_training_report: トレーニングレポートの生成を開始します")
        
        # レポート生成のロジックを更新
        all_results = {
            "regression": regression_results,
            "classification": classification_results
        }
        
        # 二値分類結果がある場合は追加
        if binary_classification_results:
            all_results["binary_classification"] = binary_classification_results
            
        # 閾値ベースの二値分類結果がある場合は追加
        if threshold_binary_classification_results:
            all_results["threshold_binary_classification"] = threshold_binary_classification_results
            
        report = generate_training_report(all_results)
        
        self.logger.info("generate_training_report: トレーニングレポートの生成を終了します")
        return report

# 実行部分（外部から呼び出す場合）
def train_models(config=None):
    """
    モデルのトレーニングを実行する関数

    Args:
        config: 設定辞書またはNone（デフォルト設定を使用）

    Returns:
        Dict: トレーニング結果のレポート
    """
    trainer = ModelTrainer(config)
    logger = trainer.logger

    # データ読み込み
    df = trainer.load_data()

    # 特徴量と目標変数の準備
    X_dict, y_dict = trainer.prepare_features(df)

    # トレーニングデータとテストデータに分割
    X_train, X_test, y_train, y_test = trainer.train_test_split(X_dict["X"], y_dict)

    # 回帰モデル（価格変動率予測）のトレーニング
    regression_results = trainer.train_regression_models(X_train, X_test, y_train, y_test)

    # 分類モデル（価格変動方向予測）のトレーニング
    classification_results = trainer.train_classification_models(X_train, X_test, y_train, y_test)
    
    # 二値分類モデル（上昇/下落予測）のトレーニング（設定で有効化されている場合）
    binary_classification_results = None
    if trainer.config.get("use_binary_classification", False):
        binary_classification_results = trainer.train_binary_classification_models(X_train, X_test, y_train, y_test)

    # 閾値ベースの二値分類モデルのトレーニング
    threshold_binary_classification_results = None
    if trainer.config.get("use_threshold_binary_classification", True):  # デフォルトで有効化
        logger.info("====== 閾値ベース分類モデルのトレーニングを開始します ======")
        logger.info(f"ターゲット変数の確認: {list(y_train.keys())}")
        for period in trainer.config.get("target_periods", [1, 2, 3]):
            target_key = f"threshold_binary_classification_{period}"
            logger.info(f"確認: {target_key} が存在するか? {target_key in y_train}")
            if target_key in y_train:
                # 値の分布を確認
                valid_values = y_train[target_key].dropna()
                logger.info(f"{target_key} の有効値数: {len(valid_values)}")
                logger.info(f"{target_key} の値のカウント: {valid_values.value_counts().to_dict()}")
        
        threshold_binary_classification_results = trainer.train_threshold_binary_classification_models(X_train, X_test, y_train, y_test)
        
        # 結果のサマリーを出力
        success_count = 0
        error_count = 0
        for key, result in threshold_binary_classification_results.items():
            if "error" in result:
                error_count += 1
                logger.error(f"{key} でエラー: {result['error']} - {result['message']}")
            else:
                success_count += 1
                logger.info(f"{key} のトレーニング成功: 精度 {result.get('accuracy', 'N/A')}")
                
        logger.info(f"閾値ベース分類モデルのトレーニング結果: 成功 {success_count}, 失敗 {error_count}")
        
    # トレーニング結果のレポート生成
    report = trainer.generate_training_report(
        regression_results, 
        classification_results,
        binary_classification_results,
        threshold_binary_classification_results
    )

    return report