# model_builder/trainers/model_trainer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from .base_trainer import BaseTrainer
from .regression_trainer import RegressionTrainer
from .classification_trainer import ClassificationTrainer
from ..config.default_config import get_default_trainer_config
from ..utils.data_utils import load_data, prepare_features, train_test_split

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

    def generate_training_report(
        self, regression_results: Dict[str, Any], classification_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        トレーニング結果の要約レポートを生成

        Args:
            regression_results: 回帰モデルのトレーニング結果
            classification_results: 分類モデルのトレーニング結果

        Returns:
            Dict: トレーニング結果のレポート
        """
        self.logger.info("generate_training_report: トレーニングレポートの生成を開始します")
        report = {
            "regression": {},
            "classification": {}
        }

        # 回帰モデルの結果
        for target_name, result in regression_results.items():
            period = int(target_name.split("_")[1])

            # 上位の特徴量重要度
            report["regression"][f"period_{period}"] = {
                "mae": result["mae"],
                "top_features": result["feature_importance"]
            }

        # 分類モデルの結果
        for target_name, result in classification_results.items():
            period = int(target_name.split("_")[1])

            report["classification"][f"period_{period}"] = {
                "accuracy": result["accuracy"],
                "class_accuracy": {
                    "-1 (下落)": result["classification_report"]["-1"]["precision"],
                    "0 (横ばい)": result["classification_report"]["0"]["precision"],
                    "1 (上昇)": result["classification_report"]["1"]["precision"]
                },
                "confusion_matrix": result["confusion_matrix"].tolist(),
                "top_features": result["feature_importance"]
            }

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

    # トレーニング結果のレポート生成
    report = trainer.generate_training_report(regression_results, classification_results)

    return report