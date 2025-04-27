# model_builder/evaluators/model_evaluator.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from ..config.default_config import get_default_evaluator_config

from .base_evaluator import BaseEvaluator
from .regression_evaluator import RegressionEvaluator
from .classification_evaluator import ClassificationEvaluator
from ..utils.data.data_loader import load_data
from ..utils.data.data_splitter import prepare_test_data
from ..utils.model_io.model_loader import load_models
from ..utils.reporting.report_generator import generate_evaluation_report
from ..utils.reporting.report_serializer import save_evaluation_report

class ModelEvaluator(BaseEvaluator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        モデル評価クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)
        
        # 専用の評価器を初期化
        # 必要な設定を評価器に渡す
        regression_config = {
            "model_dir": self.config.get("model_dir", "models"),
            "output_dir": self.config.get("output_dir", "evaluation")
        }
        self.regression_evaluator = RegressionEvaluator(regression_config)
        
        classification_config = {
            "model_dir": self.config.get("model_dir", "models"),
            "output_dir": self.config.get("output_dir", "evaluation")
        }
        self.classification_evaluator = ClassificationEvaluator(classification_config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return get_default_evaluator_config()

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

    def load_models(self) -> bool:
        """
        保存されたモデルを読み込む

        Returns:
            bool: 読み込みが成功したかどうか
        """
        self.logger.info("load_models: モデルの読み込みを開始します")
        model_dir = self.config.get("model_dir", "models")
        target_periods = self.config.get("target_periods", [1, 2, 3])
        
        loaded_models = load_models(model_dir, target_periods)
        self.models.update(loaded_models)
        
        self.logger.info(f"load_models: {len(loaded_models)} 個のモデルを読み込みました")
        self.logger.info("load_models: モデルの読み込みを終了します")
        return len(loaded_models) > 0

    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        テストデータを準備

        Args:
            df: 入力データフレーム

        Returns:
            Tuple: (特徴量DataFrame, 目標変数のDict)
        """
        self.logger.info("prepare_test_data: テストデータの準備を開始します")
        test_size = self.config.get("test_size", 0.2)
        target_periods = self.config.get("target_periods", [1, 2, 3])
        
        X_test, y_test = prepare_test_data(df, test_size, target_periods)
        
        self.logger.info("prepare_test_data: テストデータの準備を終了します")
        return X_test, y_test

    def evaluate_models(self, X_test: pd.DataFrame, y_test: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        モデルを評価

        Args:
            X_test: テスト特徴量DataFrame
            y_test: テスト目標変数のDict

        Returns:
            Dict: 評価結果
        """
        self.logger.info("evaluate_models: モデルの評価を開始します")
        if X_test.empty or not y_test:
            self.logger.warning("evaluate_models: テストデータが空です")
            return {}

        evaluation_results = {
            "regression": {},
            "classification": {}
        }

        # 回帰モデル（価格変動率予測）の評価
        self.logger.info("evaluate_models: 回帰モデルの評価を開始します")
        for period in self.config.get("target_periods", [1, 2, 3]):
            model_key = f"regression_{period}"
            target_key = f"regression_{period}"

            if model_key in self.models and target_key in y_test:
                model = self.models[model_key]
                
                # 専用の評価器を使って評価
                result = self.regression_evaluator.evaluate(
                    model, X_test, y_test[target_key], period
                )
                
                evaluation_results["regression"][f"period_{period}"] = result

        # 分類モデル（価格変動方向予測）の評価
        self.logger.info("evaluate_models: 分類モデルの評価を開始します")
        for period in self.config.get("target_periods", [1, 2, 3]):
            model_key = f"classification_{period}"
            target_key = f"classification_{period}"

            if model_key in self.models and target_key in y_test:
                model = self.models[model_key]
                
                # 専用の評価器を使って評価
                result = self.classification_evaluator.evaluate(
                    model, X_test, y_test[target_key], period
                )
                
                evaluation_results["classification"][f"period_{period}"] = result

        self.logger.info("evaluate_models: モデルの評価を終了します")
        return evaluation_results

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        評価結果の要約レポートを生成

        Args:
            evaluation_results: 評価結果のDict

        Returns:
            Dict: 評価レポート
        """
        self.logger.info("generate_evaluation_report: 評価レポートの生成を開始します")
        report = generate_evaluation_report(evaluation_results)
        self.logger.info("generate_evaluation_report: 評価レポートの生成を終了します")
        return report

    def save_evaluation_report(self, report: Dict[str, Any]) -> bool:
        """
        評価レポートを保存

        Args:
            report: 保存する評価レポート

        Returns:
            bool: 保存が成功したかどうか
        """
        self.logger.info("save_evaluation_report: 評価レポートの保存を開始します")
        output_dir = self.config.get("output_dir", "evaluation")
        result = save_evaluation_report(report, output_dir)
        if result:
            self.logger.info("save_evaluation_report: 評価レポートの保存が成功しました")
        else:
            self.logger.error("save_evaluation_report: 評価レポートの保存に失敗しました")
        return result

# 実行部分（外部から呼び出す場合）
def evaluate_models(config=None):
    """
    モデルの評価を実行する関数

    Args:
        config: 設定辞書またはNone（デフォルト設定を使用）

    Returns:
        Dict: 評価結果のレポート
    """
    evaluator = ModelEvaluator(config)

    # モデル読み込み
    if not evaluator.load_models():
        evaluator.logger.error("モデルの読み込みに失敗しました")
        return {}

    # データ読み込み
    df = evaluator.load_data()

    # テストデータの準備
    X_test, y_test = evaluator.prepare_test_data(df)

    # モデルの評価
    evaluation_results = evaluator.evaluate_models(X_test, y_test)

    # 評価レポートの生成
    report = evaluator.generate_evaluation_report(evaluation_results)

    # 評価レポートの保存
    evaluator.save_evaluation_report(report)

    return report