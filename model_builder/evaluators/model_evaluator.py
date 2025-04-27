# model_builder/evaluators/model_evaluator.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from .base_evaluator import BaseEvaluator
from .regression_evaluator import RegressionEvaluator
from .classification_evaluator import ClassificationEvaluator
from ..config.default_config import get_default_evaluator_config
from ..utils.data_utils import load_data

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
        model_dir = Path(self.config.get("model_dir", "models"))

        if not model_dir.exists():
            self.logger.error(f"load_models: モデルディレクトリ {model_dir} が存在しません")
            return False

        # 回帰モデルの読み込み
        for period in self.config.get("target_periods", [1, 2, 3]):
            regression_model_path = model_dir / f"regression_model_period_{period}.joblib"
            if regression_model_path.exists():
                model = self.load_model(regression_model_path)
                if model:
                    self.models[f"regression_{period}"] = model
                    self.logger.info(f"回帰モデル（{period}期先）を読み込みました")
                else:
                    return False
            else:
                self.logger.warning(f"回帰モデル {regression_model_path} が見つかりません")

        # 分類モデルの読み込み
        for period in self.config.get("target_periods", [1, 2, 3]):
            classification_model_path = model_dir / f"classification_model_period_{period}.joblib"
            if classification_model_path.exists():
                model = self.load_model(classification_model_path)
                if model:
                    self.models[f"classification_{period}"] = model
                    self.logger.info(f"分類モデル（{period}期先）を読み込みました")
                else:
                    return False
            else:
                self.logger.warning(f"分類モデル {classification_model_path} が見つかりません")

        self.logger.info(f"load_models: {len(self.models)} 個のモデルを読み込みました")
        self.logger.info("load_models: モデルの読み込みを終了します")
        return len(self.models) > 0

    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        テストデータを準備

        Args:
            df: 入力データフレーム

        Returns:
            Tuple: (特徴量DataFrame, 目標変数のDict)
        """
        self.logger.info("prepare_test_data: テストデータの準備を開始します")
        if df.empty:
            self.logger.warning("prepare_test_data: 入力データが空です")
            return pd.DataFrame(), {}

        # 時系列データなので、最後の一定割合をテストデータとする
        test_size = int(len(df) * self.config.get("test_size", 0.2))
        test_df = df.iloc[-test_size:].copy()

        # 目標変数（各予測期間に対して）
        y_test = {}
        for period in self.config.get("target_periods", [1, 2, 3]):
            # 回帰目標（価格変動率）
            y_test[f"regression_{period}"] = test_df[f"target_price_change_pct_{period}"]
            # 分類目標（価格変動方向）
            y_test[f"classification_{period}"] = test_df[f"target_price_direction_{period}"]

        # 特徴量（目標変数を除く）
        feature_cols = [col for col in test_df.columns if not col.startswith("target_")]
        X_test = test_df[feature_cols]

        self.logger.info(f"prepare_test_data: テストデータ: {len(X_test)}行, 特徴量: {len(feature_cols)}個")
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
        report = {
            "regression": {},
            "classification": {}
        }

        # 回帰モデルの評価結果
        if "regression" in evaluation_results:
            for period_key, result in evaluation_results["regression"].items():
                report["regression"][period_key] = {
                    "mae": result["mae"]
                }

        # 分類モデルの評価結果
        if "classification" in evaluation_results:
            for period_key, result in evaluation_results["classification"].items():
                # クラスごとの精度
                class_precision = {}
                report_dict = result["classification_report"]

                if "-1" in report_dict:
                    class_precision["下落"] = {
                        "precision": report_dict["-1"]["precision"],
                        "recall": report_dict["-1"]["recall"],
                        "f1-score": report_dict["-1"]["f1-score"],
                        "support": report_dict["-1"]["support"]
                    }

                if "0" in report_dict:
                    class_precision["横ばい"] = {
                        "precision": report_dict["0"]["precision"],
                        "recall": report_dict["0"]["recall"],
                        "f1-score": report_dict["0"]["f1-score"],
                        "support": report_dict["0"]["support"]
                    }

                if "1" in report_dict:
                    class_precision["上昇"] = {
                        "precision": report_dict["1"]["precision"],
                        "recall": report_dict["1"]["recall"],
                        "f1-score": report_dict["1"]["f1-score"],
                        "support": report_dict["1"]["support"]
                    }

                report["classification"][period_key] = {
                    "accuracy": result["accuracy"],
                    "class_metrics": class_precision,
                    "confusion_matrix": result["confusion_matrix"]
                }

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
        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(self.config.get("output_dir", "evaluation"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # レポートをJSONファイルに保存
        report_path = output_dir / "model_evaluation_report.json"
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"save_evaluation_report: 評価レポートを {report_path} に保存しました")
            self.logger.info("save_evaluation_report: 評価レポートの保存を終了します")
            return True
        except Exception as e:
            self.logger.error(f"レポート保存エラー: {e}")
            return False

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