# model_builder/model_evaluator.py
import pandas as pd
import numpy as np
import logging
import joblib
import os
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional, Union

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_evaluator")

class ModelEvaluator:
    def __init__(self, config=None):
        """
        モデル評価クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        self.config = config if config else self._get_default_config()
        self.models = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "input_dir": "data/processed",
            "input_filename": "btcusd_5m_features.csv",
            "model_dir": "models",
            "output_dir": "evaluation",
            "target_periods": [1, 2, 3],  # 予測対象期間 (1=5分後, 2=10分後, 3=15分後)
            "test_size": 0.2,             # テストデータの割合
            "classification_threshold": 0.0005  # 分類閾値（±0.05%）
        }

    def load_data(self) -> pd.DataFrame:
        """
        特徴量データを読み込む

        Returns:
            DataFrame: 読み込んだデータ
        """
        input_path = Path(self.config["input_dir"]) / self.config["input_filename"]
        logger.info(f"データを {input_path} から読み込みます")

        try:
            df = pd.read_csv(input_path, index_col="timestamp", parse_dates=True)
            logger.info(f"{len(df)} 行のデータを読み込みました")
            return df
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return pd.DataFrame()

    def load_models(self) -> bool:
        """
        保存されたモデルを読み込む

        Returns:
            bool: 読み込みが成功したかどうか
        """
        model_dir = Path(self.config["model_dir"])

        if not model_dir.exists():
            logger.error(f"モデルディレクトリ {model_dir} が存在しません")
            return False

        # 回帰モデルの読み込み
        for period in self.config["target_periods"]:
            regression_model_path = model_dir / f"regression_model_period_{period}.joblib"
            if regression_model_path.exists():
                try:
                    self.models[f"regression_{period}"] = joblib.load(regression_model_path)
                    logger.info(f"回帰モデル（{period}期先）を読み込みました")
                except Exception as e:
                    logger.error(f"回帰モデル読み込みエラー: {e}")
                    return False
            else:
                logger.warning(f"回帰モデル {regression_model_path} が見つかりません")

        # 分類モデルの読み込み
        for period in self.config["target_periods"]:
            classification_model_path = model_dir / f"classification_model_period_{period}.joblib"
            if classification_model_path.exists():
                try:
                    self.models[f"classification_{period}"] = joblib.load(classification_model_path)
                    logger.info(f"分類モデル（{period}期先）を読み込みました")
                except Exception as e:
                    logger.error(f"分類モデル読み込みエラー: {e}")
                    return False
            else:
                logger.warning(f"分類モデル {classification_model_path} が見つかりません")

        return len(self.models) > 0

    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        テストデータを準備

        Args:
            df: 入力データフレーム

        Returns:
            Tuple: (特徴量DataFrame, 目標変数のDict)
        """
        if df.empty:
            logger.warning("入力データが空です")
            return pd.DataFrame(), {}

        # 時系列データなので、最後の一定割合をテストデータとする
        test_size = int(len(df) * self.config["test_size"])
        test_df = df.iloc[-test_size:].copy()

        # 目標変数（各予測期間に対して）
        y_test = {}
        for period in self.config["target_periods"]:
            # 回帰目標（価格変動率）
            y_test[f"regression_{period}"] = test_df[f"target_price_change_pct_{period}"]
            # 分類目標（価格変動方向）
            y_test[f"classification_{period}"] = test_df[f"target_price_direction_{period}"]

        # 特徴量（目標変数を除く）
        feature_cols = [col for col in test_df.columns if not col.startswith("target_")]
        X_test = test_df[feature_cols]

        logger.info(f"テストデータ: {len(X_test)}行, 特徴量: {len(feature_cols)}個")

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
        if X_test.empty or not y_test:
            logger.warning("テストデータが空です")
            return {}

        evaluation_results = {
            "regression": {},
            "classification": {}
        }

        # 回帰モデル（価格変動率予測）の評価
        for period in self.config["target_periods"]:
            model_key = f"regression_{period}"
            target_key = f"regression_{period}"

            if model_key in self.models and target_key in y_test:
                model = self.models[model_key]

                # 予測
                y_pred = model.predict(X_test)

                # 評価
                mae = mean_absolute_error(y_test[target_key], y_pred)

                # 結果を保存
                evaluation_results["regression"][f"period_{period}"] = {
                    "mae": mae,
                    "predictions": {
                        "true": y_test[target_key].values.tolist()[:10],  # 最初の10個のみ表示
                        "pred": y_pred.tolist()[:10]
                    }
                }

                logger.info(f"回帰モデル（{period}期先）評価 - MAE: {mae:.6f}")

        # 分類モデル（価格変動方向予測）の評価
        for period in self.config["target_periods"]:
            model_key = f"classification_{period}"
            target_key = f"classification_{period}"

            if model_key in self.models and target_key in y_test:
                model = self.models[model_key]

                # 予測
                y_pred_proba = model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1) - 1  # 0, 1, 2を-1, 0, 1に変換

                # 評価
                accuracy = accuracy_score(y_test[target_key], y_pred)

                # 混同行列
                cm = confusion_matrix(y_test[target_key], y_pred)

                # 分類レポート
                report = classification_report(y_test[target_key], y_pred, output_dict=True)

                # 結果を保存
                evaluation_results["classification"][f"period_{period}"] = {
                    "accuracy": accuracy,
                    "confusion_matrix": cm.tolist(),
                    "classification_report": report,
                    "predictions": {
                        "true": y_test[target_key].values.tolist()[:10],  # 最初の10個のみ表示
                        "predictions": {
                        "true": y_test[target_key].values.tolist()[:10],  # 最初の10個のみ表示
                        "pred": y_pred.tolist()[:10]
                    }
                }}

                logger.info(f"分類モデル（{period}期先）評価 - 正解率: {accuracy:.4f}")

        return evaluation_results

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        評価結果の要約レポートを生成

        Args:
            evaluation_results: 評価結果のDict

        Returns:
            Dict: 評価レポート
        """
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

        return report

    def save_evaluation_report(self, report: Dict[str, Any]) -> bool:
        """
        評価レポートを保存

        Args:
            report: 保存する評価レポート

        Returns:
            bool: 保存が成功したかどうか
        """
        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # レポートをJSONファイルに保存
        report_path = output_dir / "model_evaluation_report.json"
        try:
            with open(report_path, "w") as f:
                import json
                json.dump(report, f, indent=2)
            logger.info(f"評価レポートを {report_path} に保存しました")
            return True
        except Exception as e:
            logger.error(f"レポート保存エラー: {e}")
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
        logger.error("モデルの読み込みに失敗しました")
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