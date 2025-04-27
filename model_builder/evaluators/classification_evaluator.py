# model_builder/evaluators/classification_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional

from .base_evaluator import BaseEvaluator
from ..config.default_config import get_default_evaluator_config

class ClassificationEvaluator(BaseEvaluator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        分類モデル評価器クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return get_default_evaluator_config()
    
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, period: int) -> Dict[str, Any]:
        """
        分類モデルを評価

        Args:
            model: 評価するモデル
            X_test: テスト特徴量DataFrame
            y_test: テスト目標変数Series
            period: 予測期間

        Returns:
            Dict: 評価結果
        """
        self.logger.info(f"evaluate: {period}期先の分類モデルの評価を開始します")
        
        # モデルで使用されている特徴量の名前を取得
        model_features = model.feature_name()
        
        # テストデータにモデルの特徴量がない場合に対処
        missing_features = [f for f in model_features if f not in X_test.columns]
        if missing_features:
            self.logger.warning(f"モデルの特徴量がテストデータに見つかりません: {missing_features}")
            return {"error": "feature_mismatch", "accuracy": 0.0}
        
        # モデルで使用されている特徴量のみを選択
        X_test_selected = X_test[model_features]
        
        # 予測
        y_pred_proba = model.predict(X_test_selected)
        y_pred = np.argmax(y_pred_proba, axis=1) - 1  # 0, 1, 2を-1, 0, 1に変換
        
        # 評価
        accuracy = accuracy_score(y_test, y_pred)
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        
        # 分類レポート
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 結果を保存
        result = {
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": {
                "true": y_test.values.tolist()[:10],  # 最初の10個のみ表示
                "pred": y_pred.tolist()[:10]
            }
        }
        
        self.logger.info(f"evaluate: {period}期先の分類モデル評価 - 正解率: {accuracy:.4f}")
        self.logger.info(f"evaluate: {period}期先の分類モデルの評価を終了します")
        
        return result