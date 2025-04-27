# model_builder/evaluators/regression_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional

from .base_evaluator import BaseEvaluator
from ..config.default_config import get_default_evaluator_config

class RegressionEvaluator(BaseEvaluator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        回帰モデル評価器クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return get_default_evaluator_config()
    
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, period: int) -> Dict[str, Any]:
        """
        回帰モデルを評価

        Args:
            model: 評価するモデル
            X_test: テスト特徴量DataFrame
            y_test: テスト目標変数Series
            period: 予測期間

        Returns:
            Dict: 評価結果
        """
        self.logger.info(f"evaluate: {period}期先の回帰モデルの評価を開始します")
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価
        mae = mean_absolute_error(y_test, y_pred)
        
        # 結果を保存
        result = {
            "mae": mae,
            "predictions": {
                "true": y_test.values.tolist()[:10],  # 最初の10個のみ表示
                "pred": y_pred.tolist()[:10]
            }
        }
        
        self.logger.info(f"evaluate: {period}期先の回帰モデル評価 - MAE: {mae:.6f}")
        self.logger.info(f"evaluate: {period}期先の回帰モデルの評価を終了します")
        
        return result