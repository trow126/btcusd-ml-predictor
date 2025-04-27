# model_builder/evaluators/base_evaluator.py
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from ..utils.logging_utils import setup_logger

class BaseEvaluator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        ベース評価器クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        self.logger = setup_logger("base_evaluator")
        self.config = self._get_default_config()

        # 渡された設定でデフォルト設定を上書き（マージ）
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                    # 辞書の場合は再帰的にマージ
                    self.config[key].update(value)
                else:
                    self.config[key] = value

        self.models = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す（サブクラスでオーバーライド）"""
        return {}

    def load_model(self, model_path: str) -> Any:
        """
        モデルを読み込む

        Args:
            model_path: モデルファイルのパス

        Returns:
            Any: 読み込んだモデル
        """
        try:
            model = joblib.load(model_path)
            self.logger.info(f"モデルを {model_path} から読み込みました")
            return model
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")
            return None