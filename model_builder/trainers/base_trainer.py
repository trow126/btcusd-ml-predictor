# model_builder/trainers/base_trainer.py
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from ..utils.logging_utils import setup_logger

class BaseTrainer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        ベーストレーナークラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        self.logger = setup_logger("base_trainer")
        self.config = self._get_default_config()

        # 渡された設定でデフォルト設定を上書き（マージ）
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                    # 辞書の場合は再帰的にマージ
                    self.config[key].update(value)
                else:
                    self.config[key] = value

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す（サブクラスでオーバーライド）"""
        return {}

    def _save_model(self, model: Any, name: str) -> bool:
        """
        モデルを保存

        Args:
            model: 保存するモデル
            name: モデル名

        Returns:
            bool: 保存が成功したかどうか
        """
        self.logger.info(f"_save_model: モデル '{name}' の保存を開始します")
        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(self.config.get("output_dir", "models"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # モデルの保存
        model_path = output_dir / f"{name}.joblib"
        try:
            joblib.dump(model, model_path)
            self.logger.info(f"_save_model: モデルを {model_path} に保存しました")
            self.logger.info(f"_save_model: モデル '{name}' の保存を終了します")
            return True
        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")
            return False