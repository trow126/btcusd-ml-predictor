# model_builder/predictors/price_predictor.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from .base_predictor import BasePredictor
from ..config.default_config import get_default_evaluator_config

class PricePredictor(BasePredictor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        価格予測器クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return get_default_evaluator_config()
    
    def load_models(self) -> bool:
        """
        保存されたモデルを読み込む

        Returns:
            bool: 読み込みが成功したかどうか
        """
        self.logger.info("load_models: モデルの読み込みを開始します")
        model_dir = Path(self.config["model_dir"])

        if not model_dir.exists():
            self.logger.error(f"load_models: モデルディレクトリ {model_dir} が存在しません")
            return False

        # 回帰モデルの読み込み
        for period in self.config["target_periods"]:
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
        for period in self.config["target_periods"]:
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
    
    def predict(self, latest_data: pd.DataFrame, period: int = 1) -> Dict[str, Any]:
        """
        次の価格変動を予測

        Args:
            latest_data: 最新のデータ
            period: 予測期間 (1=5分後, 2=10分後, 3=15分後)

        Returns:
            Dict: 予測結果
        """
        regression_model_key = f"regression_{period}"
        classification_model_key = f"classification_{period}"

        if regression_model_key not in self.models or classification_model_key not in self.models:
            self.logger.error(f"{period}期先のモデルが見つかりません")
            return {}

        regression_model = self.models[regression_model_key]
        classification_model = self.models[classification_model_key]

        # 必要な特徴量のみを抽出
        features = latest_data.iloc[-1:].copy()

        # 目標変数を除外
        feature_cols = [col for col in features.columns if not col.startswith("target_")]
        features = features[feature_cols]

        # 予測
        price_change = float(regression_model.predict(features)[0])

        # 分類予測（確率）
        direction_proba = classification_model.predict(features)[0]

        # 分類予測（クラス）
        direction = int(np.argmax(direction_proba) - 1)  # 0, 1, 2を-1, 0, 1に変換

        # 方向ラベル
        direction_label = {
            -1: "下落",
            0: "横ばい",
            1: "上昇"
        }[direction]

        # 現在価格
        current_price = float(latest_data["close"].iloc[-1])

        # 予測価格
        predicted_price = current_price * (1 + price_change)

        # 予測信頼度（確率）
        confidence = float(direction_proba[direction + 1])  # -1, 0, 1を0, 1, 2に変換

        return {
            "current_time": latest_data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": current_price,
            "period": period,
            "prediction_time": f"{period * 5}分後",
            "predicted_change": price_change,
            "predicted_price": predicted_price,
            "predicted_direction": direction,
            "predicted_direction_label": direction_label,
            "confidence": confidence
        }

# グローバル関数として提供
def predict_next_price_movement(
    models: Dict[str, Any],
    latest_data: pd.DataFrame,
    period: int = 1
) -> Dict[str, Any]:
    """
    次の価格変動を予測するヘルパー関数

    Args:
        models: 予測モデルのDict
        latest_data: 最新のデータ
        period: 予測期間 (1=5分後, 2=10分後, 3=15分後)

    Returns:
        Dict: 予測結果
    """
    regression_model_key = f"regression_{period}"
    classification_model_key = f"classification_{period}"

    if regression_model_key not in models or classification_model_key not in models:
        return {}

    regression_model = models[regression_model_key]
    classification_model = models[classification_model_key]

    # 必要な特徴量のみを抽出
    features = latest_data.iloc[-1:].copy()

    # 目標変数を除外
    feature_cols = [col for col in features.columns if not col.startswith("target_")]
    features = features[feature_cols]

    # 予測
    price_change = float(regression_model.predict(features)[0])

    # 分類予測（確率）
    direction_proba = classification_model.predict(features)[0]

    # 分類予測（クラス）
    direction = int(np.argmax(direction_proba) - 1)  # 0, 1, 2を-1, 0, 1に変換

    # 方向ラベル
    direction_label = {
        -1: "下落",
        0: "横ばい",
        1: "上昇"
    }[direction]

    # 現在価格
    current_price = float(latest_data["close"].iloc[-1])

    # 予測価格
    predicted_price = current_price * (1 + price_change)

    # 予測信頼度（確率）
    confidence = float(direction_proba[direction + 1])  # -1, 0, 1を0, 1, 2に変換

    return {
        "current_time": latest_data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": current_price,
        "period": period,
        "prediction_time": f"{period * 5}分後",
        "predicted_change": price_change,
        "predicted_price": predicted_price,
        "predicted_direction": direction,
        "predicted_direction_label": direction_label,
        "confidence": confidence
    }