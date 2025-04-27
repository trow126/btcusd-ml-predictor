# model_builder/trainers/regression_trainer.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional, Union

from .base_trainer import BaseTrainer
from ..config.default_config import get_default_regressor_config
from ..utils.feature_utils import get_feature_importance

class RegressionTrainer(BaseTrainer):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        回帰モデルトレーナークラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """回帰モデルのデフォルト設定を返す"""
        return get_default_regressor_config()
    
    def train(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series],
        period: int
    ) -> Dict[str, Any]:
        """
        回帰モデル（価格変動率予測）をトレーニング

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict
            period: 予測期間

        Returns:
            Dict: トレーニング結果
        """
        target_name = f"regression_{period}"
        
        self.logger.info(f"train: {period}期先の価格変動率予測モデルをトレーニングします")

        # LightGBMデータセットの作成
        lgb_train = lgb.Dataset(
            X_train["X"],
            y_train[target_name],
            feature_name=list(X_train["X"].columns),
            free_raw_data=False
        )

        lgb_valid = lgb.Dataset(
            X_test["X"],
            y_test[target_name],
            reference=lgb_train,
            feature_name=list(X_test["X"].columns),
            free_raw_data=False
        )

        # モデルパラメータ
        model_params = self.config["model_params"].copy()

        # ブースティングラウンド数と早期停止設定
        num_boost_round = self.config["fit_params"].get("num_boost_round", 1000)
        callbacks = []

        # 早期停止の設定
        early_stopping_rounds = self.config["fit_params"].get("early_stopping_rounds", 50)
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))

        # 進捗表示の設定
        verbose_eval = self.config["fit_params"].get("verbose_eval", 100)
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(period=verbose_eval, show_stdv=True))

        # モデルのトレーニング
        model = lgb.train(
            params=model_params,
            train_set=lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        # 予測
        y_pred = model.predict(X_test["X"])

        # 評価
        mae = mean_absolute_error(y_test[target_name], y_pred)

        # 特徴量重要度を計算
        feature_importance = get_feature_importance(
            model, X_train["X"].columns, top_n=20
        )

        # 結果を保存
        result = {
            "model": model,
            "mae": mae,
            "feature_importance": feature_importance
        }

        self.logger.info(f"train: {period}期先の価格変動率予測モデル - MAE: {mae:.6f}")
        self.logger.info(f"train: {period}期先の回帰モデルのトレーニングを終了します")

        # モデルの保存
        self._save_model(model, f"regression_model_period_{period}")

        return result