# model_builder/trainers/binary_classification_trainer.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from typing import Dict, List, Tuple, Any, Optional, Union

from .base_trainer import BaseTrainer
from ..config.default_config import get_default_binary_classifier_config
from ..utils.feature_utils import get_feature_importance

class BinaryClassificationTrainer(BaseTrainer):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        二値分類モデルトレーナークラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """二値分類モデルのデフォルト設定を返す"""
        return get_default_binary_classifier_config()
    
    def train(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series],
        period: int
    ) -> Dict[str, Any]:
        """
        二値分類モデル（上昇/下落予測）をトレーニング

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict
            period: 予測期間

        Returns:
            Dict: トレーニング結果
        """
        target_name = f"binary_classification_{period}"
        
        self.logger.info(f"train: {period}期先の価格上昇/下落予測モデルをトレーニングします")

        # クラス重み付けの計算
        class_counts = y_train[target_name].value_counts()
        total_samples = len(y_train[target_name])
        n_classes = len(class_counts)

        # 各クラスの重みを計算（サンプル数の少ないクラスほど大きな重みを持つ）
        class_weights = {
            class_idx: total_samples / (n_classes * count)
            for class_idx, count in class_counts.items()
        }

        self.logger.info(f"クラス重み: {class_weights}")

        # LightGBMデータセットの作成
        lgb_train = lgb.Dataset(
            X_train["X"],
            y_train[target_name],
            feature_name=list(X_train["X"].columns),
            free_raw_data=False,
            weight=np.array([class_weights[label] for label in y_train[target_name]])  # サンプル重みの設定
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
        y_pred_proba = model.predict(X_test["X"])
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 評価
        accuracy = accuracy_score(y_test[target_name], y_pred)
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_test[target_name], y_pred_proba)
        except:
            auc = 0.0

        # 混同行列
        cm = confusion_matrix(y_test[target_name], y_pred)

        # 分類レポート
        report = classification_report(y_test[target_name], y_pred, output_dict=True)

        # 特徴量重要度を計算
        feature_importance = get_feature_importance(
            model, X_train["X"].columns, top_n=20
        )

        # 結果を保存
        result = {
            "model": model,
            "accuracy": accuracy,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "feature_importance": feature_importance,
            "class_weights": class_weights
        }

        self.logger.info(f"train: {period}期先の価格上昇/下落予測モデル - 正解率: {accuracy:.4f}, ROC AUC: {auc:.4f}")
        self.logger.info(f"train: {period}期先の二値分類モデルのトレーニングを終了します")

        # モデルの保存
        self._save_model(model, f"binary_classification_model_period_{period}")

        return result