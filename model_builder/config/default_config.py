# model_builder/config/default_config.py
from typing import Dict, Any

def get_default_trainer_config() -> Dict[str, Any]:
    """デフォルトのトレーナー設定を返す"""
    return {
        "input_dir": "data/processed",
        "input_filename": "btcusd_5m_features.csv",
        "output_dir": "models",
        "feature_groups": {
            "price": True,            # 価格関連特徴量
            "volume": True,           # 出来高関連特徴量
            "technical": True,        # テクニカル指標関連特徴量
        },
        "target_periods": [1, 2, 3],  # 予測対象期間 (1=5分後, 2=10分後, 3=15分後)
        "cv_splits": 5,               # 時系列交差検証の分割数
        "test_size": 0.2,             # テストデータの割合
    }

def get_default_regressor_config() -> Dict[str, Any]:
    """デフォルトの回帰モデル設定を返す"""
    return {
        "model_params": {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        },
        "fit_params": {
            "num_boost_round": 1000,
            "early_stopping_rounds": 50,
            "verbose_eval": 100
        }
    }

def get_default_classifier_config() -> Dict[str, Any]:
    """デフォルトの分類モデル設定を返す"""
    return {
        "model_params": {
            "objective": "multiclass",
            "num_class": 5,       # 上昇/横ばい/下落の3クラス
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        },
        "fit_params": {
            "num_boost_round": 1000,
            "early_stopping_rounds": 50,
            "verbose_eval": 100
        }
    }

def get_default_evaluator_config() -> Dict[str, Any]:
    """デフォルトの評価器設定を返す"""
    return {
        "input_dir": "data/processed",
        "input_filename": "btcusd_5m_features.csv",
        "model_dir": "models",
        "output_dir": "evaluation",
        "target_periods": [1, 2, 3],  # 予測対象期間 (1=5分後, 2=10分後, 3=15分後)
        "test_size": 0.2,             # テストデータの割合
        "classification_threshold": 0.0005  # 分類閾値（±0.05%）
    }