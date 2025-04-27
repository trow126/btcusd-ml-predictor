# model_builder/config/default_config.py
from typing import Dict, Any

def get_default_binary_classifier_config() -> Dict[str, Any]:
    """デフォルトの二値分類モデル設定を返す"""
    return {
        "model_params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 3,
            "min_child_samples": 20,
            "max_depth": 8,
            "scale_pos_weight": 1.2,  # 上昇クラスにより重みをかける
            "verbose": -1
        },
        "fit_params": {
            "num_boost_round": 1000,
            "early_stopping_rounds": 50,
            "verbose_eval": 100
        }
    }

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
            "num_class": 3,       # 上昇/横ばい/下落の3クラス
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.03,  # 学習率を小さくして学習を安定させる
            "feature_fraction": 0.8,  # 特徴量サンプリングを強化
            "bagging_fraction": 0.7,  # バギングを強化
            "bagging_freq": 3,       # より高い频度でバギング
            "min_child_samples": 20,  # 各リーフの最小サンプル数を増やして過学習を抑制
            "max_depth": 10,        # 最大深さを制限
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
        "classification_threshold": 0.0005,  # 分類閾値（±0.05%）
        "use_dynamic_threshold": True,   # 動的閾値を使用するかどうか
        "use_binary_classification": True  # 2クラス分類を使用するかどうか
    }