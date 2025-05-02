# model_builder/trainers/high_threshold_signal_trainer.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
from pathlib import Path

from .base_trainer import BaseTrainer
from ..config.default_config import get_default_binary_classifier_config
from ..utils.feature_utils import get_feature_importance

class HighThresholdSignalTrainer(BaseTrainer):
    """
    高閾値シグナルモデルトレーナークラス
    特定の方向（ロングまたはショート）に特化したモデルを構築
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)

    def _get_default_config(self) -> Dict[str, Any]:
        """二値分類モデルのデフォルト設定を返す"""
        # 基本的な二値分類の設定を使用し、一部パラメータを調整
        default_config = get_default_binary_classifier_config()
        
        # 高閾値シグナルモデル用の調整
        default_config["model_params"]["objective"] = "binary"
        default_config["model_params"]["metric"] = "binary_logloss"
        default_config["model_params"]["boosting_type"] = "gbdt"
        default_config["model_params"]["num_leaves"] = 31
        default_config["model_params"]["learning_rate"] = 0.03  # 0.05から0.03に下げて過学習を抑制
        default_config["model_params"]["feature_fraction"] = 0.8  # 0.9から0.8に下げて汎化性能を向上
        default_config["model_params"]["bagging_fraction"] = 0.7  # 0.8から0.7に調整
        default_config["model_params"]["bagging_freq"] = 3      # 5から3に変更
        default_config["model_params"]["min_child_samples"] = 20  # 最小サンプル数を設定
        default_config["model_params"]["verbose"] = -1
        
        return default_config

    def train(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series],
        period: int, direction: str = "long", threshold: float = 0.002
    ) -> Dict[str, Any]:
        """
        高閾値シグナルモデル（ロングまたはショート）をトレーニング
        
        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict
            period: 予測期間
            direction: "long" または "short"
            threshold: シグナル閾値（0.002 = 0.2%）
            
        Returns:
            Dict: トレーニング結果
        """
        # 閾値を文字列に変換（例: 0.002 -> "2"）
        threshold_str = str(int(threshold * 1000))
        
        # 目標変数名の構築 - target_high_threshold_プレフィックスを使用
        target_name = f"target_high_threshold_{threshold_str}p_{direction}_{period}"
        
        self.logger.info(f"==== train: {period}期先の高閾値({threshold*100:.1f}%)シグナルモデル（{direction}）のトレーニングを開始 ====")

        try:
            # 1. 特徴量と目標変数の準備
            X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered = self._prepare_data(
                X_train, X_test, y_train, y_test, target_name, period
            )

            # データ検証
            if X_train_filtered.empty or X_test_filtered.empty:
                self.logger.error(f"train: 高閾値シグナルモデル（{direction}）の{period}期先モデル - 有効なデータが不足しています")
                return {
                    "error": "insufficient_data",
                    "message": "有効なデータが不足しています",
                    "train_size": len(X_train_filtered),
                    "test_size": len(X_test_filtered)
                }

            # 2. クラス重み付けパラメータの最適化
            self._optimize_class_weights(y_train_filtered)

            # 3. モデルトレーニング
            model, train_metrics = self._train_model(X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)

            # 4. 評価と結果取得
            result = self._evaluate_model(model, X_test_filtered, y_test_filtered, X_train_filtered)

            # 5. モデルの保存
            model_filename = f"target_high_threshold_{threshold_str}p_{direction}_{period}"
            self._save_trained_model(model, model_filename)

            return result
        except Exception as e:
            self.logger.error(f"train: モデルトレーニング中にエラーが発生しました: {str(e)}")
            import traceback
            self.logger.error(f"トレースバック: {traceback.format_exc()}")
            return {
                "error": "training_error",
                "message": str(e),
                "train_size": len(X_train_filtered) if 'X_train_filtered' in locals() else 0,
                "test_size": len(X_test_filtered) if 'X_test_filtered' in locals() else 0
            }

    def _prepare_data(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series],
        target_name: str, period: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        トレーニングとテストデータを準備

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict
            target_name: 目標変数名
            period: 予測期間

        Returns:
            Tuple: (X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)
        """
        self.logger.info(f"_prepare_data: 目標変数 {target_name} の準備を開始")
        
        # まずtarget_nameがあるか確認
        if target_name not in y_train:
            self.logger.error(f"目標変数 {target_name} が見つかりません。利用可能な目標変数: {list(y_train.keys())}")
            raise ValueError(f"目標変数 {target_name} が見つかりません")
            
        if target_name not in y_test:
            self.logger.error(f"テストデータの目標変数 {target_name} が見つかりません")
            raise ValueError(f"テストデータの目標変数 {target_name} が見つかりません")
        
        # 基本の特徴量セットを使用
        X_train_filtered = X_train["X"]
        y_train_filtered = y_train[target_name]
        X_test_filtered = X_test["X"]
        y_test_filtered = y_test[target_name]
        
        # NaN値が含まれていないか確認（高閾値シグナル変数にはNaNは存在しないはず）
        if y_train_filtered.isna().any():
            self.logger.warning(f"トレーニングデータにNaN値が含まれています。これらを除外します。")
            mask = ~y_train_filtered.isna()
            X_train_filtered = X_train_filtered.loc[mask]
            y_train_filtered = y_train_filtered[mask]
            self.logger.info(f"NaN除外後のトレーニングデータ: {len(y_train_filtered)}行")

        if y_test_filtered.isna().any():
            self.logger.warning(f"テストデータにNaN値が含まれています。これらを除外します。")
            mask = ~y_test_filtered.isna()
            X_test_filtered = X_test_filtered.loc[mask]
            y_test_filtered = y_test_filtered[mask]
            self.logger.info(f"NaN除外後のテストデータ: {len(y_test_filtered)}行")

        # クラスバランスを確認
        train_class_balance = y_train_filtered.value_counts()
        self.logger.info(f"トレーニングデータのクラスバランス: {train_class_balance.to_dict()}")
        
        # シグナル発生率を計算
        signal_rate = (y_train_filtered == 1).sum() / len(y_train_filtered) * 100
        self.logger.info(f"シグナル発生率: {signal_rate:.2f}%")
        
        # データが十分にあるか確認
        if len(y_train_filtered) < 100:
            self.logger.error(f"トレーニングデータが少なすぎます: {len(y_train_filtered)}行")
            raise ValueError(f"トレーニングデータが少なすぎます: {len(y_train_filtered)}行")

        if len(y_test_filtered) < 50:
            self.logger.error(f"テストデータが少なすぎます: {len(y_test_filtered)}行")
            raise ValueError(f"テストデータが少なすぎます: {len(y_test_filtered)}行")
            
        # シグナルクラス（1）が極端に少ない場合は警告
        if (y_train_filtered == 1).sum() < 50:
            self.logger.warning(f"シグナルクラスのサンプル数が少なすぎます: {(y_train_filtered == 1).sum()}サンプル")
            
        self.logger.info(f"_prepare_data: 特徴量と目標変数の準備完了。トレーニングデータ: {len(X_train_filtered)}行, テストデータ: {len(X_test_filtered)}行")
        return X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered

    def _optimize_class_weights(self, y_train_filtered: pd.Series) -> None:
        """
        クラス重み付けパラメータを最適化

        Args:
            y_train_filtered: フィルタリングされたトレーニング目標変数
        """
        # クラス重み付けの計算とクラスバランスの確認
        class_counts = y_train_filtered.value_counts()
        
        # クラスが2つ以上ある場合のみ処理
        if len(class_counts) < 2:
            self.logger.warning(f"クラスが1つしかありません: {class_counts.to_dict()}")
            return
            
        total_samples = len(y_train_filtered)
        n_classes = len(class_counts)

        # シグナルクラス（1）と非シグナルクラス（0）の比率
        if 1 in class_counts and 0 in class_counts:
            signal_count = class_counts.get(1, 0)
            non_signal_count = class_counts.get(0, 0)
            
            # シグナルが極端に少ない場合（通常のケース）
            if signal_count < non_signal_count:
                # 不均衡度合いの計算
                imbalance_ratio = non_signal_count / signal_count
                self.logger.info(f"クラス不均衡比率: {imbalance_ratio:.4f} (シグナル:非シグナル = 1:{imbalance_ratio:.1f})")
                
                # 極端な不均衡の場合は上限を設定
                if imbalance_ratio > 100:
                    imbalance_ratio = min(imbalance_ratio, 100)
                    self.logger.info(f"クラス重み比率を {imbalance_ratio:.1f} に制限します")
                
                # モデルパラメータにクラス重みを反映
                self.config["model_params"]["scale_pos_weight"] = imbalance_ratio
                self.logger.info(f"シグナルクラスの重みを増加: scale_pos_weight = {imbalance_ratio:.4f}")
            else:
                # 通常ありえないケース（シグナルの方が多い）
                self.logger.warning(f"シグナルクラスのサンプル数が非シグナルより多いケースです")
        else:
            self.logger.warning(f"クラスの値が不適切です: {class_counts.to_dict()}")

    def _train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Any, Dict[str, float]]:
        """
        LightGBMモデルをトレーニング

        Args:
            X_train: トレーニング特徴量
            y_train: トレーニング目標変数
            X_test: テスト特徴量
            y_test: テスト目標変数

        Returns:
            Tuple[Any, Dict]: (トレーニングされたモデル, 訓練指標)
        """
        # LightGBMデータセットの作成
        lgb_train = lgb.Dataset(
            X_train,
            y_train,
            feature_name=list(X_train.columns),
            free_raw_data=False
        )

        lgb_valid = lgb.Dataset(
            X_test,
            y_test,
            reference=lgb_train,
            feature_name=list(X_test.columns),
            free_raw_data=False
        )

        # モデルパラメータ
        model_params = self.config["model_params"].copy()
        self.logger.info(f"モデルパラメータ: {model_params}")

        # ブースティングラウンド数と早期停止設定
        num_boost_round = self.config.get("fit_params", {}).get("num_boost_round", 1000)
        callbacks = []

        # 早期停止の設定
        early_stopping_rounds = self.config.get("fit_params", {}).get("early_stopping_rounds", 50)
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        # 進捗表示の設定
        verbose_eval = self.config.get("fit_params", {}).get("verbose_eval", 100)
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(period=verbose_eval, show_stdv=False))

        # モデルのトレーニング
        self.logger.info(f"モデルトレーニングを開始 (最大ラウンド: {num_boost_round})")
        model = lgb.train(
            params=model_params,
            train_set=lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        self.logger.info(f"モデルトレーニング完了 (実際のラウンド: {model.current_iteration()})")

        # トレーニング指標
        train_metrics = {
            "iterations": model.current_iteration(),
            "best_iteration": model.best_iteration if hasattr(model, 'best_iteration') else model.current_iteration()
        }

        return model, train_metrics

    def _evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        トレーニングされたモデルを評価

        Args:
            model: トレーニングされたモデル
            X_test: テスト特徴量
            y_test: テスト目標変数
            X_train: トレーニング特徴量（特徴量重要度算出用）

        Returns:
            Dict: 評価結果
        """
        # 予測確率
        y_pred_proba = model.predict(X_test)
        
        # 確信度閾値ごとの評価
        confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 低い閾値も評価対象に追加
        confidence_metrics = {}
        
        for conf_threshold in confidence_thresholds:
            # 閾値を適用
            y_pred = (y_pred_proba >= conf_threshold).astype(int)
            
            # シグナル発生率
            signal_rate = y_pred.mean()
            signal_count = y_pred.sum()
            
            # シグナルが1つもない場合はスキップ
            if signal_count == 0:
                confidence_metrics[conf_threshold] = {
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "signal_rate": 0.0,
                    "signal_count": 0
                }
                continue
                
            # 精度指標
            try:
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # 混同行列
                conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
                
                confidence_metrics[conf_threshold] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "confusion_matrix": conf_matrix.tolist(),
                    "signal_rate": float(signal_rate),
                    "signal_count": int(signal_count)
                }
            except Exception as e:
                self.logger.error(f"確信度閾値 {conf_threshold} での評価中にエラー: {str(e)}")
                confidence_metrics[conf_threshold] = {
                    "error": str(e),
                    "signal_rate": float(signal_rate),
                    "signal_count": int(signal_count)
                }
        
        # デフォルト閾値（0.5）での評価
        default_precision = confidence_metrics.get(0.5, {}).get("precision", np.nan)
        default_recall = confidence_metrics.get(0.5, {}).get("recall", np.nan)
        default_f1 = confidence_metrics.get(0.5, {}).get("f1_score", np.nan)
        
        # 特徴量重要度を計算
        feature_importance = get_feature_importance(
            model, X_train.columns, top_n=20
        )

        # 結果をまとめる
        result = {
            "model": model,
            "accuracy": float((y_test == (y_pred_proba >= 0.5).astype(int)).mean()),
            "precision": float(default_precision),
            "recall": float(default_recall),
            "f1_score": float(default_f1),
            "feature_importance": feature_importance,
            "confidence_metrics": confidence_metrics,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "signal_ratio": float((y_test == 1).mean())
        }

        # ログ出力
        self.logger.info(f"評価指標 - 精度: {result['accuracy']:.4f}, 適合率: {default_precision:.4f}, 再現率: {default_recall:.4f}, F1: {default_f1:.4f}")
        self.logger.info(f"シグナル比率: {result['signal_ratio']:.4f}")
        
        # 確信度閾値ごとの結果
        self.logger.info("確信度閾値ごとの評価結果:")
        for threshold, metrics in confidence_metrics.items():
            if "precision" in metrics:
                self.logger.info(f"  閾値 {threshold:.1f}: 適合率={metrics['precision']:.4f}, 発生率={metrics['signal_rate']:.4f}, シグナル数={metrics['signal_count']}")
        
        return result

    def _save_trained_model(self, model: Any, model_filename: str) -> bool:
        """
        トレーニングされたモデルを保存

        Args:
            model: 保存するモデル
            model_filename: モデルのファイル名

        Returns:
            bool: 保存が成功したかどうか
        """
        # 直接joblibを使って保存する
        try:
            # モデルの保存先を指定
            output_dir = self.config.get("output_dir", "models")
            output_subdir = "high_threshold"  # 高閾値モデル用のサブディレクトリ
            
            # サブディレクトリを含むパスを作成
            model_dir = Path(output_dir) / output_subdir
            os.makedirs(model_dir, exist_ok=True)
            
            # パスを結合してフルパスのファイル名を生成
            if not model_filename.endswith('.joblib'):
                model_filename = f"{model_filename}.joblib"
                
            # モデルの完全パスを生成
            model_path = model_dir / model_filename
            
            # モデル保存
            self.logger.info(f"モデルを保存します: {model_path}")
            joblib.dump(model, model_path)
            
            # 確認
            if os.path.exists(model_path):
                self.logger.info(f"モデルを {model_path} に保存しました")
                return True
            else:
                self.logger.error(f"モデルの保存失敗: ファイルが見つかりません: {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"モデルの保存中にエラーが発生しました: {str(e)}")
            import traceback
            self.logger.error(f"トレースバック: {traceback.format_exc()}")
            return False