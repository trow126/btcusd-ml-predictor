# model_builder/trainers/threshold_binary_classification_trainer.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
from pathlib import Path

from .base_trainer import BaseTrainer
from ..config.default_config import get_default_binary_classifier_config
from ..utils.feature_utils import get_feature_importance

class ThresholdBinaryClassificationTrainer(BaseTrainer):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        閾値ベースの二値分類モデルトレーナークラス
        横ばいを除外し、有意な上昇と下落のみを分類

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
        閾値ベースの二値分類モデル（有意な上昇/下落予測）をトレーニング
        横ばい（NaN）は除外して学習する

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict
            period: 予測期間

        Returns:
            Dict: トレーニング結果
        """
        target_name = f"threshold_binary_classification_{period}"
        
        self.logger.info(f"==== train: {period}期先の閾値ベース二値分類モデル（上昇/下落予測）をトレーニングします ====")
        
        # NaNを除外した特徴量と目標変数を確認
        X_key = f"X_threshold_binary_{period}"
        
        if X_key in X_train:
            # 専用に準備された特徴量セットがある場合はそれを使用
            X_train_filtered = X_train[X_key]
            y_train_filtered = y_train[target_name]
            self.logger.info(f"train: 専用特徴量セット {X_key} を使用 ({len(X_train_filtered)} 行)")
        else:
            # 後方互換性のため、旧方式でのフィルタリングも対応
            self.logger.info(f"専用特徴量セット {X_key} が見つかりません。旧方式でフィルタリングします。")
            # まずtarget_nameがあるか確認
            if target_name not in y_train:
                self.logger.error(f"目標変数 {target_name} が見つかりません。利用可能な目標変数: {list(y_train.keys())}")
                return {
                    "error": "missing_target",
                    "message": f"目標変数 {target_name} が見つかりません"
                }
                
            mask_train = ~y_train[target_name].isna()
            X_train_filtered = X_train["X"].loc[mask_train]
            y_train_filtered = y_train[target_name].dropna()
            self.logger.info(f"train: 従来方式で特徴量をフィルタリング ({len(X_train_filtered)} 行)")
        
        # テストデータも同様に処理
        if X_key in X_test:
            X_test_filtered = X_test[X_key]
            y_test_filtered = y_test[target_name]
            self.logger.info(f"train: テスト用専用特徴量セット {X_key} を使用 ({len(X_test_filtered)} 行)")
        else:
            if target_name not in y_test:
                self.logger.error(f"テストデータの目標変数 {target_name} が見つかりません")
                return {
                    "error": "missing_test_target",
                    "message": f"テストデータの目標変数 {target_name} が見つかりません"
                }
                
            mask_test = ~y_test[target_name].isna()
            X_test_filtered = X_test["X"].loc[mask_test]
            y_test_filtered = y_test[target_name].dropna()
            self.logger.info(f"train: 従来方式でテスト特徴量をフィルタリング ({len(X_test_filtered)} 行)")
        
        # データの検証
        if X_train_filtered.empty or X_test_filtered.empty:
            self.logger.error(f"train: 閾値ベース二値分類の{period}期先モデル - 有効なデータが不足しています")
            return {
                "error": "insufficient_data",
                "message": "有効なデータが不足しています",
                "train_size": len(X_train_filtered),
                "test_size": len(X_test_filtered)
            }
            
        # NaN値が含まれていないか確認
        if y_train_filtered.isna().any():
            self.logger.error(f"train: 閾値ベース二値分類の{period}期先モデル - 目標変数にNaN値が含まれています")
            # NaN値を除外
            mask = ~y_train_filtered.isna()
            X_train_filtered = X_train_filtered.loc[mask]
            y_train_filtered = y_train_filtered[mask]
            self.logger.info(f"NaN値を除外しました。残りのデータ数: {len(y_train_filtered)}")
            
            # 除外後に十分なデータがあるか確認
            if len(y_train_filtered) < 100:
                self.logger.error(f"train: NaN除外後のデータが少なすぎます: {len(y_train_filtered)}行")
                return {
                    "error": "insufficient_data_after_nan_removal",
                    "message": "NaN除外後の有効データが少なすぎます",
                    "remaining_train_size": len(y_train_filtered)
                }
                
        # テストデータにも同様の処理を適用
        if y_test_filtered.isna().any():
            self.logger.info(f"train: テストデータのNaN値を除外します")
            mask = ~y_test_filtered.isna()
            X_test_filtered = X_test_filtered.loc[mask]
            y_test_filtered = y_test_filtered[mask]
            self.logger.info(f"テストデータのNaN除外後: {len(y_test_filtered)}行")
        
        # クラスバランスを確認
        train_class_balance = y_train_filtered.value_counts()
        self.logger.info(f"train: トレーニングデータのクラスバランス: {train_class_balance.to_dict()}")
        
        # カラム名の表示
        self.logger.info(f"特徴量列: {X_train_filtered.columns.tolist()[:5]}... (全{len(X_train_filtered.columns)}列)")
        
        # クラス重み付けの計算とクラスバランスの確認
        class_counts = y_train_filtered.value_counts()
        total_samples = len(y_train_filtered)
        n_classes = len(class_counts)
        
        # 各クラスの重みを計算（サンプル数の少ないクラスほど大きな重みを持つ）
        if n_classes > 1:
            # クラス対称判定
            class_ratio = class_counts.min() / class_counts.max()
            self.logger.info(f"クラス比率: {class_ratio:.4f} (1に近いほど均等)")
            
            # 重み付け計算
            class_weights = {}
            for class_idx, count in class_counts.items():
                class_weights[class_idx] = total_samples / (n_classes * count)
                
            self.logger.info(f"クラス重み: {class_weights}")
            
            # モデルパラメータにクラス重みのソフトバージョンを反映
            # 白黒の重み付けではなく、モデルパラメータで控える
            if 0 in class_weights and 1 in class_weights:
                # 0が少ない場合は大きい重み
                if class_weights[0] > class_weights[1]:
                    # 0クラス（下落）に重みを置くケース
                    scale_pos_weight = 1.0 / class_ratio  # 上昇(1)クラスに対する下落(0)クラスのスケール
                else:
                    # 1クラス（上昇）に重みを置くケース
                    scale_pos_weight = class_ratio  # 下落(0)クラスに対する上昇(1)クラスのスケール
                    
                self.logger.info(f"scale_pos_weight パラメータを {scale_pos_weight:.4f} に設定します")
                
                # モデルパラメータに追加
                self.config["model_params"]["scale_pos_weight"] = scale_pos_weight
        
        # LightGBMデータセットの作成
        try:
            # サンプル重みを使用しないバージョン
            lgb_train = lgb.Dataset(
                X_train_filtered,
                y_train_filtered,
                feature_name=list(X_train_filtered.columns),
                free_raw_data=False
            )

            lgb_valid = lgb.Dataset(
                X_test_filtered,
                y_test_filtered,
                reference=lgb_train,
                feature_name=list(X_test_filtered.columns),
                free_raw_data=False
            )

            # モデルパラメータ
            model_params = self.config["model_params"].copy()
            self.logger.info(f"モデルパラメータ: {model_params}")

            # ブースティングラウンド数と早期停止設定
            num_boost_round = self.config["fit_params"].get("num_boost_round", 1000)
            callbacks = []

            # 早期停止の設定
            early_stopping_rounds = self.config["fit_params"].get("early_stopping_rounds", 50)
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

            # 進捗表示の設定
            verbose_eval = self.config["fit_params"].get("verbose_eval", 100)
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

            # 予測
            y_pred_proba = model.predict(X_test_filtered)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # 評価指標の計算
            accuracy = accuracy_score(y_test_filtered, y_pred)
            precision = precision_score(y_test_filtered, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test_filtered, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test_filtered, y_pred, average='binary', zero_division=0)
            conf_matrix = confusion_matrix(y_test_filtered, y_pred).tolist()

            # 特徴量重要度を計算
            feature_importance = get_feature_importance(
                model, X_train_filtered.columns, top_n=20
            )

            # 結果を保存
            result = {
                "model": model,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix,
                "feature_importance": feature_importance,
                "train_samples": len(X_train_filtered),
                "test_samples": len(X_test_filtered),
                "class_balance": train_class_balance.to_dict()
            }

            self.logger.info(f"train: {period}期先の閾値ベース二値分類モデル - 精度: {accuracy:.4f}, F1: {f1:.4f}")
            self.logger.info(f"train: {period}期先の閾値ベース二値分類モデルのトレーニングを終了します")

            # モデルの保存
            model_filename = f"threshold_binary_classification_model_period_{period}"  # .joblib拡張子を削除
            output_dir = self.config.get("output_dir", "models")
            os.makedirs(output_dir, exist_ok=True)
            model_path = Path(output_dir) / f"{model_filename}.joblib"  # 表示用に拡張子を追加
            
            save_result = self._save_model(model, model_filename)
            if save_result:
                self.logger.info(f"モデルを {model_path} に保存しました")
            else:
                self.logger.error(f"モデルの保存に失敗しました: {model_path}")

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
