# model_builder/evaluators/threshold_binary_classification_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional, Set
import logging

from .base_evaluator import BaseEvaluator

class ThresholdBinaryClassificationEvaluator(BaseEvaluator):
    """
    閾値ベースの二値分類モデル評価クラス
    横ばいを除外し、有意な上昇と下落のみを分類
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, period: int) -> Dict[str, Any]:
        """
        閾値ベースの二値分類モデル（有意な上昇/下落予測）を評価

        Args:
            model: 評価するモデル
            X_test: テスト特徴量DataFrame
            y_test: テスト目標変数のSeries
            period: 予測期間

        Returns:
            Dict: 評価結果
        """
        self.logger.info(f"evaluate: {period}期先の閾値ベース二値分類モデルを評価します")

        try:
            # データが空の場合はエラー
            if X_test.empty or len(y_test) == 0:
                self.logger.warning(f"evaluate: テストデータが空です。期間: {period}")
                return {
                    "error": "empty_data",
                    "message": "テストデータが空です"
                }

            # NaN値を持つサンプルをフィルタリング
            if y_test.isna().any():
                self.logger.info(f"NaN値を含むサンプルをフィルタリングします")
                mask = ~y_test.isna()
                X_test_filtered = X_test.loc[mask]
                y_test_filtered = y_test[mask]

                self.logger.info(f"フィルタリング後のサンプル数: {len(y_test_filtered)} (元: {len(y_test)})")

                if len(y_test_filtered) == 0:
                    self.logger.error("フィルタリング後にデータがありません")
                    return {
                        "error": "no_valid_data",
                        "message": "フィルタリング後に有効なデータがありません"
                    }

                X_test = X_test_filtered
                y_test = y_test_filtered

            # モデルの特徴量名を取得し、特徴量の整合性を確認
            self.logger.info(f"X_testの形状: {X_test.shape}, 特徴量数: {X_test.shape[1]}")

            # 特徴量の不一致問題を処理
            X_test_adjusted = self._adjust_features_for_model(model, X_test)
            if X_test_adjusted is None:
                return {
                    "error": "feature_mismatch",
                    "message": "テスト特徴量をモデル特徴量に合わせることができませんでした"
                }

            # 二値分類の予測確率
            try:
                y_pred_proba = model.predict(X_test_adjusted)
            except Exception as predict_error:
                # 形状チェックエラーの場合は形状チェックを無効化して再試行
                if "shape" in str(predict_error).lower() or "feature" in str(predict_error).lower():
                    self.logger.warning(f"形状チェックエラーが発生しました: {str(predict_error)}")
                    try:
                        # LightGBMの場合、形状チェックをスキップするオプションを指定
                        y_pred_proba = self._predict_with_shape_check_disabled(model, X_test_adjusted)
                    except Exception as retry_error:
                        self.logger.error(f"形状チェック無効での再試行も失敗: {str(retry_error)}")
                        return {
                            "error": "prediction_error",
                            "message": str(retry_error)
                        }
                else:
                    # 形状以外のエラーの場合はそのままエラーとして処理
                    self.logger.error(f"予測中にエラーが発生: {str(predict_error)}")
                    return {
                        "error": "prediction_error",
                        "message": str(predict_error)
                    }

            # 閾値で分類（デフォルトは0.5）
            y_pred = (y_pred_proba > 0.5).astype(int)

            # 評価指標の計算
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

            # 混同行列
            cm = confusion_matrix(y_test, y_pred)

            # 各クラスの割合を計算
            class_counts = pd.Series(y_test).value_counts().to_dict()

            # フィルタリング率（NaNの割合）
            filtered_ratio = 0.0
            if hasattr(self, 'orig_test_size') and self.orig_test_size > 0:
                filtered_ratio = 1.0 - (len(y_test) / self.orig_test_size)

            # 評価結果
            result = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "filtered_ratio": filtered_ratio,
                "data_size": class_counts
            }

            self.logger.info(f"evaluate: {period}期先の閾値ベース二値分類モデル - 正解率: {accuracy:.4f}, F1: {f1:.4f}")
            self.logger.info(f"evaluate: クラス分布: {class_counts}")

            return result
        except Exception as e:
            self.logger.error(f"evaluate: 評価中にエラーが発生しました: {str(e)}")
            import traceback
            self.logger.error(f"トレースバック: {traceback.format_exc()}")
            return {
                "error": "evaluation_error",
                "message": str(e)
            }

    def _adjust_features_for_model(self, model: Any, X_test: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        モデルが期待する特徴量に合わせてテスト特徴量を調整

        Args:
            model: 評価するモデル
            X_test: テスト特徴量DataFrame

        Returns:
            Optional[pd.DataFrame]: 調整された特徴量、または調整できない場合はNone
        """
        # モデルの特徴量名を取得
        model_features = self._get_model_feature_names(model)
        if not model_features:
            self.logger.warning("モデルから特徴量名を取得できません。元の特徴量を使用します。")
            return X_test

        self.logger.info(f"モデルが期待する特徴量数: {len(model_features)}")

        # モデル特徴量がすべてテスト特徴量に含まれているか確認
        if set(model_features).issubset(set(X_test.columns)):
            self.logger.info(f"テスト特徴量を調整します: {X_test.shape} → ({X_test.shape[0]}, {len(model_features)})")
            return X_test[model_features]
        else:
            # 不足している特徴量を特定
            missing_features = set(model_features) - set(X_test.columns)
            self.logger.error(f"テストデータに不足している特徴量があります: {missing_features}")

            # 共通の特徴量のみを使用する試み
            common_features = list(set(model_features) & set(X_test.columns))
            if len(common_features) > 0:
                self.logger.warning(f"共通特徴量のみを使用します: {len(common_features)} 個")
                # モデルの期待する順序で特徴量を並べる
                ordered_features = [f for f in model_features if f in common_features]
                if len(ordered_features) > 0:
                    return X_test[ordered_features]

            return None

    def _get_model_feature_names(self, model: Any) -> List[str]:
        """
        モデルの特徴量名を取得

        Args:
            model: 特徴量名を取得するモデル

        Returns:
            List[str]: 特徴量名のリスト、または取得できない場合は空リスト
        """
        # LightGBMモデルからの特徴量名取得を試みる
        try:
            if hasattr(model, 'feature_name_'):
                return model.feature_name_
            elif hasattr(model, 'feature_name'):
                return model.feature_name()
            elif hasattr(model, 'booster_'):
                if hasattr(model.booster_, 'feature_name'):
                    return model.booster_.feature_name()
                elif hasattr(model.booster_, 'feature_names'):
                    return model.booster_.feature_names
            elif hasattr(model, '_Booster'):
                if hasattr(model._Booster, 'feature_name'):
                    return model._Booster.feature_name()
        except Exception as e:
            self.logger.warning(f"特徴量名の取得中にエラーが発生: {str(e)}")

        # 特徴量名を取得できなかった場合
        return []

    def _predict_with_shape_check_disabled(self, model: Any, X_test: pd.DataFrame) -> np.ndarray:
        """
        形状チェックを無効にして予測を実行

        Args:
            model: 予測に使用するモデル
            X_test: テスト特徴量

        Returns:
            np.ndarray: 予測確率
        """
        # LightGBMモデルの場合
        if hasattr(model, 'booster_'):
            self.logger.info("LightGBMのbooster_属性を使用して直接予測します")
            # predict_disable_shape_checkを使用
            return model.predict(X_test, pred_leaf=False, pred_contrib=False,
                              approx_contribs=False, pred_interactions=False,
                              predict_disable_shape_check=True)

        # その他のモデルの場合、特殊なパラメータを持つかもしれないので試行
        try:
            return model.predict(X_test, predict_disable_shape_check=True)
        except TypeError:
            # パラメータが受け入れられない場合は通常の予測を試行
            self.logger.info("通常の予測メソッドを試行します")
            return model.predict(X_test)