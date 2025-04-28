# model_builder/evaluators/threshold_binary_classification_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional

from .base_evaluator import BaseEvaluator

class ThresholdBinaryClassificationEvaluator(BaseEvaluator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        閾値ベースの二値分類モデル評価クラス
        横ばいを除外し、有意な上昇と下落のみを分類

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
        
        # データがない場合
        if X_test.empty or len(y_test) == 0:
            self.logger.warning(f"evaluate: テストデータが空です。期間: {period}")
            return {
                "error": "empty_data",
                "message": "テストデータが空です"
            }
        
        try:
            # モデルの特徴量名を取得し、特徴量の整合性を確認
            self.logger.info(f"X_testの形状: {X_test.shape}, 特徴量数: {X_test.shape[1]}")
            
            # LightGBMモデルの特徴量名を取得
            if hasattr(model, 'feature_name_'):
                model_features = model.feature_name_
                self.logger.info(f"モデルが期待する特徴量数: {len(model_features)}")
                
                # テストデータの特徴量が多すぎる場合は調整
                if set(model_features).issubset(set(X_test.columns)):
                    self.logger.info(f"特徴量を調整します: {X_test.shape} → {(X_test.shape[0], len(model_features))}")
                    X_test = X_test[model_features]
                    self.logger.info(f"調整後のX_testの形状: {X_test.shape}")
                else:
                    missing_features = set(model_features) - set(X_test.columns)
                    self.logger.error(f"テストデータに不足している特徴量があります: {missing_features}")
            # モデルが特徴量名を直接持っていない場合、他の属性をチェック
            elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
                model_features = model.booster_.feature_name()
                self.logger.info(f"booster_属性から特徴量名を取得しました: {len(model_features)}")
                
                if set(model_features).issubset(set(X_test.columns)):
                    self.logger.info(f"特徴量を調整します: {X_test.shape} → {(X_test.shape[0], len(model_features))}")
                    X_test = X_test[model_features]
                else:
                    missing_features = set(model_features) - set(X_test.columns)
                    self.logger.error(f"テストデータに不足している特徴量があります: {missing_features}")
            else:
                self.logger.warning(f"モデルの特徴量名を取得できません。予測時に入力特徴量の形状無視オプションを使用します")
            
            # 二値分類の予測確率
            try:
                y_pred_proba = model.predict(X_test)
            except Exception as predict_error:
                if "shape" in str(predict_error).lower() or "feature" in str(predict_error).lower():
                    self.logger.warning(f"形状チェックエラーが発生しました: {str(predict_error)}")
                    self.logger.info(f"predict_disable_shape_check=Trueオプションで再試行します")
                    try:
                        # LightGBMの場合、形状チェックをスキップするオプションを指定
                        if hasattr(model, 'booster_'):
                            self.logger.info("モデルのbooster_属性を使用して直接予測します")
                            # booster_から直接予測
                            y_pred_proba = model.booster_.predict(X_test.values, pred_leaf=False, pred_contrib=False, 
                                                               approx_contribs=False, pred_interactions=False, 
                                                               predict_disable_shape_check=True)
                        else:
                            self.logger.info("形状チェック無効オプションを指定して予測します")
                            y_pred_proba = model.predict(X_test, predict_disable_shape_check=True)
                    except Exception as retry_error:
                        self.logger.error(f"形状チェック無効オプションでの再試行も失敗しました: {str(retry_error)}")
                        return {
                            "error": "prediction_error",
                            "message": str(retry_error)
                        }
                else:
                    # 形状以外のエラーの場合はそのままエラーとして処理
                    raise predict_error
            # 閾値で分類（デフォルトは0.5）
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 評価指標の計算
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            
            # 各クラスの割合を計算
            class_counts = pd.Series(y_test).value_counts().to_dict()
            
            result = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "class_distribution": class_counts,
                "samples": len(y_test)
            }
            
            self.logger.info(f"evaluate: {period}期先の閾値ベース二値分類モデル - 正解率: {accuracy:.4f}, F1: {f1:.4f}")
            self.logger.info(f"evaluate: クラス分布: {class_counts}")
            
            # 詳細なログ
            self.logger.info(f"混同行列: \n{cm}")
            self.logger.info(f"精度 (Precision): {precision:.4f}")
            self.logger.info(f"再現率 (Recall): {recall:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"evaluate: 評価中にエラーが発生しました: {str(e)}")
            return {
                "error": "evaluation_error",
                "message": str(e)
            }