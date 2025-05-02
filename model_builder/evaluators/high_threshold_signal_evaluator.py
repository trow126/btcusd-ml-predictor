# model_builder/evaluators/high_threshold_signal_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import joblib
from pathlib import Path

from .base_evaluator import BaseEvaluator

class HighThresholdSignalEvaluator(BaseEvaluator):
    """
    高閾値シグナルモデル評価クラス
    特定の方向（ロングまたはショート）に特化したモデルを評価
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        super().__init__(config)

    def evaluate(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
        period: int, direction: str = "long", threshold: float = 0.002
    ) -> Dict[str, Any]:
        """
        高閾値シグナルモデルを評価

        Args:
            model: 評価するモデル
            X_test: テスト特徴量DataFrame
            y_test: テスト目標変数のSeries
            period: 予測期間
            direction: "long" または "short"
            threshold: シグナル閾値（0.002 = 0.2%）

        Returns:
            Dict: 評価結果
        """
        threshold_str = str(int(threshold * 1000))
        self.logger.info(f"evaluate: {period}期先の高閾値({threshold*100:.1f}%)シグナルモデル（{direction}）を評価します")

        try:
            # データが空の場合はエラー
            if X_test.empty or len(y_test) == 0:
                self.logger.warning(f"evaluate: テストデータが空です")
                return {
                    "error": "empty_data",
                    "message": "テストデータが空です"
                }

            # NaN値を持つサンプルをフィルタリング（通常はないはず）
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

            # シグナル比率の計算
            signal_ratio = y_test.mean()
            self.logger.info(f"シグナル比率: {signal_ratio:.4f}")

            # モデルによる予測確率
            y_pred_proba = model.predict(X_test)

            # 確信度閾値ごとの評価
            confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            confidence_metrics = {}
            
            for conf_threshold in confidence_thresholds:
                # 閾値を適用
                y_pred = (y_pred_proba >= conf_threshold).astype(int)
                
                # シグナル発生率と数
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
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # 混同行列
                    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    
                    # 真陽性(TP)と偽陽性(FP)
                    tp = conf_matrix[1, 1]  # 真陽性: 実際に1で予測も1
                    fp = conf_matrix[0, 1]  # 偽陽性: 実際は0だが予測は1
                    
                    # トレード効率指標（真陽性率 / シグナル率）
                    if signal_rate > 0:
                        trading_efficiency = precision / signal_rate
                    else:
                        trading_efficiency = 0.0
                    
                    confidence_metrics[conf_threshold] = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "confusion_matrix": conf_matrix.tolist(),
                        "signal_rate": float(signal_rate),
                        "signal_count": int(signal_count),
                        "true_positives": int(tp),
                        "false_positives": int(fp),
                        "trading_efficiency": float(trading_efficiency)
                    }
                except Exception as e:
                    self.logger.error(f"確信度閾値 {conf_threshold} での評価中にエラー: {str(e)}")
                    confidence_metrics[conf_threshold] = {
                        "error": str(e),
                        "signal_rate": float(signal_rate),
                        "signal_count": int(signal_count)
                    }

            # デフォルト閾値（0.5）での評価
            default_metrics = confidence_metrics.get(0.5, {})
            default_accuracy = default_metrics.get("accuracy", np.nan)
            default_precision = default_metrics.get("precision", np.nan)
            default_recall = default_metrics.get("recall", np.nan)
            default_f1 = default_metrics.get("f1_score", np.nan)
            
            # 評価結果
            result = {
                "model_type": "high_threshold_signal",
                "direction": direction,
                "threshold": float(threshold),
                "period": int(period),
                "accuracy": float(default_accuracy),
                "precision": float(default_precision),
                "recall": float(default_recall),
                "f1_score": float(default_f1),
                "confidence_metrics": confidence_metrics,
                "test_samples": len(X_test),
                "signal_ratio": float(signal_ratio)
            }

            # 結果のログ出力
            self.logger.info(f"評価指標(閾値0.5) - 精度: {default_accuracy:.4f}, 適合率: {default_precision:.4f}, "
                           f"再現率: {default_recall:.4f}, F1: {default_f1:.4f}")
            
            # 確信度閾値ごとの結果をログ出力
            self.logger.info("確信度閾値ごとの評価:")
            for threshold, metrics in confidence_metrics.items():
                if "precision" in metrics:
                    self.logger.info(f"  閾値 {threshold:.1f}: 適合率={metrics['precision']:.4f}, 発生率={metrics['signal_rate']:.4f}, "
                                  f"シグナル数={metrics['signal_count']}, 効率={metrics.get('trading_efficiency', 0):.4f}")

            return result
        except Exception as e:
            self.logger.error(f"evaluate: 評価中にエラーが発生しました: {str(e)}")
            import traceback
            self.logger.error(f"トレースバック: {traceback.format_exc()}")
            return {
                "error": "evaluation_error",
                "message": str(e)
            }

    def evaluate_all_models(
        self, periods: List[int] = None, 
        directions: List[str] = None, 
        thresholds: List[float] = None, 
        X_test: pd.DataFrame = None, 
        y_dict: Dict[str, pd.Series] = None
    ) -> Dict[str, Any]:
        """
        すべての高閾値シグナルモデルを評価

        Args:
            periods: 予測期間のリスト（デフォルト: [1, 2, 3]）
            directions: 方向のリスト（デフォルト: ["long", "short"]）
            thresholds: 閾値のリスト（デフォルト: [0.001, 0.002, 0.003, 0.005]）
            X_test: テスト特徴量DataFrame
            y_dict: テスト目標変数のDict

        Returns:
            Dict: 評価結果のDict
        """
        # デフォルト値の設定
        if periods is None:
            periods = [1, 2, 3]
        if directions is None:
            directions = ["long", "short"]
        if thresholds is None:
            thresholds = [0.001, 0.002, 0.003, 0.005]
            
        if X_test is None or y_dict is None:
            self.logger.error("テストデータが指定されていません")
            return {"error": "missing_test_data"}
            
        self.logger.info(f"evaluate_all_models: すべての高閾値シグナルモデルを評価します")
        self.logger.info(f"期間: {periods}, 方向: {directions}, 閾値: {thresholds}")
        
        # 結果の保存用
        results = {}
        
        # モデルディレクトリ
        model_dir = Path(self.config.get("model_dir", "models/high_threshold"))
        
        if not model_dir.exists():
            self.logger.error(f"モデルディレクトリが存在しません: {model_dir}")
            return {"error": "model_dir_not_found", "path": str(model_dir)}
            
        # 各モデルを評価
        for threshold in thresholds:
            threshold_str = str(int(threshold * 1000))
            threshold_results = {}
            
            for direction in directions:
                direction_results = {}
                
                for period in periods:
                    # 目標変数名とモデルファイル名の構築
                    target_name = f"target_high_threshold_{threshold_str}p_{direction}_{period}"
                    model_filename = f"target_high_threshold_{threshold_str}p_{direction}_{period}.joblib"
                    model_path = model_dir / model_filename
                    
                    # 目標変数の存在確認
                    if target_name not in y_dict:
                        self.logger.warning(f"目標変数 {target_name} が見つかりません。スキップします。")
                        direction_results[f"period_{period}"] = {
                            "error": "target_not_found",
                            "message": f"目標変数 {target_name} が見つかりません"
                        }
                        continue
                        
                    # モデルの存在確認
                    if not model_path.exists():
                        self.logger.warning(f"モデルファイル {model_path} が見つかりません。スキップします。")
                        direction_results[f"period_{period}"] = {
                            "error": "model_not_found",
                            "message": f"モデルファイル {model_path} が見つかりません"
                        }
                        continue
                        
                    # モデルの読み込みと評価
                    try:
                        self.logger.info(f"モデル {model_path} を読み込み中...")
                        model = joblib.load(model_path)
                        
                        # モデルの評価
                        self.logger.info(f"モデル {model_path} を評価中...")
                        eval_result = self.evaluate(
                            model, X_test, y_dict[target_name], 
                            period=period, direction=direction, threshold=threshold
                        )
                        
                        # 結果を格納
                        direction_results[f"period_{period}"] = eval_result
                        self.logger.info(f"モデル {model_path} の評価が完了しました")
                    except Exception as e:
                        self.logger.error(f"モデル {model_path} の評価中にエラーが発生しました: {str(e)}")
                        direction_results[f"period_{period}"] = {
                            "error": "evaluation_error",
                            "message": str(e)
                        }
                
                threshold_results[direction] = direction_results
            
            results[f"threshold_{threshold_str}p"] = threshold_results
            
        # 結果のサマリーをログに出力
        self.logger.info("===== 評価結果サマリー =====")
        for threshold_key, threshold_data in results.items():
            self.logger.info(f"閾値: {threshold_key}")
            for direction_key, direction_data in threshold_data.items():
                self.logger.info(f"  方向: {direction_key}")
                for period_key, period_data in direction_data.items():
                    if "error" in period_data:
                        self.logger.info(f"    期間: {period_key}: エラー - {period_data['error']}")
                    elif "accuracy" in period_data:
                        self.logger.info(f"    期間: {period_key}")
                        self.logger.info(f"      精度: {period_data['accuracy']:.4f}")
                        self.logger.info(f"      適合率: {period_data['precision']:.4f}")
                        self.logger.info(f"      シグナル率: {period_data.get('signal_ratio', 0):.4f}")
        
        return results