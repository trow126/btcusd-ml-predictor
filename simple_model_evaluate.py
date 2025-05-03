#!/usr/bin/env python
"""
シンプルなモデル評価スクリプト
CSVファイルから特徴量を読み込み、トレーニング済みモデルで評価します
"""
import pandas as pd
import numpy as np
import os
import logging
import json
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("simple_evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simple_evaluate")

def evaluate_models():
    """トレーニング済みモデルを評価"""
    try:
        # ファイルパス設定
        features_path = Path("data/processed/btcusd_5m_features.csv")
        models_dir = Path("models")
        output_dir = Path("evaluation")
        output_dir.mkdir(exist_ok=True)
        
        if not features_path.exists():
            logger.error(f"特徴量ファイルが見つかりません: {features_path}")
            return False
            
        # データ読み込み
        logger.info(f"ファイル読み込み: {features_path}")
        df = pd.read_csv(features_path)
        logger.info(f"読み込み成功: 行数={len(df)}, 列数={len(df.columns)}")
        
        # 特徴量と目標変数の確認
        target_cols = [col for col in df.columns if col.startswith("target_")]
        logger.info(f"目標変数: {len(target_cols)}個")
        
        # 特徴量の準備
        # 価格関連特徴量
        price_cols = [col for col in df.columns if (
            col.startswith("price_") or
            col in ["open", "high", "low", "close"] or
            "candle" in col or "body" in col
        )]
        
        # 出来高関連特徴量
        volume_cols = [col for col in df.columns if (
            "volume" in col or
            "turnover" in col
        )]
        
        # テクニカル指標関連特徴量
        technical_cols = [col for col in df.columns if (
            col.startswith("sma_") or
            col.startswith("ema_") or
            col.startswith("rsi") or
            col.startswith("bb_") or
            col.startswith("macd") or
            col.startswith("stoch_")
        )]
        
        # 特徴量を結合
        feature_cols = price_cols + volume_cols + technical_cols
        # 重複を削除
        feature_cols = list(set(feature_cols))
        # 目標変数を除外
        feature_cols = [col for col in feature_cols if not col.startswith("target_")]
        
        logger.info(f"使用する特徴量: {len(feature_cols)}個")
        
        # データをトレーニングとテストに分割（テスト用に20%のデータを使用）
        test_size = int(len(df) * 0.2)
        df_test = df.iloc[-test_size:]
        logger.info(f"テストデータ: {len(df_test)}行")
        
        # 評価結果を格納する辞書
        evaluation_results = {
            "binary_classification": {},
            "high_threshold_models": {}
        }
        
        # 二値分類モデルの評価
        binary_model_files = [f for f in os.listdir(models_dir) if "binary_classification_model_period_" in f]
        logger.info(f"二値分類モデルファイル: {binary_model_files}")
        
        for model_file in binary_model_files:
            try:
                period = int(model_file.split("_")[-1].split(".")[0])
                logger.info(f"二値分類モデル（期間{period}）の評価を開始")
                
                # モデル読み込み
                model_path = models_dir / model_file
                model = joblib.load(model_path)
                
                # 目標変数名を構築
                target_col = f'target_binary_{period}'
                
                if target_col in df_test.columns:
                    # 特徴量と目標変数
                    X_test = df_test[feature_cols]
                    y_test = df_test[target_col]
                    
                    # 予測
                    y_pred = model.predict(X_test)
                    
                    # 評価
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    logger.info(f"評価結果: 精度={accuracy:.4f}, 適合率={precision:.4f}, 再現率={recall:.4f}, F1={f1:.4f}")
                    
                    # 結果を格納
                    evaluation_results["binary_classification"][f"period_{period}"] = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "confusion_matrix": cm.tolist()
                    }
                else:
                    logger.warning(f"目標変数 {target_col} が見つかりません")
            except Exception as e:
                logger.error(f"二値分類モデルの評価中にエラーが発生: {str(e)}")
        
        # 高閾値シグナルモデルの評価
        high_threshold_model_files = [f for f in os.listdir(models_dir) if "high_threshold_" in f]
        logger.info(f"高閾値シグナルモデルファイル: {high_threshold_model_files}")
        
        for model_file in high_threshold_model_files:
            try:
                # モデルファイル名からパラメータを解析
                # 例: high_threshold_2p_long_model_period_1.joblib
                parts = model_file.split("_")
                threshold_str = parts[1]  # "2p"
                direction = parts[2]      # "long"
                period = int(parts[-1].split(".")[0])  # "1"
                
                logger.info(f"高閾値シグナルモデル（閾値:{threshold_str}, 方向:{direction}, 期間:{period}）の評価を開始")
                
                # モデル読み込み
                model_path = models_dir / model_file
                model = joblib.load(model_path)
                
                # 目標変数名を構築
                target_col = f'target_high_threshold_{threshold_str}_{direction}_{period}'
                
                if target_col in df_test.columns:
                    # 特徴量と目標変数
                    X_test = df_test[feature_cols]
                    y_test = df_test[target_col]
                    
                    # 予測
                    y_pred = model.predict(X_test)
                    
                    # 予測確率（閾値分析用）
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # 評価
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    logger.info(f"評価結果: 精度={accuracy:.4f}, 適合率={precision:.4f}, 再現率={recall:.4f}, F1={f1:.4f}")
                    
                    # 確信度閾値ごとの評価
                    confidence_metrics = {}
                    for conf_threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        y_pred_threshold = (y_prob >= conf_threshold).astype(int)
                        
                        # シグナル率
                        signal_rate = y_pred_threshold.mean()
                        signal_count = y_pred_threshold.sum()
                        
                        if signal_count > 0:
                            # 精度指標
                            precision_t = precision_score(y_test, y_pred_threshold, zero_division=0)
                            recall_t = recall_score(y_test, y_pred_threshold, zero_division=0)
                            f1_t = f1_score(y_test, y_pred_threshold, zero_division=0)
                            
                            logger.info(f"確信度閾値 {conf_threshold}: 適合率={precision_t:.4f}, シグナル率={signal_rate:.4f}")
                            
                            confidence_metrics[str(conf_threshold)] = {
                                "precision": float(precision_t),
                                "recall": float(recall_t),
                                "f1_score": float(f1_t),
                                "signal_rate": float(signal_rate),
                                "signal_count": int(signal_count)
                            }
                        else:
                            logger.info(f"確信度閾値 {conf_threshold}: シグナルなし")
                            
                            confidence_metrics[str(conf_threshold)] = {
                                "precision": 0.0,
                                "recall": 0.0,
                                "f1_score": 0.0,
                                "signal_rate": 0.0,
                                "signal_count": 0
                            }
                    
                    # 結果を格納
                    model_key = f"{threshold_str}_{direction}_period_{period}"
                    evaluation_results["high_threshold_models"][model_key] = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "confusion_matrix": cm.tolist(),
                        "confidence_metrics": confidence_metrics
                    }
                else:
                    logger.warning(f"目標変数 {target_col} が見つかりません")
            except Exception as e:
                logger.error(f"高閾値シグナルモデルの評価中にエラーが発生: {str(e)}")
        
        # 評価レポートをJSON形式で保存
        output_file = output_dir / "model_evaluation_report.json"
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"評価レポートを保存しました: {output_file}")
        return True
        
    except Exception as e:
        logger.exception(f"エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    success = evaluate_models()
    print("処理完了:", "成功" if success else "失敗")
