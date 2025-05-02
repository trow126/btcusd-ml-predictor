#!/usr/bin/env python
"""
高閾値シグナルモデルの評価スクリプト
訓練済みのロングとショート専用モデルを評価します
"""
import argparse
import logging
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import datetime as dt

from model_builder.evaluators.high_threshold_signal_evaluator import HighThresholdSignalEvaluator
from model_builder.utils.data.data_loader import load_data
from model_builder.utils.data.feature_selector import prepare_features
from model_builder.utils.data.data_splitter import prepare_test_data

# ロガーの設定
def setup_logger(log_file="high_threshold_evaluation.log", level=logging.INFO):
    """ロガーを設定する関数"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("high_threshold_evaluation")

def evaluate_high_threshold_models(
    data_path: str = "data/processed/btcusd_5m_features.csv",
    model_dir: str = "models/high_threshold",
    output_dir: str = "evaluation/high_threshold",
    periods: list = None,
    directions: list = None,
    thresholds: list = None,
    test_size: float = 0.2,
    save_results: bool = True,
    debug: bool = False
):
    """
    高閾値シグナルモデルを評価する関数

    Args:
        data_path: 特徴量データのパス
        model_dir: モデルが保存されているディレクトリ
        output_dir: 評価結果の保存ディレクトリ
        periods: 予測期間のリスト（デフォルト: [1, 2, 3]）
        directions: 方向のリスト（デフォルト: ["long", "short"]）
        thresholds: 閾値のリスト（デフォルト: [0.001, 0.002, 0.003, 0.005]）
        test_size: テストデータの割合
        save_results: 結果を保存するかどうか
        debug: デバッグモードを有効にするかどうか

    Returns:
        Dict: 評価結果のサマリー
    """
    # デフォルト値の設定
    if periods is None:
        periods = [1, 2, 3]
    if directions is None:
        directions = ["long", "short"]
    if thresholds is None:
        thresholds = [0.001, 0.002, 0.003, 0.005]
    
    # ロガーの設定
    log_level = logging.DEBUG if debug else logging.INFO
    logger = setup_logger(level=log_level)
    
    logger.info(f"高閾値シグナルモデルの評価を開始します")
    logger.info(f"期間: {periods}, 方向: {directions}, 閾値: {thresholds}")
    
    # 出力ディレクトリの作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    logger.info(f"データの読み込み: {data_path}")
    try:
        data_dir = str(Path(data_path).parent)
        data_file = Path(data_path).name
        df = load_data(data_dir, data_file)
    except Exception as e:
        logger.error(f"データの読み込みに失敗しました: {str(e)}")
        return None
    
    if df.empty:
        logger.error("データが空です")
        return None
        
    # データの基本情報を表示
    logger.info(f"データ情報: 行数={len(df)}, 列数={len(df.columns)}")
    
    # 高閾値ターゲット変数の確認
    high_threshold_cols = [col for col in df.columns if 'high_threshold' in col]
    if not high_threshold_cols:
        logger.warning("高閾値ターゲット変数が見つかりません。先に特徴量生成を実行してください。")
        # カラム情報の出力（最初の20個だけ表示）
        logger.info(f"利用可能なカラム（最初の20個）: {list(df.columns)[:20]}")
        logger.info(f"ターゲット変数（target_prefix付き）: {[col for col in df.columns if col.startswith('target_')][:10]}")
        return None
    else:
        logger.info(f"高閾値ターゲット変数が{len(high_threshold_cols)}個見つかりました")
        # 最初の10個だけログに出力
        for i, col in enumerate(high_threshold_cols[:10]):
            logger.info(f"  {i+1}. {col}")
        if len(high_threshold_cols) > 10:
            logger.info(f"  ...その他 {len(high_threshold_cols)-10} 個")
            
        # クラス分布の確認（最初の数個だけ）
        for col in high_threshold_cols[:3]:
            try:
                value_counts = df[col].value_counts()
                logger.info(f"{col} のクラス分布: {value_counts.to_dict()}")
            except Exception as e:
                logger.warning(f"{col} のクラス分布確認中にエラー: {e}")
                continue
    
    # テストデータの準備
    logger.info(f"テストデータの準備 (test_size={test_size})")
    X_test_dict, y_test = prepare_test_data(df, test_size, periods)
    
    if "X" not in X_test_dict or X_test_dict["X"].empty:
        logger.error("テストデータの準備に失敗しました")
        return None
    
    # 高閾値シグナル変数の確認
    high_threshold_targets = {}
    for key in y_test.keys():
        if "high_threshold" in key:
            high_threshold_targets[key] = y_test[key]
            
    logger.info(f"利用可能な高閾値シグナル変数: {len(high_threshold_targets)}個")
    
    # 評価器の初期化
    evaluator_config = {
        "model_dir": model_dir,
        "output_dir": output_dir
    }
    evaluator = HighThresholdSignalEvaluator(evaluator_config)
    
    # 全モデルの評価
    logger.info("全モデルの評価を開始します")
    results = evaluator.evaluate_all_models(
        periods=periods,
        directions=directions,
        thresholds=thresholds,
        X_test=X_test_dict["X"],
        y_dict=y_test
    )
    
    # 評価結果の要約
    summary = {
        "model_count": 0,
        "success_count": 0,
        "error_count": 0,
        "best_models": []
    }
    
    # 最良モデルを各閾値・方向ごとに特定
    if isinstance(results, dict):
        for threshold_key, threshold_data in results.items():
            if not isinstance(threshold_data, dict):
                logger.warning(f"閾値 {threshold_key} の結果が辞書形式ではありません: {type(threshold_data)}")
                continue
                
            for direction_key, direction_data in threshold_data.items():
                if not isinstance(direction_data, dict):
                    logger.warning(f"方向 {direction_key} の結果が辞書形式ではありません: {type(direction_data)}")
                    continue
                    
                for period_key, period_data in direction_data.items():
                    summary["model_count"] += 1
                    
                    if isinstance(period_data, dict) and "error" in period_data:
                        summary["error_count"] += 1
                        continue
                        
                    summary["success_count"] += 1
                    
                    # 最良モデルの候補に追加（適合率ベース）
                    if isinstance(period_data, dict) and "precision" in period_data and period_data["precision"] > 0.4: # 閾値を0.5から0.4に下げる
                        # 確信度閾値0.7での指標
                        if "confidence_metrics" in period_data:
                            conf_metrics = period_data.get("confidence_metrics", {}).get(0.7, {})
                            
                            if isinstance(conf_metrics, dict) and "precision" in conf_metrics and conf_metrics["precision"] > 0.6 and conf_metrics["signal_rate"] > 0.005: # 閾値を緩和
                                best_model = {
                                    "threshold": threshold_key,
                                    "direction": direction_key,
                                    "period": period_key,
                                    "precision": conf_metrics["precision"],
                                    "signal_rate": conf_metrics["signal_rate"],
                                    "efficiency": conf_metrics.get("trading_efficiency", 0)
                                }
                                summary["best_models"].append(best_model)
    else:
        logger.error(f"評価結果が辞書形式ではありません: {type(results)}")    
    
    # 効率順にソート
    if summary["best_models"]:
        summary["best_models"].sort(key=lambda x: x.get("efficiency", 0), reverse=True)
    
    # 結果のログ出力
    logger.info(f"評価完了: 合計 {summary['model_count']} モデル, 成功: {summary['success_count']}, エラー: {summary['error_count']}")
    
    if summary["best_models"]:
        logger.info("最良モデル（上位5件）:")
        for i, model in enumerate(summary["best_models"][:5]):
            logger.info(f"  {i+1}. 閾値: {model['threshold']}, 方向: {model['direction']}, 期間: {model['period']}")
            logger.info(f"     適合率: {model['precision']:.4f}, シグナル率: {model['signal_rate']:.4f}, 効率: {model['efficiency']:.4f}")
    else:
        logger.info("基準を満たす優良モデルは見つかりませんでした")
    
    # 結果を保存
    if save_results:
        # 評価結果
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(output_dir) / f"high_threshold_evaluation_{timestamp}.json"
        
        # JSONに変換できるように調整
        if isinstance(results, dict):
            json_results = {}
            for threshold_key, threshold_data in results.items():
                if not isinstance(threshold_data, dict):
                    continue
                json_results[threshold_key] = {}
                for direction_key, direction_data in threshold_data.items():
                    if not isinstance(direction_data, dict):
                        continue
                    json_results[threshold_key][direction_key] = {}
                    for period_key, period_data in direction_data.items():
                        if not isinstance(period_data, dict):
                            continue
                        # numpy型をPythonネイティブ型に変換
                        clean_data = {}
                        for k, v in period_data.items():
                            if k == "confidence_metrics" and isinstance(v, dict):
                                # 確信度閾値をキーとして持つ辞書
                                clean_confidence = {}
                                for conf_threshold, conf_metrics in v.items():
                                    if isinstance(conf_metrics, dict):
                                        # 文字列キーに変換
                                        clean_confidence[str(conf_threshold)] = {
                                            ck: float(cv) if isinstance(cv, (np.float32, np.float64)) else cv
                                            for ck, cv in conf_metrics.items()
                                        }
                                clean_data[k] = clean_confidence
                            elif isinstance(v, (np.float32, np.float64)):
                                clean_data[k] = float(v)
                            else:
                                clean_data[k] = v
                        json_results[threshold_key][direction_key][period_key] = clean_data
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"評価結果を {results_file} に保存しました")
        else:
            logger.error("評価結果が辞書形式ではないため、保存をスキップします")
        
        # サマリー
        summary_file = Path(output_dir) / f"high_threshold_summary_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"評価サマリーを {summary_file} に保存しました")
    
    logger.info("高閾値シグナルモデルの評価が完了しました")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高閾値シグナルモデルの評価スクリプト")
    parser.add_argument("--data-path", type=str, default="data/processed/btcusd_5m_features.csv", help="特徴量データのパス")
    parser.add_argument("--model-dir", type=str, default="models/high_threshold", help="モデルが保存されているディレクトリ")
    parser.add_argument("--output-dir", type=str, default="evaluation/high_threshold", help="評価結果の保存ディレクトリ")
    parser.add_argument("--periods", type=int, nargs="+", default=[1, 2, 3], help="予測期間")
    parser.add_argument("--directions", type=str, nargs="+", default=["long", "short"], help="予測方向")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.002, 0.003, 0.005], help="シグナル閾値")
    parser.add_argument("--test-size", type=float, default=0.2, help="テストデータの割合")
    parser.add_argument("--no-save", action="store_true", help="結果を保存しない")
    parser.add_argument("--debug", action="store_true", help="デバッグモードを有効化")
    
    args = parser.parse_args()
    
    evaluate_high_threshold_models(
        data_path=args.data_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        periods=args.periods,
        directions=args.directions,
        thresholds=args.thresholds,
        test_size=args.test_size,
        save_results=not args.no_save,
        debug=args.debug
    )