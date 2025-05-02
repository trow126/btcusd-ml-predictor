#!/usr/bin/env python
"""
高閾値シグナルモデルのトレーニングスクリプト
ロングとショート専用モデルを異なる閾値で訓練します
"""
import argparse
import logging
import pandas as pd
import json
import os
from pathlib import Path
import datetime as dt

from model_builder.trainers.high_threshold_signal_trainer import HighThresholdSignalTrainer
from model_builder.utils.data.data_loader import load_data
from model_builder.utils.data.feature_selector import prepare_features
from model_builder.utils.data.data_splitter import train_test_split

# ロガーの設定
def setup_logger(log_file="high_threshold_training.log", level=logging.INFO):
    """ロガーを設定する関数"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("high_threshold_training")

def train_high_threshold_models(
    data_path: str = "data/processed/btcusd_5m_features.csv",
    output_dir: str = "models/high_threshold",
    periods: list = None,
    directions: list = None,
    thresholds: list = None,
    save_results: bool = True,
    debug: bool = False
):
    """
    高閾値シグナルモデルを訓練する関数

    Args:
        data_path: 特徴量データのパス
        output_dir: モデル保存ディレクトリ
        periods: 予測期間のリスト（デフォルト: [1, 2, 3]）
        directions: 方向のリスト（デフォルト: ["long", "short"]）
        thresholds: 閾値のリスト（デフォルト: [0.001, 0.002, 0.003, 0.005]）
        save_results: 結果を保存するかどうか
        debug: デバッグモードを有効にするかどうか

    Returns:
        Dict: 訓練結果のサマリー
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
    
    logger.info(f"高閾値シグナルモデルの訓練を開始します")
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
    
    # 特徴量と目標変数の準備
    logger.info(f"特徴量と目標変数の準備")
    feature_groups = {"price": True, "volume": True, "technical": True}
    X_dict, y_dict = prepare_features(df, feature_groups, periods)
    
    # データ分割
    logger.info(f"トレーニングデータとテストデータの分割")
    X_train, X_test, y_train, y_test = train_test_split(X_dict["X"], y_dict, test_size=0.2)
    
    # 訓練結果の保存用
    results = {}
    
    # 高閾値シグナル変数の確認
    logger.info("利用可能な目標変数の確認:")
    for key in y_train.keys():
        if "high_threshold" in key:
            logger.info(f"- {key}")
    
    # モデル訓練
    for threshold in thresholds:
        threshold_str = str(int(threshold * 1000))
        threshold_results = {}
        
        for direction in directions:
            direction_results = {}
            
            for period in periods:
                logger.info(f"閾値:{threshold*100}% 方向:{direction} 期間:{period}のモデルを訓練")
                
                # 目標変数名の生成と存在確認
                # 元の目標変数名の形式を使用
                target_name = f"target_high_threshold_{threshold_str}p_{direction}_{period}"
                if target_name not in y_train:
                    logger.warning(f"目標変数 {target_name} が見つかりません。スキップします。")
                    continue
                
                # モデルトレーナーの初期化
                trainer_config = {
                    "output_dir": output_dir
                }
                trainer = HighThresholdSignalTrainer(trainer_config)
                
                # モデルトレーニング
                result = trainer.train(
                    X_train, X_test, y_train, y_test,
                    period=period, direction=direction, threshold=threshold
                )
                
                # エラーチェック
                if "error" in result:
                    logger.error(f"モデルトレーニングに失敗しました: {result.get('message', 'Unknown error')}")
                    direction_results[f"period_{period}"] = {
                        "error": result.get("error"),
                        "message": result.get("message", "Unknown error")
                    }
                    continue
                
                # 結果の要約
                summary = {
                    "accuracy": result.get("accuracy"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                    "f1_score": result.get("f1_score"),
                    "signal_ratio": result.get("signal_ratio"),
                    "train_samples": result.get("train_samples"),
                    "test_samples": result.get("test_samples"),
                    "confidence_metrics": result.get("confidence_metrics")
                }
                
                # 結果のログ出力
                logger.info(f"モデル評価: 精度={summary['accuracy']:.4f}, 適合率={summary['precision']:.4f}, 再現率={summary['recall']:.4f}")
                
                # 確信度閾値ごとの結果
                for conf_thresh, metrics in result.get("confidence_metrics", {}).items():
                    if "precision" in metrics:
                        prec = metrics.get("precision")
                        sig_rate = metrics.get("signal_rate")
                        sig_count = metrics.get("signal_count")
                        logger.info(f"  確信度閾値={conf_thresh:.1f}: 適合率={prec:.4f}, シグナル率={sig_rate:.4f}, シグナル数={sig_count}")
                
                # 特徴量重要度のトップ5をログ出力
                if "feature_importance" in result and isinstance(result["feature_importance"], list) and len(result["feature_importance"]) > 0:
                    logger.info("重要な特徴量（トップ5）:")
                    for i, (feature, importance) in enumerate(result["feature_importance"][:min(5, len(result["feature_importance"]))]):
                        logger.info(f"  {i+1}. {feature}: {importance:.4f}")
                else:
                    logger.info("特徴量重要度情報が利用できません")
                
                # モデルオブジェクトを削除（JSON保存のため）
                if "model" in summary:
                    del summary["model"]
                
                # 結果を格納
                direction_results[f"period_{period}"] = summary
            
            threshold_results[direction] = direction_results
        
        results[f"threshold_{threshold_str}p"] = threshold_results
    
    # 結果のサマリーをログに出力
    logger.info("===== トレーニング結果サマリー =====")
    for threshold_key, threshold_data in results.items():
        logger.info(f"閾値: {threshold_key}")
        for direction_key, direction_data in threshold_data.items():
            logger.info(f"  方向: {direction_key}")
            for period_key, period_data in direction_data.items():
                if "accuracy" in period_data:
                    logger.info(f"    期間: {period_key}")
                    logger.info(f"      精度: {period_data['accuracy']:.4f}")
                    logger.info(f"      適合率: {period_data['precision']:.4f}")
                    logger.info(f"      シグナル率: {period_data.get('signal_ratio', 0):.4f}")
    
    # 結果をJSONファイルに保存
    if save_results:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(output_dir) / f"high_threshold_results_{timestamp}.json"
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"訓練結果を {results_file} に保存しました")
    
    logger.info("高閾値シグナルモデルの訓練が完了しました")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高閾値シグナルモデルのトレーニングスクリプト")
    parser.add_argument("--data-path", type=str, default="data/processed/btcusd_5m_features.csv", help="特徴量データのパス")
    parser.add_argument("--output-dir", type=str, default="models/high_threshold", help="モデル保存ディレクトリ")
    parser.add_argument("--periods", type=int, nargs="+", default=[1, 2, 3], help="予測期間")
    parser.add_argument("--directions", type=str, nargs="+", default=["long", "short"], help="予測方向")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.002, 0.003, 0.005], help="シグナル閾値")
    parser.add_argument("--no-save", action="store_true", help="結果を保存しない")
    parser.add_argument("--debug", action="store_true", help="デバッグモードを有効化")
    
    args = parser.parse_args()
    
    train_high_threshold_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
        periods=args.periods,
        directions=args.directions,
        thresholds=args.thresholds,
        save_results=not args.no_save,
        debug=args.debug
    )