#!/usr/bin/env python
"""
BTCUSD ML Predictor - ビットコイン価格予測パイプライン
コマンドライン引数でパイプラインのステップを制御できます
"""
import asyncio
import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import datetime as dt

from data_collector.bybit_collector import BTCDataCollector
from data_processor.optimized_feature_generator import OptimizedFeatureGenerator
from model_builder.trainers import ModelTrainer
from model_builder.evaluators import ModelEvaluator
from model_builder.utils.config_loader import load_json_config

# ロガーの設定
def setup_logger(log_file="btcusd_ml_predictor.log", level=logging.INFO):
    """ロガーを設定する関数"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("btcusd_predictor")

# 閾値ベースの二値分類ターゲットを検証
def verify_threshold_binary_targets(features_df, logger):
    """
    閾値ベースの二値分類ターゲットの存在と分布を検証
    """
    logger.info("閾値ベースの二値分類ターゲット変数を確認")

    # 閾値ベースの二値分類ターゲットがあるか確認
    threshold_binary_cols = [col for col in features_df.columns if "target_threshold_binary" in col]

    if not threshold_binary_cols:
        logger.warning("閾値ベースの二値分類ターゲットが見つかりません")
        return False

    logger.info(f"閾値ベース二値分類ターゲット列: {threshold_binary_cols}")

    # クラスバランスを確認
    for col in threshold_binary_cols:
        # 統計情報
        valid_count = features_df[col].dropna().count()
        nan_count = features_df[col].isna().sum()

        logger.info(f"{col} の統計: 有効データ {valid_count}行, NaN {nan_count}行")

        # クラスバランス
        if valid_count > 0:
            class_dist = features_df[col].value_counts(dropna=True)
            logger.info(f"{col} のクラス分布: {class_dist.to_dict()}")

            # クラスバランスチェック
            if len(class_dist) > 1:
                ratio = class_dist.min() / class_dist.max()
                if ratio < 0.3:  # 極端な不均衡の場合は警告
                    logger.warning(f"{col} のクラス不均衡が検出されました: {ratio:.2f}")

    return True

# 設定ロード関数
def load_configs(logger):
    """設定ファイルをロードする関数"""
    try:
        # 設定ファイルパスの設定
        data_config_path = Path("config") / "data_config.json"
        model_config_path = Path("config") / "model_config.json"

        # 設定ファイルの存在確認
        if not data_config_path.exists():
            logger.error(f"データ設定ファイルが見つかりません: {data_config_path}")
            return None

        if not model_config_path.exists():
            logger.error(f"モデル設定ファイルが見つかりません: {model_config_path}")
            return None

        # 設定のロード
        data_config = load_json_config(str(data_config_path))
        model_config = load_json_config(str(model_config_path))

        # 設定を統合
        config = {
            "data_collector": data_config,
            "data_processor": model_config.get("data_processor", {}),
            "model_trainer": model_config.get("model_trainer", {}),
            "model_evaluator": model_config.get("model_evaluator", {})
        }

        logger.info("設定ファイルをロードしました")
        return config
    except Exception as e:
        logger.error(f"設定ロード中にエラーが発生しました: {str(e)}")
        return None

# 1. データ収集ステップ
async def collect_data(config, logger):
    """データ収集ステップ"""
    logger.info("データ収集を開始します")

    try:
        # 日付変換処理
        data_collector_config = config.get("data_collector", {})

        # start_dateとend_dateが文字列の場合はdatetimeオブジェクトに変換
        if isinstance(data_collector_config.get("start_date"), str):
            start_date_str = data_collector_config.get("start_date")
            try:
                data_collector_config["start_date"] = dt.datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
            except ValueError:
                # エラーが発生した場合はデフォルト値を使用
                data_collector_config["start_date"] = dt.datetime(2023, 1, 1)
                logger.warning(f"無効な開始日フォーマット: {start_date_str}、デフォルト値を使用します")

        if isinstance(data_collector_config.get("end_date"), str) and data_collector_config.get("end_date"):
            end_date_str = data_collector_config.get("end_date")
            try:
                data_collector_config["end_date"] = dt.datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except ValueError:
                # エラーが発生した場合は現在時刻を使用
                data_collector_config["end_date"] = dt.datetime.now()
                logger.warning(f"無効な終了日フォーマット: {end_date_str}、現在時刻を使用します")
        elif data_collector_config.get("end_date") is None:
            data_collector_config["end_date"] = dt.datetime.now()

        # データコレクタの初期化と実行
        data_collector = BTCDataCollector(data_collector_config)
        historical_data = await data_collector.collect_historical_data()

        if historical_data.empty:
            logger.error("データ収集に失敗しました")
            return None

        # データ保存
        data_collector.save_data(historical_data)
        logger.info(f"データ収集と保存が完了しました: {len(historical_data)}行")

        return historical_data
    except Exception as e:
        logger.error(f"データ収集中にエラーが発生しました: {str(e)}")
        return None

# 2. 特徴量生成ステップ
def generate_features(config, input_data, logger):
    """特徴量生成ステップ"""
    logger.info("特徴量エンジニアリングを開始します")

    try:
        # 入力データチェック
        if input_data is None or input_data.empty:
            logger.error("特徴量生成の入力データが空または無効です")

            # 保存済みのデータがあれば読み込み
            raw_data_path = Path("data/raw/btcusd_5m_data.csv")
            if raw_data_path.exists():
                logger.info(f"保存済みのデータを読み込みます: {raw_data_path}")
                input_data = pd.read_csv(raw_data_path, parse_dates=["timestamp"])

                if input_data.empty:
                    logger.error("保存済みデータの読み込みに失敗しました")
                    return None
            else:
                logger.error("保存済みデータもありません")
                return None

        # 特徴量ジェネレータの初期化と実行
        feature_generator = OptimizedFeatureGenerator(config.get("data_processor"))
        features_df = feature_generator.generate_features(input_data)

        if features_df.empty:
            logger.error("特徴量生成に失敗しました")
            return None

        # 特徴量の保存
        feature_generator.save_features(features_df)
        logger.info(f"特徴量エンジニアリングと保存が完了しました: {len(features_df)}行, {len(features_df.columns)}列")

        # 閾値ベースの二値分類ターゲットを検証
        verify_threshold_binary_targets(features_df, logger)

        return features_df
    except Exception as e:
        logger.error(f"特徴量生成中にエラーが発生しました: {str(e)}")
        return None

# 3. モデルトレーニングステップ
def train_models(config, features_df, logger):
    """モデルトレーニングステップ"""
    logger.info("モデル訓練を開始します")

    try:
        # 入力チェック
        if features_df is None or features_df.empty:
            logger.error("モデルトレーニングの入力データが空または無効です")

            # データローダーから読み込み
            model_trainer = ModelTrainer(config.get("model_trainer"))
            features_df = model_trainer.load_data()

            if features_df.empty:
                logger.error("保存済み特徴量データの読み込みに失敗しました")
                return None

        # モデルトレーナの初期化
        model_trainer = ModelTrainer(config.get("model_trainer"))

        # 閾値ベースの二値分類ターゲットを検証
        verify_threshold_binary_targets(features_df, logger)

        # 特徴量と目標変数の準備
        X_dict, y_dict = model_trainer.prepare_features(features_df)
        if not X_dict or not y_dict:
            logger.error("特徴量と目標変数の準備に失敗しました")
            return None

        # 閾値ベースの二値分類ターゲットがあるか確認
        threshold_keys = [key for key in y_dict.keys() if "threshold_binary_classification" in key]
        if not threshold_keys:
            logger.warning("閾値ベースの二値分類ターゲットが準備されていません")
            logger.info(f"利用可能なターゲット: {list(y_dict.keys())}")
        else:
            logger.info(f"閾値ベースの二値分類ターゲット: {threshold_keys}")

        # データ分割
        X_train, X_test, y_train, y_test = model_trainer.train_test_split(X_dict["X"], y_dict)
        if not X_train or not X_test or not y_train or not y_test:
            logger.error("訓練/テストデータの分割に失敗しました")
            return None

        # 特殊な特徴量セットのデータ分割処理
        for key in X_dict:
            if key != "X" and key not in X_train:
                # データの総行数から、トレーニングとテストのデータ数の比率を計算
                train_ratio = len(X_train["X"]) / len(X_dict["X"])

                # 特殊なデータセットをトレーニングとテストに分割
                special_data = X_dict[key]
                train_size = int(len(special_data) * train_ratio)

                # 時系列データなので、前半をトレーニング、後半をテストに
                X_train[key] = special_data.iloc[:train_size]
                X_test[key] = special_data.iloc[train_size:]

                logger.info(f"特殊特徴量セット '{key}' の分割: トレーニング({len(X_train[key])}行), テスト({len(X_test[key])}行)")

        # モデルトレーニング - 各モデルタイプごとに実行
        training_results = {}

        # 1. 回帰モデル（価格変動率予測）
        logger.info("回帰モデルのトレーニングを開始")
        regression_results = model_trainer.train_regression_models(X_train, X_test, y_train, y_test)
        training_results["regression"] = regression_results

        # 2. 分類モデル（価格変動方向予測）
        classification_results = None
        if config.get("model_trainer", {}).get("use_classification", True):
            logger.info("分類モデルのトレーニングを開始")
            classification_results = model_trainer.train_classification_models(X_train, X_test, y_train, y_test)
            training_results["classification"] = classification_results
        else:
            logger.info("分類モデルのトレーニングは設定により無効化されています")
            training_results["classification"] = {}

        # 3. 二値分類モデル（上昇/下落予測） - オプション
        binary_classification_results = None
        if config.get("model_trainer", {}).get("use_binary_classification", True):
            logger.info("二値分類モデルのトレーニングを開始")
            binary_classification_results = model_trainer.train_binary_classification_models(X_train, X_test, y_train, y_test)
            training_results["binary_classification"] = binary_classification_results
        else:
            logger.info("二値分類モデルのトレーニングは設定により無効化されています")
            training_results["binary_classification"] = {}

        # 4. 閾値ベースの二値分類モデル（有意な上昇/下落予測）
        threshold_binary_classification_results = None
        if config.get("model_trainer", {}).get("use_threshold_binary_classification", True):
            logger.info("閾値ベース分類モデルのトレーニングを開始")
            threshold_binary_classification_results = model_trainer.train_threshold_binary_classification_models(X_train, X_test, y_train, y_test)
            training_results["threshold_binary_classification"] = threshold_binary_classification_results

            # 結果のサマリー
            if threshold_binary_classification_results:
                success_count = sum(1 for result in threshold_binary_classification_results.values()
                                  if not (isinstance(result, dict) and "error" in result))
                error_count = sum(1 for result in threshold_binary_classification_results.values()
                                if isinstance(result, dict) and "error" in result)
                logger.info(f"閾値ベース分類モデルの結果: 成功 {success_count}, 失敗 {error_count}")

        # トレーニングレポートの生成
        report = model_trainer.generate_training_report(
            regression_results,
            classification_results,
            binary_classification_results,
            threshold_binary_classification_results
        )

        logger.info("モデル訓練が完了しました")
        return training_results
    except Exception as e:
        logger.error(f"モデルトレーニング中にエラーが発生しました: {str(e)}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return None

# 4. モデル評価ステップ
def evaluate_models(config, features_df, logger):
    """モデル評価ステップ"""
    logger.info("モデル評価を開始します")

    try:
        # モデル評価器の初期化
        model_evaluator = ModelEvaluator(config.get("model_evaluator"))

        # 評価用特徴量データの準備
        if features_df is None or features_df.empty:
            logger.info("評価用データを読み込み中...")
            features_df = model_evaluator.load_data()

            if features_df.empty:
                logger.error("評価用特徴量データの読み込みに失敗しました")
                return None

        # 閾値ベースの二値分類ターゲットを検証
        verify_threshold_binary_targets(features_df, logger)

        # モデルの読み込み
        if not model_evaluator.load_models():
            logger.error("モデルの読み込みに失敗しました")
            return None

        # テストデータの準備
        X_test_dict, y_test = model_evaluator.prepare_test_data(features_df)

        # データチェック
        if not X_test_dict or "X" not in X_test_dict or X_test_dict["X"].empty or not y_test:
            logger.error("評価用テストデータの準備に失敗しました")
            return None

        # モデル評価の実行
        evaluation_results = model_evaluator.evaluate_models(X_test_dict, y_test)

        # 評価レポートの生成と保存
        evaluation_report = model_evaluator.generate_evaluation_report(evaluation_results)
        model_evaluator.save_evaluation_report(evaluation_report)

        logger.info("モデル評価とレポート保存が完了しました")
        return evaluation_report
    except Exception as e:
        logger.error(f"モデル評価中にエラーが発生しました: {str(e)}")
        return None

# フルパイプライン実行関数
async def run_full_pipeline(args, logger):
    """フルパイプラインの実行"""
    logger.info("=== BTCUSD ML Predictor パイプラインを開始します ===")

    # 1. 設定をロード
    config = load_configs(logger)
    if config is None:
        logger.error("設定のロードに失敗しました。パイプラインを中断します。")
        return False

    # 2. データ収集
    if args.skip_data_collection:
        logger.info("データ収集ステップをスキップします")
        historical_data = None
    else:
        historical_data = await collect_data(config, logger)
        if historical_data is None and not args.continue_on_error:
            logger.error("データ収集に失敗しました。パイプラインを中断します。")
            return False

    # 3. 特徴量生成
    if args.skip_feature_generation:
        logger.info("特徴量生成ステップをスキップします")
        features_df = None
    else:
        features_df = generate_features(config, historical_data, logger)
        if features_df is None and not args.continue_on_error:
            logger.error("特徴量生成に失敗しました。パイプラインを中断します。")
            return False

    # 4. モデルトレーニング
    if args.skip_training:
        logger.info("モデルトレーニングステップをスキップします")
        training_results = None
    else:
        training_results = train_models(config, features_df, logger)
        if training_results is None and not args.continue_on_error:
            logger.error("モデルトレーニングに失敗しました。パイプラインを中断します。")
            return False

    # 5. モデル評価
    if args.skip_evaluation:
        logger.info("モデル評価ステップをスキップします")
        evaluation_results = None
    else:
        evaluation_results = evaluate_models(config, features_df, logger)
        if evaluation_results is None and not args.continue_on_error:
            logger.error("モデル評価に失敗しました。パイプラインを中断します。")
            return False

    logger.info("=== BTCUSD ML Predictor パイプラインが完了しました ===")
    return True

# メイン関数
async def main():
    """メイン関数"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description="BTCUSD ML Predictor - ビットコイン価格予測パイプライン")
    parser.add_argument("--skip-data-collection", action="store_true", help="データ収集ステップをスキップ")
    parser.add_argument("--skip-feature-generation", action="store_true", help="特徴量生成ステップをスキップ")
    parser.add_argument("--skip-training", action="store_true", help="モデルトレーニングステップをスキップ")
    parser.add_argument("--skip-evaluation", action="store_true", help="モデル評価ステップをスキップ")
    parser.add_argument("--continue-on-error", action="store_true", help="エラー発生時も続行")
    parser.add_argument("--debug", action="store_true", help="デバッグモード有効化")
    args = parser.parse_args()

    # ロガーの設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(level=log_level)

    # パイプライン実行
    try:
        success = await run_full_pipeline(args, logger)
        if success:
            logger.info("パイプラインが正常に完了しました")
            return 0
        else:
            logger.error("パイプラインの実行に失敗しました")
            return 1
    except Exception as e:
        logger.error(f"パイプライン実行中に予期しないエラーが発生しました: {str(e)}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))