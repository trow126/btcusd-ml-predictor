import asyncio
import logging
from pathlib import Path

from data_collector.bybit_collector import BTCDataCollector
from data_processor.optimized_feature_generator import OptimizedFeatureGenerator
from model_builder.trainers import ModelTrainer
from model_builder.evaluators import ModelEvaluator
from model_builder.utils.config_loader import load_json_config

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

async def main():
    logger.info("BTCUSD ML Predictor パイプラインを開始します")

    # 設定のロード
    data_config_path = Path(__file__).parent / "config" / "data_config.json"
    model_config_path = Path(__file__).parent / "config" / "model_config.json"
    
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

    # 1. データ収集
    logger.info("データ収集を開始します")
    
    # 日付変換処理
    import datetime as dt
    data_collector_config = config.get("data_collector", {})
    
    # start_dateとend_dateが文字列の場合はdatetimeオブジェクトに変換
    if isinstance(data_collector_config.get("start_date"), str):
        start_date_str = data_collector_config.get("start_date")
        try:
            data_collector_config["start_date"] = dt.datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
        except:
            # エラーが発生した場合はデフォルト値を使用
            data_collector_config["start_date"] = dt.datetime(2023, 1, 1)
    
    if isinstance(data_collector_config.get("end_date"), str) and data_collector_config.get("end_date"):
        end_date_str = data_collector_config.get("end_date")
        try:
            data_collector_config["end_date"] = dt.datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except:
            # エラーが発生した場合は現在時刻を使用
            data_collector_config["end_date"] = dt.datetime.now()
    elif data_collector_config.get("end_date") is None:
        data_collector_config["end_date"] = dt.datetime.now()
    
    data_collector = BTCDataCollector(data_collector_config)
    historical_data = await data_collector.collect_historical_data()
    if not historical_data.empty:
        data_collector.save_data(historical_data)
        logger.info("データ収集と保存が完了しました")
    else:
        logger.error("データ収集に失敗しました。パイプラインを中断します。")
        return

    # 2. 特徴量エンジニアリング
    logger.info("特徴量エンジニアリングを開始します")
    feature_generator = OptimizedFeatureGenerator(config.get("data_processor"))
    features_df = feature_generator.generate_features(historical_data)
    if not features_df.empty:
        feature_generator.save_features(features_df)
        logger.info("特徴量エンジニアリングと保存が完了しました")
    else:
        logger.error("特徴量エンジニアリングに失敗しました。パイプラインを中断します。")
        return

    # 3. モデル訓練
    logger.info("モデル訓練を開始します")
    model_trainer = ModelTrainer(config.get("model_trainer"))
    # 特徴量データを再ロード（特徴量エンジニアリングでNaN行が削除されている可能性があるため）
    features_df_for_training = model_trainer.load_data()
    if not features_df_for_training.empty:
        X, y_dict = model_trainer.prepare_features(features_df_for_training)
        if X and y_dict:
            X_train, X_test, y_train, y_test = model_trainer.train_test_split(X["X"], y_dict)
            if X_train and X_test and y_train and y_test:
                regression_results = model_trainer.train_regression_models(
                    {"X": X_train["X"]}, {"X": X_test["X"]}, y_train, y_test
                )
                classification_results = model_trainer.train_classification_models(
                    {"X": X_train["X"]}, {"X": X_test["X"]}, y_train, y_test
                )
                
                # 二値分類モデル（上昇/下落予測）のトレーニング
                binary_classification_results = None
                if config.get("model_trainer", {}).get("use_binary_classification", False):
                    binary_classification_results = model_trainer.train_binary_classification_models(
                        {"X": X_train["X"]}, {"X": X_test["X"]}, y_train, y_test
                    )
                
                # 結果を辞書にまとめる
                all_results = {
                    "regression": regression_results,
                    "classification": classification_results
                }
                
                if binary_classification_results:
                    all_results["binary_classification"] = binary_classification_results
                    
                report = model_trainer.generate_training_report(
                    regression_results,
                    classification_results,
                    binary_classification_results
                )
                # 訓練レポートの保存はModelTrainer内で行われる想定
                logger.info("モデル訓練が完了しました")
            else:
                 logger.error("訓練/テストデータの分割に失敗しました。パイプラインを中断します。")
                 return
        else:
            logger.error("特徴量と目標変数の準備に失敗しました。パイプラインを中断します。")
            return
    else:
        logger.error("訓練用の特徴量データのロードに失敗しました。パイプラインを中断します。")
        return


    # 4. モデル評価
    logger.info("モデル評価を開始します")
    model_evaluator = ModelEvaluator(config.get("model_evaluator"))
    # 評価用の特徴量データを再ロード
    features_df_for_evaluation = model_evaluator.load_data()
    if not features_df_for_evaluation.empty:
        if model_evaluator.load_models():
            X_test_eval, y_test_eval = model_evaluator.prepare_test_data(features_df_for_evaluation)
            if not X_test_eval.empty and y_test_eval:
                evaluation_results = model_evaluator.evaluate_models(X_test_eval, y_test_eval)
                evaluation_report = model_evaluator.generate_evaluation_report(evaluation_results)
                model_evaluator.save_evaluation_report(evaluation_report)
                logger.info("モデル評価とレポート保存が完了しました")
            else:
                logger.error("評価用テストデータの準備に失敗しました。パイプラインを中断します。")
                return
        else:
            logger.error("モデルのロードに失敗しました。パイプラインを中断します。")
            return
    else:
        logger.error("評価用の特徴量データのロードに失敗しました。パイプラインを中断します。")
        return


    logger.info("BTCUSD ML Predictor パイプラインが完了しました")


if __name__ == "__main__":
    asyncio.run(main())
