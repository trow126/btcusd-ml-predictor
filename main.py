import asyncio
import logging
from pathlib import Path

from data_collector.bybit_collector import BTCDataCollector
from data_processor.optimized_feature_generator import OptimizedFeatureGenerator
from model_builder.model_trainer import ModelTrainer
from model_builder.model_evaluator import ModelEvaluator
from model_builder.utils import load_json_config

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

async def main():
    logger.info("BTCUSD ML Predictor パイプラインを開始します")

    # 設定のロード
    config_path = Path(__file__).parent / "config" / "data_config.json"
    config = load_json_config(str(config_path))
    logger.info("設定ファイルをロードしました")

    # 1. データ収集
    logger.info("データ収集を開始します")
    data_collector = BTCDataCollector(config.get("data_collector"))
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
                report = model_trainer.generate_training_report(regression_results, classification_results)
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
