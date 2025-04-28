import asyncio
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from data_collector.bybit_collector import BTCDataCollector
from data_processor.optimized_feature_generator import OptimizedFeatureGenerator
from model_builder.trainers import ModelTrainer
from model_builder.evaluators import ModelEvaluator
from model_builder.utils.config_loader import load_json_config

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,  # DEBUGレベルに変更してより詳細なログを出力
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("btcusd_ml_predictor.log"),  # ログファイルに出力
        logging.StreamHandler()  # コンソールにも出力
    ]
)
logger = logging.getLogger("main")

def check_threshold_binary_targets(features_df):
    """
    閾値ベースの二値分類ターゲットがデータに存在することを確認し、
    情報をログに出力します。
    """
    logger.info("閾値ベースの二値分類ターゲット変数を確認します")
    
    # 閾値ベースの二値分類ターゲットがあるか確認
    threshold_binary_cols = [col for col in features_df.columns if "target_threshold_binary" in col]
    
    if threshold_binary_cols:
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
    else:
        logger.warning("閾値ベースの二値分類ターゲットが見つかりません")
    
    return features_df

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

    # 閾値ベースの二値分類ターゲットを確認
    features_df = check_threshold_binary_targets(features_df)

    # 3. モデル訓練
    logger.info("モデル訓練を開始します")
    model_trainer = ModelTrainer(config.get("model_trainer"))
    # 特徴量データを再ロード（特徴量エンジニアリングでNaN行が削除されている可能性があるため）
    features_df_for_training = model_trainer.load_data()
    
    if not features_df_for_training.empty:
        # 閾値ベースの二値分類ターゲットを確認
        features_df_for_training = check_threshold_binary_targets(features_df_for_training)
        
        # 特徴量と目標変数の準備（返り値はX_dictとy_dict）
        X_dict, y_dict = model_trainer.prepare_features(features_df_for_training)
        if X_dict and y_dict:
            # 閾値ベースの二値分類ターゲットがあるか確認
            threshold_keys = [key for key in y_dict.keys() if "threshold_binary_classification" in key]
            if not threshold_keys:
                logger.warning("閾値ベースの二値分類ターゲットが準備されていません。")
                # 利用可能なターゲットを表示
                logger.info(f"利用可能なターゲット: {list(y_dict.keys())}")
            else:
                logger.info(f"閾値ベースの二値分類ターゲット: {threshold_keys}")
            
            # 全体のデータをトレーニングとテストに分割
            X_train, X_test, y_train, y_test = model_trainer.train_test_split(X_dict["X"], y_dict)
            
            if X_train and X_test and y_train and y_test:
                # X_dictから特殊なX_keyに対応するデータを追加
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
                        
                        logger.info(f"特殊特徴量セット '{key}' をトレーニング({len(X_train[key])}行)とテスト({len(X_test[key])}行)に分割しました")
                
                # 回帰モデル（価格変動率予測）のトレーニング
                regression_results = model_trainer.train_regression_models(
                    X_train, X_test, y_train, y_test
                )
                
                # 分類モデル（価格変動方向予測）のトレーニング
                classification_results = model_trainer.train_classification_models(
                    X_train, X_test, y_train, y_test
                )
                
                # 二値分類モデル（上昇/下落予測）のトレーニング
                binary_classification_results = None
                if config.get("model_trainer", {}).get("use_binary_classification", False):
                    binary_classification_results = model_trainer.train_binary_classification_models(
                        X_train, X_test, y_train, y_test
                    )
                
                # 閾値ベースの二値分類モデル（有意な上昇/下落予測）のトレーニング
                threshold_binary_classification_results = None
                if config.get("model_trainer", {}).get("use_threshold_binary_classification", True):
                    logger.info("====== 閾値ベース分類モデルのトレーニングを開始します ======")
                    logger.info(f"ターゲット変数の確認: {list(y_train.keys())}")
                    for period in model_trainer.config.get("target_periods", [1, 2, 3]):
                        target_key = f"threshold_binary_classification_{period}"
                        logger.info(f"確認: {target_key} が存在するか? {target_key in y_train}")
                        if target_key in y_train:
                            # 値の分布を確認
                            valid_values = y_train[target_key].dropna()
                            logger.info(f"{target_key} の有効値数: {len(valid_values)}")
                            logger.info(f"{target_key} の値のカウント: {valid_values.value_counts().to_dict()}")
                    
                    threshold_binary_classification_results = model_trainer.train_threshold_binary_classification_models(
                        X_train, X_test, y_train, y_test
                    )
                    
                    # 結果のサマリーを出力
                    if threshold_binary_classification_results:
                        success_count = 0
                        error_count = 0
                        for key, result in threshold_binary_classification_results.items():
                            if isinstance(result, dict) and "error" in result:
                                error_count += 1
                                logger.error(f"{key} でエラー: {result['error']} - {result.get('message', 'No message')}")
                            else:
                                success_count += 1
                                logger.info(f"{key} のトレーニング成功")
                        
                        logger.info(f"閾値ベース分類モデルのトレーニング結果: 成功 {success_count}, 失敗 {error_count}")
                
                # 結果を辞書にまとめる
                all_results = {
                    "regression": regression_results,
                    "classification": classification_results
                }
                
                if binary_classification_results:
                    all_results["binary_classification"] = binary_classification_results
                    
                if threshold_binary_classification_results:
                    all_results["threshold_binary_classification"] = threshold_binary_classification_results
                    
                report = model_trainer.generate_training_report(
                    regression_results,
                    classification_results,
                    binary_classification_results,
                    threshold_binary_classification_results
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
        # 閾値ベースの二値分類ターゲットを確認
        features_df_for_evaluation = check_threshold_binary_targets(features_df_for_evaluation)
        
        if model_evaluator.load_models():
            X_test_eval_dict, y_test_eval = model_evaluator.prepare_test_data(features_df_for_evaluation)
            # 辞書型チェックを修正
            if X_test_eval_dict and "X" in X_test_eval_dict and not X_test_eval_dict["X"].empty and y_test_eval:
                evaluation_results = model_evaluator.evaluate_models(X_test_eval_dict, y_test_eval)
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