
"""
BTCUSD ML Predictorのデータ処理スクリプト
"""
import os
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_processing")

def process_data():
    """データ生成と前処理を実行"""
    try:
        # サンプルデータ生成
        logger.info("サンプルデータの生成を開始")
        import generate_sample_data
        logger.info("サンプルデータの生成が完了しました")
        
        # 特徴量エンジニアリングを実行
        logger.info("特徴量エンジニアリングを開始")
        from data_processor.optimized_feature_generator import generate_optimized_features
        
        # 設定ファイルから設定をロード
        from model_builder.utils.config_loader import load_json_config
        config_path = Path("config/model_config.json")
        config = load_json_config(str(config_path))
        
        # 特徴量生成実行
        feature_df, report = generate_optimized_features(config.get("data_processor", {}))
        
        logger.info(f"特徴量エンジニアリングが完了しました。生成された特徴量: {len(feature_df.columns)}列")
        logger.info(f"生成された行数: {len(feature_df)}行")
        
        # 目標変数の確認
        target_columns = [col for col in feature_df.columns if col.startswith("target_")]
        logger.info(f"生成された目標変数: {target_columns}")
        
        # ターゲット変数ごとの統計情報を表示
        for col in target_columns:
            if "binary" in col or "ternary" in col:
                value_counts = feature_df[col].value_counts()
                logger.info(f"{col} の値のカウント: {value_counts.to_dict()}")
        
        return True
    except Exception as e:
        logger.error(f"データ処理中にエラーが発生: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    process_data()
