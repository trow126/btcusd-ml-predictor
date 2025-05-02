
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
        # データ読み込みは通常通り実行
        logger.info("サンプルデータの生成が完了しました")
        
        # 設定ファイルから設定をロード
        from model_builder.utils.config_loader import load_json_config
        config_path = Path("config/model_config.json")
        config = load_json_config(str(config_path))
        
        # 高閾値シグナル設定を解析と追加確認
        if "high_threshold_models" not in config.get("data_processor", {}):
            logger.info("高閾値シグナル設定を追加します")
            if "data_processor" not in config:
                config["data_processor"] = {}
                
            config["data_processor"]["high_threshold_models"] = {
                "target_periods": [1, 2, 3],
                "directions": ["long", "short"],
                "thresholds": [0.001, 0.002, 0.003, 0.005]  # 0.1%, 0.2%, 0.3%, 0.5%
            }
        
        logger.info(f"高閾値シグナル設定: {config['data_processor']['high_threshold_models']}")
        
        # 設定をJSONで詳細に出力
        import json
        logger.info(f"使用する設定の全内容: {json.dumps(config['data_processor'], indent=2)}")
        
        # 特徴量生成用関数を直接ロードして設定を渡す
        from data_processor.optimized_feature_generator import OptimizedFeatureGenerator
        generator = OptimizedFeatureGenerator(config["data_processor"])
        
        # データ読み込み
        df = generator.load_data()
        
        # 特徴量生成 - 直接生成関数を実行
        feature_df = generator.generate_features(df)
        
        # 特徴量を保存
        generator.save_features(feature_df)
        
        # 必要に応じてレポートを生成
        report = generator.generate_feature_report(feature_df)
        
        logger.info(f"特徴量エンジニアリングが完了しました。生成された特徴量: {len(feature_df.columns)}列")
        logger.info(f"生成された行数: {len(feature_df)}行")
        
        # 目標変数の確認
        target_columns = [col for col in feature_df.columns if col.startswith("target_")]
        logger.info(f"生成された目標変数: {len(target_columns)}個")
        
        # 高閾値シグナル変数の確認
        high_threshold_cols = [col for col in feature_df.columns if 'high_threshold' in col]
        if high_threshold_cols:
            logger.info(f"高閾値シグナル変数: {len(high_threshold_cols)}個生成されました")
            for i, col in enumerate(high_threshold_cols[:10]):
                logger.info(f"  {i+1}. {col}")
            if len(high_threshold_cols) > 10:
                logger.info(f"  ...その他 {len(high_threshold_cols)-10} 個")
            
            # 上位数個のクラス分布を表示
            for col in high_threshold_cols[:5]:
                value_counts = feature_df[col].value_counts()
                logger.info(f"{col} の値のカウント: {value_counts.to_dict()}")
        else:
            logger.error("高閾値シグナル変数が生成されませんでした。target_features.pyの修正が反映されていない可能性があります。")
            
            # 設定が正しく渡されているか確認
            logger.info("DEBUG: 設定の確認")
            if "high_threshold_models" in config.get("data_processor", {}):
                logger.info(f"DEBUG: high_threshold_models設定は存在します: {config['data_processor']['high_threshold_models']}")
            else:
                logger.error("DEBUG: high_threshold_models設定が存在しません。設定が正しく渡されていない可能性があります。")
                
                # 設定を強制的に追加して再生成を試みる
                try:
                    logger.info("DEBUG: 直接特徴量生成を試みます")
                    from data_processor.feature_modules.target_features import generate_target_features
                    
                    # ハードコードした設定で試す
                    high_threshold_config = {
                        "thresholds": [0.001, 0.002, 0.003, 0.005],
                        "directions": ["long", "short"],
                        "target_periods": [1, 2, 3]
                    }
                    
                    # 特徴量DataFrame内の目標変数列を取得
                    target_change_cols = [col for col in feature_df.columns if col.startswith("target_price_change_pct_")]
                    if target_change_cols:
                        logger.info(f"DEBUG: 価格変動率列を検出: {target_change_cols}")
                        
                        # 高閾値シグナル変数を直接生成
                        target_features = generate_target_features(
                            feature_df,
                            [1, 2, 3],
                            0.0005,
                            high_threshold_config=high_threshold_config
                        )
                        
                        # 生成された変数を確認
                        high_threshold_cols = [col for col in target_features.columns if 'high_threshold' in col]
                        if high_threshold_cols:
                            logger.info(f"DEBUG: 直接生成に成功。変数: {len(high_threshold_cols)}個")
                            for i, col in enumerate(high_threshold_cols[:5]):
                                logger.info(f"  {i+1}. {col}")
                            
                            # 特徴量にマージ
                            feature_df = pd.concat([feature_df, target_features[high_threshold_cols]], axis=1)
                            logger.info(f"DEBUG: 特徴量に追加。新しい列数: {len(feature_df.columns)}")
                            
                            # 保存
                            output_path = Path("data/processed") / "btcusd_5m_features.csv"
                            feature_df.to_csv(output_path)
                            logger.info(f"DEBUG: 更新された特徴量を {output_path} に保存しました")
                        else:
                            logger.error("DEBUG: 直接生成も失敗しました。")
                    else:
                        logger.error("DEBUG: 価格変動率列が見つかりません。")
                except Exception as e:
                    logger.error(f"DEBUG: 直接生成中にエラー: {str(e)}")
            
            # どのような列が実際に生成されているか確認
            logger.info("DEBUG: 生成された列の一部:")
            target_cols = [col for col in feature_df.columns if col.startswith("target_")]
            for i, col in enumerate(target_cols[:20]):
                logger.info(f"  {i+1}. {col}")
            if len(target_cols) > 20:
                logger.info(f"  ...その他 {len(target_cols)-20} 個")
        
        # その他のターゲット変数の統計情報を表示
        standard_targets = [col for col in target_columns if 'high_threshold' not in col]
        logger.info(f"標準目標変数: {len(standard_targets)}個")
        for col in standard_targets[:5]:  # 最初の5個だけ表示
            if "binary" in col or "ternary" in col:
                value_counts = feature_df[col].value_counts()
                logger.info(f"{col} の値のカウント: {value_counts.to_dict()}")
        
        return True
    except Exception as e:
        logger.error(f"データ処理中にエラーが発生: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    process_data()
