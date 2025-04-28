# debug_tools/train_threshold_model.py
"""
閾値ベースの二値分類モデルを単体でトレーニングするスクリプト
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import datetime as dt

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# モジュールのインポート
from data_processor.optimized_feature_generator import OptimizedFeatureGenerator
from model_builder.trainers import ModelTrainer
from model_builder.evaluators import ModelEvaluator
from model_builder.utils.config_loader import load_json_config

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("threshold_binary_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("threshold_trainer")

def inspect_features_file(file_path):
    """特徴量ファイルを検査"""
    logger.info(f"特徴量ファイルの検査: {file_path}")
    
    try:
        # ファイルが存在するか確認
        if not Path(file_path).exists():
            logger.error(f"ファイルが見つかりません: {file_path}")
            return False
            
        # ファイルサイズを確認
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MBに変換
        logger.info(f"ファイルサイズ: {file_size:.2f} MB")
        
        # CSVヘッダーだけを読み込んでカラム名を確認
        df_headers = pd.read_csv(file_path, nrows=0)
        columns = df_headers.columns.tolist()
        logger.info(f"カラム数: {len(columns)}")
        
        # 目標変数カラムがあるか確認
        threshold_columns = [col for col in columns if "threshold_binary" in col]
        if threshold_columns:
            logger.info(f"閾値ベース二値分類の目標変数が見つかりました: {threshold_columns}")
        else:
            logger.warning("閾値ベース二値分類の目標変数が見つかりません")
            logger.info("利用可能な目標変数: " + ", ".join([col for col in columns if "target_" in col]))
            
        # 一部のデータのみを読み込んで統計情報を確認
        logger.info("データサンプルを読み込んでいます...")
        df_sample = pd.read_csv(file_path, index_col="timestamp", parse_dates=True, nrows=100000)
        
        # 目標変数の確認
        for col in threshold_columns:
            if col in df_sample.columns:
                # NaNの数
                nan_count = df_sample[col].isna().sum()
                valid_count = len(df_sample[col].dropna())
                
                logger.info(f"{col}: 有効データ {valid_count}行, NaN {nan_count}行")
                
                # クラスバランス
                class_dist = df_sample[col].value_counts(dropna=True)
                logger.info(f"{col} のクラス分布: {class_dist.to_dict()}")
        
        return True
    except Exception as e:
        logger.error(f"ファイル検査中にエラーが発生しました: {e}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return False

def generate_threshold_targets(input_file, output_file):
    """閾値ベースの目標変数を生成または更新"""
    logger.info("閾値ベースの目標変数を生成します")
    
    try:
        # データを読み込む
        df = pd.read_csv(input_file, index_col="timestamp", parse_dates=True)
        logger.info(f"データ読み込み完了: {len(df)}行, {len(df.columns)}列")
        
        # 目標変数が既にあるか確認
        existing_threshold_cols = [col for col in df.columns if "target_threshold_binary" in col]
        if existing_threshold_cols:
            logger.info(f"既存の閾値ベース目標変数: {existing_threshold_cols}")
        
        # 価格変動率列があるか確認
        price_change_cols = [col for col in df.columns if "target_price_change_pct" in col]
        if not price_change_cols:
            logger.error("価格変動率の目標変数が見つかりません。目標変数を生成できません。")
            return False
            
        logger.info(f"価格変動率の目標変数: {price_change_cols}")
        
        # 閾値を設定
        classification_threshold = 0.0005  # 0.05%
        logger.info(f"分類閾値: {classification_threshold}")
        
        # 閾値ベースの目標変数を生成
        for col in price_change_cols:
            period = col.split('_')[-1]  # 期間を抽出
            target_name = f'target_threshold_binary_{period}'
            
            # 目標変数を生成（上昇:1、下落:0、横ばい:NaN）
            df[target_name] = np.where(
                df[col] >= classification_threshold,
                1,  # 上昇
                np.where(
                    df[col] <= -classification_threshold,
                    0,  # 下落
                    np.nan  # 横ばい
                )
            )
            
            # 統計情報を表示
            valid_count = df[target_name].dropna().count()
            nan_count = df[target_name].isna().sum()
            
            logger.info(f"{target_name} を生成しました: 有効データ {valid_count}行 ({valid_count/len(df)*100:.2f}%), NaN {nan_count}行")
            
            # クラスバランス
            class_dist = df[target_name].value_counts(dropna=True)
            logger.info(f"{target_name} のクラス分布: {class_dist.to_dict()}")
        
        # 修正したデータを保存
        df.to_csv(output_file)
        logger.info(f"修正したデータを保存しました: {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"目標変数生成中にエラーが発生しました: {e}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return False

def train_threshold_models():
    """閾値ベースの二値分類モデルをトレーニング"""
    logger.info("閾値ベースの二値分類モデルのトレーニングを開始します")
    
    # 設定のロード
    config_path = project_root / "config" / "model_config.json"
    model_config = load_json_config(str(config_path))
    
    # 設定を修正して閾値ベース二値分類に特化
    trainer_config = model_config.get("model_trainer", {})
    trainer_config["use_threshold_binary_classification"] = True  # 明示的に有効化
    
    # モデルトレーナーの初期化
    trainer = ModelTrainer(trainer_config)
    
    # データ読み込み
    features_df = trainer.load_data()
    if features_df.empty:
        logger.error("特徴量データのロードに失敗しました")
        return False
    
    logger.info(f"特徴量データをロードしました: {len(features_df)}行, {len(features_df.columns)}列")
    
    # 目標変数の確認
    target_cols = [col for col in features_df.columns if col.startswith("target_")]
    threshold_binary_cols = [col for col in target_cols if "threshold_binary" in col]
    logger.info(f"閾値ベース二値分類の目標変数: {threshold_binary_cols}")
    
    # 特徴量と目標変数の準備
    X_dict, y_dict = trainer.prepare_features(features_df)
    
    # 閾値ベース二値分類の目標変数があるか確認
    threshold_target_keys = [key for key in y_dict.keys() if "threshold_binary_classification" in key]
    if not threshold_target_keys:
        logger.error("閾値ベース二値分類の目標変数が見つかりません")
        logger.info(f"利用可能な目標変数: {list(y_dict.keys())}")
        return False
    
    logger.info(f"閾値ベース二値分類の目標変数: {threshold_target_keys}")
    
    # 各目標変数の統計情報
    for key in threshold_target_keys:
        valid_samples = y_dict[key].dropna()
        class_balance = valid_samples.value_counts()
        logger.info(f"{key} の有効サンプル数: {len(valid_samples)}, クラスバランス: {class_balance.to_dict()}")
    
    # トレーニングデータとテストデータに分割
    X_train, X_test, y_train, y_test = trainer.train_test_split(X_dict["X"], y_dict)
    
    # 特殊な特徴量セットがある場合は分割
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
    
    # 閾値ベースの二値分類モデルだけをトレーニング
    logger.info("閾値ベースの二値分類モデルのトレーニングを開始します")
    threshold_results = trainer.train_threshold_binary_classification_models(X_train, X_test, y_train, y_test)
    
    # 結果のサマリーを出力
    success_count = 0
    error_count = 0
    for key, result in threshold_results.items():
        if isinstance(result, dict) and "error" in result:
            error_count += 1
            logger.error(f"{key} でエラー: {result['error']} - {result['message']}")
        else:
            success_count += 1
            logger.info(f"{key} のトレーニング成功")
            
            if hasattr(result, 'get'):
                metrics = {
                    'accuracy': result.get('accuracy', 'N/A'),
                    'precision': result.get('precision', 'N/A'),
                    'recall': result.get('recall', 'N/A'),
                    'f1_score': result.get('f1_score', 'N/A')
                }
                logger.info(f"{key} の評価指標: {metrics}")
    
    logger.info(f"閾値ベース分類モデルのトレーニング結果: 成功 {success_count}, 失敗 {error_count}")
    
    if success_count > 0:
        logger.info("モデルのトレーニングが完了しました")
        
        # モデルの評価 - 評価も行う
        evaluator_config = model_config.get("model_evaluator", {})
        evaluator = ModelEvaluator(evaluator_config)
        
        # モデルのロード
        if evaluator.load_models():
            # テストデータの準備
            X_test_dict, y_test_dict = evaluator.prepare_test_data(features_df)
            
            # モデルの評価
            evaluation_results = evaluator.evaluate_models(X_test_dict, y_test_dict)
            
            # 閾値ベースの二値分類の結果を表示
            if "threshold_binary_classification" in evaluation_results:
                threshold_eval = evaluation_results["threshold_binary_classification"]
                logger.info(f"閾値ベース二値分類モデルの評価結果: {threshold_eval}")
                
                # 評価レポートの生成と保存
                report = evaluator.generate_evaluation_report(evaluation_results)
                evaluator.save_evaluation_report(report)
                logger.info("評価レポートを保存しました")
            else:
                logger.warning("閾値ベース二値分類モデルの評価結果がありません")
        else:
            logger.error("モデルのロードに失敗しました")
        
        return True
    else:
        logger.error("閾値ベース分類モデルのトレーニングに失敗しました")
        return False

def main():
    logger.info("=== 閾値ベース二値分類モデルのデバッグ・トレーニングスクリプトを開始 ===")
    
    # パスの設定
    data_dir = project_root / "data" / "processed"
    input_file = data_dir / "btcusd_5m_features.csv"
    fixed_file = data_dir / "btcusd_5m_features_fixed.csv"
    
    # ファイルの検査
    if not inspect_features_file(input_file):
        logger.error("特徴量ファイルの検査に失敗しました")
        return
    
    # 閾値ベースの目標変数を生成（必要な場合）
    generate_threshold_targets(input_file, fixed_file)
    
    # 閾値ベースの二値分類モデルをトレーニング
    train_threshold_models()
    
    logger.info("=== スクリプトが完了しました ===")

if __name__ == "__main__":
    main()
