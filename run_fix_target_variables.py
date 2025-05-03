#!/usr/bin/env python
"""
目標変数修正スクリプトを実行してから、BTCUSDパイプラインを再実行するスクリプト
"""
import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fix_and_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fix_and_run")

async def run_fix_and_pipeline():
    """目標変数を修正してからパイプラインを実行"""
    try:
        # サンプルデータ（まだ生成されていなければ）
        features_path = Path("data/processed/btcusd_5m_features.csv")
        if not features_path.exists():
            logger.info("特徴量ファイルが見つかりません。サンプルデータを生成します。")
            import generate_sample_data
            logger.info("サンプルデータの生成が完了しました")
            
            # データ処理を実行
            logger.info("データ処理を実行します")
            import run_data_processing
            run_data_processing.process_data()
            logger.info("データ処理が完了しました")
        
        # 目標変数を修正
        logger.info("目標変数の修正を開始します")
        import fix_target_variables
        fix_success = fix_target_variables.fix_target_variables()
        
        if fix_success:
            logger.info("目標変数の修正が成功しました")
            
            # 修正結果を確認
            import pandas as pd
            df = pd.read_csv(features_path)
            
            # 目標変数を確認
            target_cols = [col for col in df.columns if col.startswith("target_")]
            logger.info(f"修正後の目標変数: {len(target_cols)}個")
            
            # target_binary_* 変数を確認
            binary_cols = [col for col in target_cols if "target_binary_" in col]
            logger.info(f"target_binary_* 変数: {len(binary_cols)}個")
            for col in binary_cols:
                value_counts = df[col].value_counts()
                logger.info(f"{col} の値分布: {value_counts.to_dict()}")
            
            # target_smoothed_change_* 変数を確認
            smoothed_cols = [col for col in target_cols if "target_smoothed_change_" in col]
            logger.info(f"target_smoothed_change_* 変数: {len(smoothed_cols)}個")
            for col in smoothed_cols:
                logger.info(f"{col} の統計: 平均={df[col].mean():.6f}, 最小={df[col].min():.6f}, 最大={df[col].max():.6f}")
            
            # high_threshold 変数を確認
            high_threshold_cols = [col for col in target_cols if "high_threshold" in col]
            logger.info(f"high_threshold_* 変数: {len(high_threshold_cols)}個")
            if high_threshold_cols:
                for col in high_threshold_cols[:5]:  # 最初の5つだけ表示
                    signal_count = df[col].sum()
                    signal_ratio = signal_count / len(df) * 100
                    logger.info(f"{col}: シグナル数={signal_count}個 ({signal_ratio:.2f}%)")
            
            # BTCUSDパイプラインの実行（データ収集と特徴量生成をスキップ）
            logger.info("BTCUSDパイプラインの実行を開始します (データ収集と特徴量生成をスキップ)")
            
            # コマンドライン引数オブジェクトを作成（データ収集と特徴量生成をスキップ）
            from main import run_full_pipeline
            
            # 引数オブジェクトを作成
            args = argparse.Namespace()
            args.skip_data_collection = True
            args.skip_feature_generation = True
            args.skip_training = False
            args.skip_evaluation = False
            args.continue_on_error = True
            args.debug = True
            
            # 高閾値シグナル関連のオプションも追加（run_full_pipelineでこれらが使われる可能性がある）
            args.skip_high_threshold_training = False
            args.skip_high_threshold_evaluation = False
            
            # 修正済みデータを使ってパイプラインを実行
            await run_full_pipeline(args, logger)
            logger.info("BTCUSDパイプラインの実行が完了しました")
            
            # 結果の確認
            if Path("evaluation/model_evaluation_report.json").exists():
                logger.info("評価レポートが正常に生成されました")
            else:
                logger.warning("評価レポートが生成されませんでした")
        else:
            logger.error("目標変数の修正に失敗しました")
            return False
            
        return True
    except Exception as e:
        logger.error(f"実行中にエラーが発生: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(run_fix_and_pipeline())
