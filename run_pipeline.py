
"""
BTCUSD ML Predictorのパイプライン実行スクリプト
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("btcusd_ml_predictor_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_pipeline")

async def run_pipeline():
    """修正したコードでパイプラインを実行"""
    try:
        logger.info("サンプルデータの生成を開始")
        # サンプルデータ生成
        import generate_sample_data
        logger.info("サンプルデータの生成が完了しました")
        
        # 必要なディレクトリが存在するか確認
        Path("models").mkdir(exist_ok=True)
        Path("evaluation").mkdir(exist_ok=True)
        
        # 高閾値シグナル変数を生成
        logger.info("高閾値シグナル変数の生成を開始")
        import add_high_threshold_signals
        success = add_high_threshold_signals.add_high_threshold_signals()
        if success:
            logger.info("高閾値シグナル変数の生成が完了しました")
        else:
            logger.error("高閾値シグナル変数の生成に失敗しました")
            # 失敗した場合は既存のスクリプトを試す
            logger.info("既存のスクリプトで高閾値シグナル変数の生成を試みます")
            import generate_high_threshold_signals
            success = generate_high_threshold_signals.generate_high_threshold_signals()
            if success:
                logger.info("既存スクリプトでの高閾値シグナル変数の生成が完了しました")
            else:
                logger.error("既存スクリプトでの高閾値シグナル変数生成に失敗しました")
                # 最後の手段として修正スクリプトを実行
                logger.info("修正スクリプトで高閾値シグナル変数の修正を試みます")
                import fix_high_threshold_signals
                fix_success = fix_high_threshold_signals.fix_signals()
                if fix_success:
                    logger.info("高閾値シグナル変数の修正が完了しました")
                else:
                    logger.error("高閾値シグナル変数の修正に失敗しました")
        
        # メインモジュールからパイプラインを実行
        logger.info("BTCUSDパイプラインの実行を開始")
        from main import main
        await main()
        logger.info("BTCUSDパイプラインの実行が完了しました")
        
        # 結果の確認
        if Path("evaluation/model_evaluation_report.json").exists():
            logger.info("評価レポートが正常に生成されました")
        else:
            logger.error("評価レポートが生成されませんでした")
            
    except Exception as e:
        logger.error(f"パイプライン実行中にエラーが発生: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_pipeline())
