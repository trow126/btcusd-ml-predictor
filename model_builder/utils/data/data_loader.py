# model_builder/utils/data/data_loader.py
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger("data_loader")

def load_data(input_dir: str, input_filename: str) -> pd.DataFrame:
    """
    特徴量データを読み込む

    Args:
        input_dir: 入力ディレクトリパス
        input_filename: 入力ファイル名
        
    Returns:
        DataFrame: 読み込んだデータ
    """
    input_path = Path(input_dir) / input_filename
    logger.info(f"load_data: データを {input_path} から読み込みます")

    try:
        df = pd.read_csv(input_path, index_col="timestamp", parse_dates=True)
        logger.info(f"load_data: {len(df)} 行のデータを読み込みました")
        return df
    except Exception as e:
        logger.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame()