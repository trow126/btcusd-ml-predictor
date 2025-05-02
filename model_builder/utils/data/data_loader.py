# model_builder/utils/data/data_loader.py
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger("data_loader")

def load_data(input_dir: str, input_filename: str) -> pd.DataFrame:
    """
    特徴量データを読み込む（修正版）

    Args:
        input_dir: 入力ディレクトリパス
        input_filename: 入力ファイル名
        
    Returns:
        DataFrame: 読み込んだデータ
    """
    input_path = Path(input_dir) / input_filename
    logger.info(f"load_data: データを {input_path} から読み込みます")

    try:
        # まずファイルの存在確認
        if not input_path.exists():
            logger.error(f"ファイルが存在しません: {input_path}")
            return pd.DataFrame()
            
        # まず最初の行だけを読み込んでカラム名を調べる
        try:
            header_df = pd.read_csv(input_path, nrows=1)
            columns = header_df.columns.tolist()
            logger.info(f"ファイルのカラム: {columns[:5]}... (全{len(columns)}列)")
            
            # timestampカラムがあるか確認
            if 'timestamp' in columns:
                # timestampカラムがある場合はインデックスとして読み込み
                logger.info("timestampカラムをインデックスとして読み込みます")
                df = pd.read_csv(input_path, index_col="timestamp", parse_dates=True)
            else:
                # インデックス指定なしで読み込む
                logger.info("timestampカラムが見つかりません。インデックス指定なしで読み込みます")
                df = pd.read_csv(input_path)
                
                # 先頭カラムが日時っぽい場合は自動でインデックスにする
                first_col = columns[0]
                try:
                    # 最初の値を試しに日時変換してみる
                    sample_df = pd.read_csv(input_path, nrows=1)
                    pd.to_datetime(sample_df[first_col][0])
                    # 変換できたらインデックスに設定
                    logger.info(f"先頭カラム '{first_col}' を日時インデックスとして設定します")
                    df = pd.read_csv(input_path)
                    df.set_index(first_col, inplace=True)
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    # 日時に変換できない場合はそのまま
                    logger.info(f"先頭カラムを日時に変換できません。インデックスなしで読み込みます")
                    pass
                    
            logger.info(f"load_data: {len(df)} 行のデータを読み込みました")
            return df
        except Exception as e:
            # 上記方法で失敗した場合は、シンプルに読み込む
            logger.warning(f"カラム分析でエラーが発生しました。シンプルな読み込みを試みます: {e}")
            df = pd.read_csv(input_path)
            logger.info(f"load_data: シンプル読み込みで {len(df)} 行のデータを読み込みました")
            return df
    except Exception as e:
        logger.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame()