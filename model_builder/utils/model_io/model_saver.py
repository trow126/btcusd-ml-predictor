# model_builder/utils/model_io/model_saver.py
import logging
import joblib
from pathlib import Path
from typing import Dict, Any
import os

logger = logging.getLogger("model_saver")

def save_model(model: Any, output_dir: str, name: str) -> bool:
    """
    モデルを保存

    Args:
        model: 保存するモデル
        output_dir: 出力ディレクトリパス
        name: モデル名

    Returns:
        bool: 保存が成功したかどうか
    """
    logger.info(f"save_model: モデル '{name}' の保存を開始します")
    
    # 出力ディレクトリが存在しない場合は作成
    output_path = Path(output_dir)
    try:
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"出力ディレクトリを確認/作成しました: {output_path}")
    except Exception as e:
        logger.error(f"ディレクトリ作成エラー: {e}")
        return False

    # モデルの保存
    model_path = output_path / name
    
    # 対象ファイル名が既に.joblib拡張子を持つか確認
    if not str(model_path).lower().endswith('.joblib'):
        model_path = Path(str(model_path) + '.joblib')
    
    try:
        joblib.dump(model, model_path)
        logger.info(f"save_model: モデルを {model_path} に保存しました")
        
        # 保存されたファイルサイズをチェック
        file_size = os.path.getsize(model_path)
        logger.info(f"保存されたモデルのファイルサイズ: {file_size/1024:.2f} KB")
        
        # ファイルが正常に保存されたか確認
        if os.path.exists(model_path):
            logger.info(f"ファイルの存在を確認: {model_path} は正常に保存されました")
        else:
            logger.error(f"ファイルの存在を確認できません: {model_path}")
            return False
            
        logger.info(f"save_model: モデル '{name}' の保存を終了します")
        return True
    except Exception as e:
        logger.error(f"モデル保存エラー: {e}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return False