# model_builder/utils/model_io/model_saver.py
import logging
import joblib
from pathlib import Path
from typing import Dict, Any

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
    output_path.mkdir(parents=True, exist_ok=True)

    # モデルの保存
    model_path = output_path / f"{name}.joblib"
    try:
        joblib.dump(model, model_path)
        logger.info(f"save_model: モデルを {model_path} に保存しました")
        logger.info(f"save_model: モデル '{name}' の保存を終了します")
        return True
    except Exception as e:
        logger.error(f"モデル保存エラー: {e}")
        return False