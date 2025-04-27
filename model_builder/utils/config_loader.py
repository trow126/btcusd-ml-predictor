# model_builder/utils/config_loader.py
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("config_loader")

def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    JSONファイルから設定を読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        Dict: 設定辞書
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"設定を {config_path} から読み込みました")
        return config
    except Exception as e:
        logger.error(f"設定読み込みエラー: {e}")
        return {}