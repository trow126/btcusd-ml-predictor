# model_builder/utils/model_io/model_loader.py
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("model_loader")

def load_model(model_path: str) -> Any:
    """
    モデルを読み込む

    Args:
        model_path: モデルファイルのパス

    Returns:
        Any: 読み込んだモデル
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"モデルを {model_path} から読み込みました")
        return model
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
        return None

def load_models(model_dir: str, target_periods: list) -> Dict[str, Any]:
    """
    複数のモデルを読み込む

    Args:
        model_dir: モデルディレクトリのパス
        target_periods: 予測対象期間のリスト

    Returns:
        Dict[str, Any]: 読み込んだモデルの辞書
    """
    logger.info("load_models: モデルの読み込みを開始します")
    models = {}
    model_dir_path = Path(model_dir)

    if not model_dir_path.exists():
        logger.error(f"load_models: モデルディレクトリ {model_dir_path} が存在しません")
        return models

    # 回帰モデルの読み込み
    for period in target_periods:
        regression_model_path = model_dir_path / f"regression_model_period_{period}.joblib"
        if regression_model_path.exists():
            model = load_model(regression_model_path)
            if model:
                models[f"regression_{period}"] = model
                logger.info(f"回帰モデル（{period}期先）を読み込みました")
        else:
            logger.warning(f"回帰モデル {regression_model_path} が見つかりません")

    # 分類モデルの読み込み
    for period in target_periods:
        classification_model_path = model_dir_path / f"classification_model_period_{period}.joblib"
        if classification_model_path.exists():
            model = load_model(classification_model_path)
            if model:
                models[f"classification_{period}"] = model
                logger.info(f"分類モデル（{period}期先）を読み込みました")
        else:
            logger.warning(f"分類モデル {classification_model_path} が見つかりません")

    logger.info(f"load_models: {len(models)} 個のモデルを読み込みました")
    logger.info("load_models: モデルの読み込みを終了します")
    return models