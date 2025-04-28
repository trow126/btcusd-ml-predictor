# model_builder/utils/model_io/model_loader.py
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger("model_loader")

def load_model(model_path: str) -> Any:
    """
    モデルをロードする

    Args:
        model_path: モデルファイルのパス

    Returns:
        Any: ロードされたモデル、またはNone（失敗時）
    """
    try:
        logger.info(f"モデルをロード中: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"モデルを {model_path} から読み込みました")

        # モデルの詳細情報を表示
        if model:
            try:
                # モデルの特徴量名を取得
                feature_names = _get_model_feature_names(model)
                if feature_names:
                    logger.info(f"モデルの特徴量数: {len(feature_names)}")
                    # 特徴量名の一部を表示
                    if len(feature_names) > 0:
                        sample_features = feature_names[:min(5, len(feature_names))]
                        logger.info(f"特徴量サンプル: {sample_features}...")
                else:
                    logger.warning("モデルから特徴量名を取得できません")

                # モデルタイプの確認
                model_type = type(model).__name__
                logger.info(f"モデルタイプ: {model_type}")
            except Exception as e:
                logger.warning(f"モデル情報の取得中にエラーが発生しました: {str(e)}")

        return model
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {str(e)}")
        return None

def _get_model_feature_names(model: Any) -> List[str]:
    """
    モデルの特徴量名を取得

    Args:
        model: 特徴量名を取得するモデル

    Returns:
        List[str]: 特徴量名のリスト、または空リスト（取得失敗時）
    """
    feature_names = []

    # LightGBMモデルからの特徴量名取得を試みる
    try:
        if hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
        elif hasattr(model, 'feature_name'):
            feature_names = model.feature_name()
        elif hasattr(model, 'booster_'):
            if hasattr(model.booster_, 'feature_name'):
                feature_names = model.booster_.feature_name()
            elif hasattr(model.booster_, 'feature_names'):
                feature_names = model.booster_.feature_names
    except Exception as e:
        logger.warning(f"特徴量名の取得中にエラーが発生: {str(e)}")

    return feature_names

def load_models(model_dir: str, target_periods: list) -> Dict[str, Any]:
    """
    複数のモデルをロードする

    Args:
        model_dir: モデルディレクトリのパス
        target_periods: 予測対象期間のリスト

    Returns:
        Dict[str, Any]: ロードされたモデルの辞書
    """
    logger.info("=== load_models: モデルのロードを開始します ===")
    models = {}
    model_dir_path = Path(model_dir)

    # ディレクトリが存在しない場合は作成
    if not model_dir_path.exists():
        logger.error(f"load_models: モデルディレクトリ {model_dir_path} が存在しません")
        os.makedirs(model_dir_path, exist_ok=True)
        logger.info(f"モデルディレクトリ {model_dir_path} を作成しました")
        return models

    # 利用可能なモデルファイルを一覧表示
    try:
        available_models = list(model_dir_path.glob("*.joblib"))
        logger.info(f"利用可能なモデルファイル: {[m.name for m in available_models]}")
    except Exception as e:
        logger.error(f"モデルファイル検索エラー: {e}")
        available_models = []

    # モデルタイプごとにロード
    _load_model_by_type(models, model_dir_path, "regression_model_period", target_periods, "regression")
    _load_model_by_type(models, model_dir_path, "classification_model_period", target_periods, "classification")
    _load_model_by_type(models, model_dir_path, "binary_classification_model_period", target_periods, "binary_classification")

    # 閾値ベースの二値分類モデルは特別処理（ファイル名の問題に対応）
    _load_threshold_binary_models(models, model_dir_path, target_periods)

    logger.info(f"=== load_models: {len(models)} 個のモデルを読み込みました ===")
    if len(models) > 0:
        logger.info(f"読み込まれたモデルの種類: {list(models.keys())}")
    logger.info("load_models: モデルのロードを終了します")
    return models

def _load_model_by_type(models: Dict[str, Any], model_dir: Path, file_prefix: str,
                        target_periods: list, model_type: str) -> None:
    """
    特定のタイプのモデルをロード

    Args:
        models: モデルを格納する辞書（更新される）
        model_dir: モデルディレクトリのパス
        file_prefix: モデルファイル名のプレフィックス
        target_periods: 予測対象期間のリスト
        model_type: モデルタイプの識別子
    """
    for period in target_periods:
        model_path = model_dir / f"{file_prefix}_{period}.joblib"
        if model_path.exists():
            model = load_model(str(model_path))
            if model:
                models[f"{model_type}_{period}"] = model
                logger.info(f"{model_type}モデル（{period}期先）をロードしました")
        else:
            logger.warning(f"{model_type}モデル {model_path.name} が見つかりません")

def _load_threshold_binary_models(models: Dict[str, Any], model_dir: Path, target_periods: list) -> None:
    """
    閾値ベースの二値分類モデルをロード（ファイル名のバリエーションに対応）

    Args:
        models: モデルを格納する辞書（更新される）
        model_dir: モデルディレクトリのパス
        target_periods: 予測対象期間のリスト
    """
    model_type = "threshold_binary_classification"

    for period in target_periods:
        # 標準パターンでの検索
        model_path = model_dir / f"{model_type}_model_period_{period}.joblib"

        # ファイルが見つからない場合はワイルドカードパターンで検索
        if not model_path.exists():
            pattern = f"{model_type}_model_period_{period}*"
            model_files = list(model_dir.glob(pattern))
            if model_files:
                logger.info(f"代替パターンでモデルを検索: {[f.name for f in model_files]}")
                model_path = model_files[0]  # 最初に見つかったファイルを使用

        # モデルのロード
        if model_path.exists():
            model = load_model(str(model_path))
            if model:
                models[f"{model_type}_{period}"] = model
                logger.info(f"{model_type}モデル（{period}期先）をロードしました")
        else:
            logger.warning(f"{model_type}モデル {period}期先 が見つかりません")