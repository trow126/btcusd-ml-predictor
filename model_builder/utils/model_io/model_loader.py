# model_builder/utils/model_io/model_loader.py
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import os

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
        
        # モデルの詳細情報を表示
        if model:
            try:
                # モデルの特徴量名を取得
                feature_names = None
                if hasattr(model, 'feature_name_'):
                    feature_names = model.feature_name_
                    logger.info(f"モデルの特徴量名: {feature_names[:5]}... (全{len(feature_names)}個)")
                elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
                    feature_names = model.booster_.feature_name()
                    logger.info(f"booster_から特徴量名を取得: {feature_names[:5]}... (全{len(feature_names)}個)")
                else:
                    logger.warning("モデルから特徴量名を取得できません")
                    
                # モデルタイプの確認
                model_type = type(model).__name__
                logger.info(f"モデルタイプ: {model_type}")
            except Exception as e:
                logger.warning(f"モデル情報の取得中にエラーが発生しました: {str(e)}")
        
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
    logger.info("=== load_models: モデルの読み込みを開始します ===")
    models = {}
    model_dir_path = Path(model_dir)

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

    # 回帰モデルの読み込み
    for period in target_periods:
        regression_model_path = model_dir_path / f"regression_model_period_{period}.joblib"
        if regression_model_path.exists():
            model = load_model(str(regression_model_path))
            if model:
                models[f"regression_{period}"] = model
                logger.info(f"回帰モデル（{period}期先）を読み込みました")
        else:
            logger.warning(f"回帰モデル {regression_model_path.name} が見つかりません")

    # 分類モデルの読み込み
    for period in target_periods:
        classification_model_path = model_dir_path / f"classification_model_period_{period}.joblib"
        if classification_model_path.exists():
            model = load_model(str(classification_model_path))
            if model:
                models[f"classification_{period}"] = model
                logger.info(f"分類モデル（{period}期先）を読み込みました")
        else:
            logger.warning(f"分類モデル {classification_model_path.name} が見つかりません")
            
    # 二値分類モデルの読み込み
    for period in target_periods:
        binary_model_path = model_dir_path / f"binary_classification_model_period_{period}.joblib"
        if binary_model_path.exists():
            model = load_model(str(binary_model_path))
            if model:
                models[f"binary_classification_{period}"] = model
                logger.info(f"二値分類モデル（{period}期先）を読み込みました")
        else:
            logger.warning(f"二値分類モデル {binary_model_path.name} が見つかりません")
            
    # 閾値ベースの二値分類モデルの読み込み - ワイルドカードパターン検索を追加
    for period in target_periods:
        # 通常のパスでの検索
        threshold_model_path = model_dir_path / f"threshold_binary_classification_model_period_{period}.joblib"
        
        # ファイルが見つからない場合はワイルドカードパターンで検索
        if not threshold_model_path.exists():
            pattern = f"threshold_binary_classification_model_period_{period}*"
            model_files = list(model_dir_path.glob(pattern))
            if model_files:
                logger.info(f"代替パターンで閾値ベースの二値分類モデルを検索: {[f.name for f in model_files]}")
                threshold_model_path = model_files[0]  # 最初に見つかったファイルを使用
            
        if threshold_model_path.exists():
            model = load_model(str(threshold_model_path))
            if model:
                models[f"threshold_binary_classification_{period}"] = model
                logger.info(f"閾値ベースの二値分類モデル（{period}期先）を読み込みました")
        else:
            logger.warning(f"閾値ベースの二値分類モデル {period}期先 が見つかりません")

    logger.info(f"=== load_models: {len(models)} 個のモデルを読み込みました ===")
    if len(models) > 0:
        logger.info(f"読み込まれたモデルの種類: {list(models.keys())}")
    logger.info("load_models: モデルの読み込みを終了します")
    return models