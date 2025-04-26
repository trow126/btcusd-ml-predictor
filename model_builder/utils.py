# model_builder/utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import json

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_utils")

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

def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> Dict[str, float]:
    """
    モデルの特徴量重要度を取得

    Args:
        model: LightGBMモデル
        feature_names: 特徴量名のリスト
        top_n: 上位N個の特徴量を返す

    Returns:
        Dict: 特徴量名と重要度のDict
    """
    importance = model.feature_importance(importance_type="gain")
    feature_importance = dict(zip(feature_names, importance))

    # 上位N個の特徴量を取得
    sorted_importance = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return dict(sorted_importance)

def format_confusion_matrix(cm: np.ndarray) -> str:
    """
    混同行列を見やすい形式にフォーマット

    Args:
        cm: 混同行列（numpy配列）

    Returns:
        str: フォーマットされた混同行列の文字列
    """
    # クラスラベル
    labels = ["下落(-1)", "横ばい(0)", "上昇(1)"]

    # 行ラベルと列ラベルを追加
    cm_str = f"{'実際\\予測':<10}"
    for label in labels:
        cm_str += f"{label:>10}"
    cm_str += "\n"

    # 各行のデータを追加
    for i, label in enumerate(labels):
        cm_str += f"{label:<10}"
        for j in range(cm.shape[1]):
            cm_str += f"{cm[i, j]:>10}"
        cm_str += "\n"

    return cm_str

def format_model_report(report: Dict[str, Any]) -> str:
    """
    モデル評価レポートを見やすい形式にフォーマット

    Args:
        report: モデル評価レポート

    Returns:
        str: フォーマットされたレポートの文字列
    """
    report_str = "=== モデル評価レポート ===\n\n"

    # 回帰モデルのレポート
    report_str += "## 回帰モデル（価格変動率予測）\n\n"
    for period_key, result in report["regression"].items():
        period = period_key.split("_")[1]
        report_str += f"### {period}期先予測\n"
        report_str += f"MAE: {result['mae']:.6f}\n\n"

    # 分類モデルのレポート
    report_str += "## 分類モデル（価格変動方向予測）\n\n"
    for period_key, result in report["classification"].items():
        period = period_key.split("_")[1]
        report_str += f"### {period}期先予測\n"
        report_str += f"正解率: {result['accuracy']:.4f}\n\n"

        # クラスごとの精度
        report_str += "クラスごとの精度:\n"
        for class_label, metrics in result["class_metrics"].items():
            report_str += f"  {class_label}: Precision={metrics['precision']:.4f}, "
            report_str += f"Recall={metrics['recall']:.4f}, "
            report_str += f"F1-score={metrics['f1-score']:.4f}, "
            report_str += f"Support={metrics['support']}\n"
        report_str += "\n"

        # 混同行列
        report_str += "混同行列:\n"
        cm = np.array(result["confusion_matrix"])
        report_str += format_confusion_matrix(cm)
        report_str += "\n"

    return report_str

def predict_next_price_movement(
    models: Dict[str, Any],
    latest_data: pd.DataFrame,
    period: int = 1
) -> Dict[str, Any]:
    """
    次の価格変動を予測

    Args:
        models: 予測モデルのDict
        latest_data: 最新のデータ
        period: 予測期間 (1=5分後, 2=10分後, 3=15分後)

    Returns:
        Dict: 予測結果
    """
    regression_model_key = f"regression_{period}"
    classification_model_key = f"classification_{period}"

    if regression_model_key not in models or classification_model_key not in models:
        logger.error(f"{period}期先のモデルが見つかりません")
        return {}

    regression_model = models[regression_model_key]
    classification_model = models[classification_model_key]

    # 必要な特徴量のみを抽出
    features = latest_data.iloc[-1:].copy()

    # 目標変数を除外
    feature_cols = [col for col in features.columns if not col.startswith("target_")]
    features = features[feature_cols]

    # 予測
    price_change = float(regression_model.predict(features)[0])

    # 分類予測（確率）
    direction_proba = classification_model.predict(features)[0]

    # 分類予測（クラス）
    direction = int(np.argmax(direction_proba) - 1)  # 0, 1, 2を-1, 0, 1に変換

    # 方向ラベル
    direction_label = {
        -1: "下落",
        0: "横ばい",
        1: "上昇"
    }[direction]

    # 現在価格
    current_price = float(latest_data["close"].iloc[-1])

    # 予測価格
    predicted_price = current_price * (1 + price_change)

    # 予測信頼度（確率）
    confidence = float(direction_proba[direction + 1])  # -1, 0, 1を0, 1, 2に変換

    return {
        "current_time": latest_data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": current_price,
        "period": period,
        "prediction_time": f"{period * 5}分後",
        "predicted_change": price_change,
        "predicted_price": predicted_price,
        "predicted_direction": direction,
        "predicted_direction_label": direction_label,
        "confidence": confidence
    }