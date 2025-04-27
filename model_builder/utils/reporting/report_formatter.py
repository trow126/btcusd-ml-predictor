# model_builder/utils/reporting/report_formatter.py
import numpy as np
from typing import Dict, Any

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

def generate_training_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    トレーニング結果の要約レポートを生成

    Args:
        results: トレーニング結果の辞書

    Returns:
        Dict: トレーニング結果のレポート
    """
    report = {
        "regression": {},
        "classification": {},
        "binary_classification": {}
    }

    # 回帰モデルの結果
    if "regression" in results:
        regression_results = results["regression"]
        for target_name, result in regression_results.items():
            period = int(target_name.split("_")[1])

            report["regression"][f"period_{period}"] = {
                "mae": result["mae"],
                "top_features": result["feature_importance"]
            }

    # 分類モデルの結果
    if "classification" in results:
        classification_results = results["classification"]
        for target_name, result in classification_results.items():
            period = int(target_name.split("_")[1])

            report["classification"][f"period_{period}"] = {
                "accuracy": result["accuracy"],
                "class_accuracy": {
                    "-1 (下落)": result["classification_report"]["-1"]["precision"] if "-1" in result["classification_report"] else 0,
                    "0 (横ばい)": result["classification_report"]["0"]["precision"] if "0" in result["classification_report"] else 0,
                    "1 (上昇)": result["classification_report"]["1"]["precision"] if "1" in result["classification_report"] else 0
                },
                "confusion_matrix": result["confusion_matrix"].tolist(),
                "top_features": result["feature_importance"]
            }

    # 二値分類モデルの結果
    if "binary_classification" in results and results["binary_classification"]:
        binary_classification_results = results["binary_classification"]
        for target_name, result in binary_classification_results.items():
            period = int(target_name.split("_")[1])

            report["binary_classification"][f"period_{period}"] = {
                "accuracy": result["accuracy"],
                "roc_auc": result.get("roc_auc", 0),
                "class_accuracy": {
                    "0 (下落)": result["classification_report"]["0"]["precision"] if "0" in result["classification_report"] else 0,
                    "1 (上昇)": result["classification_report"]["1"]["precision"] if "1" in result["classification_report"] else 0
                },
                "confusion_matrix": result["confusion_matrix"].tolist(),
                "top_features": result["feature_importance"]
            }

    return report