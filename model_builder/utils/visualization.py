# model_builder/utils/visualization.py
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