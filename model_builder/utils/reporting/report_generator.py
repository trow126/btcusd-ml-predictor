# model_builder/utils/reporting/report_generator.py
import logging
from typing import Dict, Any

logger = logging.getLogger("report_generator")

def generate_evaluation_report(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    評価結果の要約レポートを生成

    Args:
        evaluation_results: 評価結果のDict

    Returns:
        Dict: 評価レポート
    """
    logger.info("generate_evaluation_report: 評価レポートの生成を開始します")
    report = {
        "regression": {},
        "classification": {},
        "binary_classification": {},
        "threshold_binary_classification": {}
    }

    # 回帰モデルの評価結果
    if "regression" in evaluation_results:
        for period_key, result in evaluation_results["regression"].items():
            report["regression"][period_key] = {
                "mae": result["mae"]
            }

    # 分類モデルの評価結果
    if "classification" in evaluation_results:
        for period_key, result in evaluation_results["classification"].items():
            # クラスごとの精度
            class_precision = {}
            report_dict = result["classification_report"]

            if "-1" in report_dict:
                class_precision["下落"] = {
                    "precision": report_dict["-1"]["precision"],
                    "recall": report_dict["-1"]["recall"],
                    "f1-score": report_dict["-1"]["f1-score"],
                    "support": report_dict["-1"]["support"]
                }

            if "0" in report_dict:
                class_precision["横ばい"] = {
                    "precision": report_dict["0"]["precision"],
                    "recall": report_dict["0"]["recall"],
                    "f1-score": report_dict["0"]["f1-score"],
                    "support": report_dict["0"]["support"]
                }

            if "1" in report_dict:
                class_precision["上昇"] = {
                    "precision": report_dict["1"]["precision"],
                    "recall": report_dict["1"]["recall"],
                    "f1-score": report_dict["1"]["f1-score"],
                    "support": report_dict["1"]["support"]
                }

            report["classification"][period_key] = {
                "accuracy": result["accuracy"],
                "class_metrics": class_precision,
                "confusion_matrix": result["confusion_matrix"]
            }

    # 二値分類モデルの評価結果
    if "binary_classification" in evaluation_results:
        for period_key, result in evaluation_results["binary_classification"].items():
            # エラーがある場合はそのまま渡す
            if "error" in result:
                report["binary_classification"][period_key] = result
                continue
                
            report["binary_classification"][period_key] = {
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "confusion_matrix": result["confusion_matrix"]
            }
    
    # 閾値ベースの二値分類モデルの評価結果
    if "threshold_binary_classification" in evaluation_results:
        for period_key, result in evaluation_results["threshold_binary_classification"].items():
            # エラーがある場合はそのまま渡す
            if "error" in result:
                report["threshold_binary_classification"][period_key] = result
                continue
                
            report["threshold_binary_classification"][period_key] = {
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "confusion_matrix": result["confusion_matrix"],
                "filtered_ratio": result.get("filtered_ratio", 0.0),
                "data_size": result.get("data_size", {})
            }

    logger.info("generate_evaluation_report: 評価レポートの生成を終了します")
    return report