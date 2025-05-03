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

    try:
        # 回帰モデルの評価結果
        if "regression" in evaluation_results:
            for period_key, result in evaluation_results["regression"].items():
                # 結果が辞書でない場合や必要なキーが存在しない場合に対応
                if not isinstance(result, dict):
                    report["regression"][period_key] = {"error": "invalid_result"}
                    continue
                    
                report["regression"][period_key] = {
                    "mae": result.get("mae", 0.0)  # キーが存在しない場合はデフォルト値を使用
                }

        # 分類モデルの評価結果
        if "classification" in evaluation_results:
            for period_key, result in evaluation_results["classification"].items():
                # エラーチェック
                if not isinstance(result, dict):
                    report["classification"][period_key] = {"error": "invalid_result"}
                    continue
                    
                # エラーがある場合はそのまま渡す
                if "error" in result:
                    report["classification"][period_key] = result
                    continue
                    
                # クラスごとの精度
                class_precision = {}
                report_dict = result.get("classification_report", {})

                if isinstance(report_dict, dict):
                    if "-1" in report_dict:
                        class_precision["下落"] = {
                            "precision": report_dict["-1"].get("precision", 0.0),
                            "recall": report_dict["-1"].get("recall", 0.0),
                            "f1-score": report_dict["-1"].get("f1-score", 0.0),
                            "support": report_dict["-1"].get("support", 0)
                        }

                    if "0" in report_dict:
                        class_precision["横ばい"] = {
                            "precision": report_dict["0"].get("precision", 0.0),
                            "recall": report_dict["0"].get("recall", 0.0),
                            "f1-score": report_dict["0"].get("f1-score", 0.0),
                            "support": report_dict["0"].get("support", 0)
                        }

                    if "1" in report_dict:
                        class_precision["上昇"] = {
                            "precision": report_dict["1"].get("precision", 0.0),
                            "recall": report_dict["1"].get("recall", 0.0),
                            "f1-score": report_dict["1"].get("f1-score", 0.0),
                            "support": report_dict["1"].get("support", 0)
                        }

                report["classification"][period_key] = {
                    "accuracy": result.get("accuracy", 0.0),
                    "class_metrics": class_precision,
                    "confusion_matrix": result.get("confusion_matrix", [])
                }

        # 二値分類モデルの評価結果
        if "binary_classification" in evaluation_results:
            for period_key, result in evaluation_results["binary_classification"].items():
                # エラーチェック
                if not isinstance(result, dict):
                    report["binary_classification"][period_key] = {"error": "invalid_result"}
                    continue
                    
                # エラーがある場合はそのまま渡す
                if "error" in result:
                    report["binary_classification"][period_key] = result
                    continue
                    
                report["binary_classification"][period_key] = {
                    "accuracy": result.get("accuracy", 0.0),
                    "precision": result.get("precision", 0.0),
                    "recall": result.get("recall", 0.0),
                    "f1_score": result.get("f1_score", 0.0),
                    "confusion_matrix": result.get("confusion_matrix", [])
                }
        
        # 閾値ベースの二値分類モデルの評価結果
        if "threshold_binary_classification" in evaluation_results:
            for period_key, result in evaluation_results["threshold_binary_classification"].items():
                # 結果がdict型でなければエラーとして扱う
                if not isinstance(result, dict):
                    report["threshold_binary_classification"][period_key] = {"error": "invalid_result"}
                    continue
                    
                # エラーがある場合はそのまま渡す
                if "error" in result:
                    report["threshold_binary_classification"][period_key] = result
                    continue
                    
                # すべてのキーに対してget()を使用してデフォルト値を設定
                report["threshold_binary_classification"][period_key] = {
                    "accuracy": result.get("accuracy", 0.0),
                    "precision": result.get("precision", 0.0),
                    "recall": result.get("recall", 0.0),
                    "f1_score": result.get("f1_score", 0.0),
                    "confusion_matrix": result.get("confusion_matrix", []),
                    "filtered_ratio": result.get("filtered_ratio", 0.0),
                    "data_size": result.get("data_size", {})
                }
    except Exception as e:
        logger.error(f"レポート生成中にエラーが発生しました: {str(e)}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        # 最小限のレポートを返す
        report = {
            "error": "report_generation_error",
            "message": str(e),
            "regression": {},
            "classification": {},
            "binary_classification": {},
            "threshold_binary_classification": {}
        }

    logger.info("generate_evaluation_report: 評価レポートの生成を終了します")
    return report