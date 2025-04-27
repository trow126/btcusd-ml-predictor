# model_builder/utils/reporting/report_serializer.py
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("report_serializer")

def save_evaluation_report(report: Dict[str, Any], output_dir: str, filename: str = "model_evaluation_report.json") -> bool:
    """
    評価レポートを保存

    Args:
        report: 保存する評価レポート
        output_dir: 出力ディレクトリパス
        filename: 出力ファイル名（デフォルト: model_evaluation_report.json）

    Returns:
        bool: 保存が成功したかどうか
    """
    logger.info("save_evaluation_report: 評価レポートの保存を開始します")
    # 出力ディレクトリが存在しない場合は作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # レポートをJSONファイルに保存
    report_path = output_path / filename
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"save_evaluation_report: 評価レポートを {report_path} に保存しました")
        logger.info("save_evaluation_report: 評価レポートの保存を終了します")
        return True
    except Exception as e:
        logger.error(f"レポート保存エラー: {e}")
        return False