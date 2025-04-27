# model_builder/utils/logging_utils.py
import logging

def setup_logger(logger_name):
    """ロガーの設定を行う関数"""
    # ロガーの設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(logger_name)