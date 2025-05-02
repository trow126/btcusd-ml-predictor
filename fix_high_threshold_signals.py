#!/usr/bin/env python
"""
高閾値シグナル変数を生成するための修正スクリプト
"""
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fix_signals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fix_signals")

def fix_signals():
    """高閾値シグナル変数を生成"""
    # ファイルパス設定
    input_path = Path("data/processed/btcusd_5m_features.csv")
    if not input_path.exists():
        logger.error(f"ファイルが見つかりません: {input_path}")
        return False
    
    # ファイル読み込み
    try:
        logger.info(f"ファイル読み込み: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"読み込み成功: 行数={len(df)}, 列数={len(df.columns)}")
        
        # 列名確認
        logger.info(f"最初の10列: {list(df.columns)[:10]}")
        
        # ターゲット変数確認
        target_cols = [col for col in df.columns if col.startswith("target_")]
        logger.info(f"ターゲット変数: {target_cols}")
        
        # 価格変動率の列確認
        price_change_cols = [col for col in df.columns if "price_change" in col]
        logger.info(f"価格変動率列: {price_change_cols}")
        
        # パラメータ設定
        periods = [1, 2, 3]
        thresholds = [0.001, 0.002, 0.003, 0.005]
        directions = ["long", "short"]
        
        # 正しい名前のターゲット列を特定
        valid_target_cols = []
        for period in periods:
            # 考えられる形式をすべて試す
            candidates = [
                f"target_price_change_pct_{period}",
                f"target_price_change_{period}",
                f"target_change_{period}"
            ]
            
            for col in candidates:
                if col in df.columns:
                    valid_target_cols.append((period, col))
                    logger.info(f"期間{period}の有効なターゲット列: {col}")
                    break
        
        if not valid_target_cols:
            logger.error("有効なターゲット列が見つかりません")
            return False
        
        # 高閾値シグナル変数生成
        logger.info("高閾値シグナル変数を生成します")
        signals_added = 0
        
        for period, target_col in valid_target_cols:
            logger.info(f"期間{period}の変数を生成中 ({target_col}を使用)")
            target_values = df[target_col].values
            
            for threshold in thresholds:
                threshold_str = str(int(threshold * 1000))
                
                # ロングシグナル
                long_col = f"target_high_threshold_{threshold_str}p_long_{period}"
                df[long_col] = np.where(target_values >= threshold, 1, 0)
                signals_added += 1
                
                # ショートシグナル
                short_col = f"target_high_threshold_{threshold_str}p_short_{period}"
                df[short_col] = np.where(target_values <= -threshold, 1, 0)
                signals_added += 1
        
        logger.info(f"合計{signals_added}個の高閾値シグナル変数を追加しました")
        
        # 変数の確認
        high_threshold_cols = [col for col in df.columns if "high_threshold" in col]
        logger.info(f"生成された高閾値シグナル変数: {high_threshold_cols}")
        
        # 統計情報
        for col in high_threshold_cols:
            signal_count = df[col].sum()
            signal_ratio = signal_count / len(df) * 100
            logger.info(f"{col}: シグナル数={signal_count}個 ({signal_ratio:.2f}%)")
        
        # ファイル保存
        df.to_csv(input_path, index=False)
        logger.info(f"更新されたファイルを保存しました: {input_path}")
        
        return True
    
    except Exception as e:
        logger.exception(f"エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    success = fix_signals()
    print("処理完了:", "成功" if success else "失敗")
