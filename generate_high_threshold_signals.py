#!/usr/bin/env python
"""
高閾値シグナル変数を直接生成するスクリプト
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("high_threshold_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("high_threshold_generation")

def generate_high_threshold_signals():
    """高閾値シグナル変数を直接生成して既存の特徴量ファイルに追加"""
    try:
        logger.info("高閾値シグナル変数の生成を開始します")
        
        # ステップ1: 特徴量ファイルを読み込む
        input_path = Path("data/processed/btcusd_5m_features.csv")
        if not input_path.exists():
            logger.error(f"特徴量ファイルが見つかりません: {input_path}")
            return False
            
        logger.info(f"特徴量ファイルを読み込み中: {input_path}")
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        logger.info(f"特徴量ファイルを読み込みました。行数: {len(df)}, 列数: {len(df.columns)}")
        
        # ステップ2: 目標変数（価格変動率）を確認
        target_cols = [col for col in df.columns if col.startswith("target_price_change_pct_")]
        if not target_cols:
            logger.error("目標変数（価格変動率）が見つかりません")
            return False
            
        logger.info(f"目標変数: {target_cols}")
        
        # ステップ3: 高閾値シグナル変数を直接生成
        periods = [1, 2, 3]  # 予測期間（5分後、10分後、15分後）
        thresholds = [0.001, 0.002, 0.003, 0.005]  # 0.1%, 0.2%, 0.3%, 0.5%
        directions = ["long", "short"]  # ロング/ショート

        # 新しい列を格納するデータフレーム
        high_threshold_df = pd.DataFrame(index=df.index)
        added_columns = []

        for period in periods:
            # 対応する価格変動率の列名
            target_col = f"target_price_change_pct_{period}"
            if target_col not in df.columns:
                logger.warning(f"目標変数 {target_col} が見つかりません。スキップします。")
                continue
                
            logger.info(f"期間 {period} の高閾値シグナル変数を生成中...")
            
            # 価格変動率
            target_change = df[target_col]
            
            for threshold in thresholds:
                threshold_str = str(int(threshold * 1000))  # 0.002 -> '2'
                
                if "long" in directions:
                    # ロング専用シグナル
                    col_name = f"target_high_threshold_{threshold_str}p_long_{period}"
                    high_threshold_df[col_name] = np.where(
                        target_change >= threshold,
                        1,  # 上昇シグナル
                        0   # シグナルなし
                    )
                    added_columns.append(col_name)
                    logger.info(f"生成: {col_name}")
                
                if "short" in directions:
                    # ショート専用シグナル
                    col_name = f"target_high_threshold_{threshold_str}p_short_{period}"
                    high_threshold_df[col_name] = np.where(
                        target_change <= -threshold,
                        1,  # 下落シグナル
                        0   # シグナルなし
                    )
                    added_columns.append(col_name)
                    logger.info(f"生成: {col_name}")

        logger.info(f"生成された高閾値シグナル変数: {len(added_columns)}個")
        
        # 生成された高閾値シグナル変数の統計情報を記録
        for col in added_columns:
            signal_count = high_threshold_df[col].sum()
            signal_ratio = signal_count / len(high_threshold_df) * 100
            logger.info(f"{col}: シグナル数 {signal_count}個 ({signal_ratio:.2f}%)")
        
        # ステップ4: 既存の特徴量に新しい列を追加
        result_df = pd.concat([df, high_threshold_df], axis=1)
        logger.info(f"結合後のデータフレーム: 行数 {len(result_df)}, 列数 {len(result_df.columns)}")
        
        # ステップ5: 特徴量を保存
        output_path = input_path  # 同じファイルを上書き
        result_df.to_csv(output_path)
        logger.info(f"更新された特徴量を保存しました: {output_path}")
        
        # 追加された列を確認
        high_threshold_cols = [col for col in result_df.columns if "high_threshold" in col]
        logger.info(f"最終的な高閾値シグナル変数の数: {len(high_threshold_cols)}個")
        
        return True
    
    except Exception as e:
        logger.exception(f"高閾値シグナル変数の生成中にエラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    success = generate_high_threshold_signals()
    sys.exit(0 if success else 1)
