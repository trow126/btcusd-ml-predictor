#!/usr/bin/env python
"""
CSVファイルを検証し、必要な目標変数を追加する修正スクリプト
"""
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("verify_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("verify_fix")

def verify_and_fix_csv():
    """CSVファイルを検証し修正"""
    try:
        # ファイルパス設定
        features_path = Path("data/processed/btcusd_5m_features.csv")
        
        if not features_path.exists():
            logger.error(f"特徴量ファイルが見つかりません: {features_path}")
            return False
            
        # データ読み込み
        logger.info(f"ファイル読み込み: {features_path}")
        df = pd.read_csv(features_path)
        logger.info(f"読み込み成功: 行数={len(df)}, 列数={len(df.columns)}")
        
        # すべての列名をリスト表示
        logger.info("すべての列名:")
        for i, col in enumerate(sorted(df.columns)):
            logger.info(f"  {i+1}. {col}")
        
        # target_ で始まる列名を確認
        target_cols = [col for col in df.columns if col.startswith("target_")]
        logger.info(f"target_ で始まる列: {len(target_cols)}個")
        for i, col in enumerate(sorted(target_cols)):
            logger.info(f"  {i+1}. {col}")
            
        # 必要な列が存在するか確認
        required_targets = [
            'target_binary_1', 'target_binary_2', 'target_binary_3',
            'target_smoothed_change_1', 'target_smoothed_change_2', 'target_smoothed_change_3',
            'target_high_threshold_2p_long_1', 'target_high_threshold_2p_short_1',
            'target_high_threshold_3p_long_1', 'target_high_threshold_3p_short_1',
            'target_high_threshold_2p_long_2', 'target_high_threshold_2p_short_2',
            'target_high_threshold_3p_long_2', 'target_high_threshold_3p_short_2',
            'target_high_threshold_2p_long_3', 'target_high_threshold_2p_short_3',
            'target_high_threshold_3p_long_3', 'target_high_threshold_3p_short_3'
        ]
        
        # 不足している列を確認
        missing_cols = [col for col in required_targets if col not in df.columns]
        logger.info(f"不足している列: {len(missing_cols)}個")
        for col in missing_cols:
            logger.info(f"  - {col}")
            
        # price_change_pct列の確認
        price_change_cols = [col for col in df.columns if 'price_change_pct' in col]
        logger.info(f"price_change_pct列: {price_change_cols}")
        
        # 変数追加処理
        if missing_cols:
            logger.info("不足している変数を追加します")
            
            # 平滑化した価格系列の作成（必要な場合）
            if 'smoothed_close' not in df.columns:
                df['smoothed_close'] = df['close'].rolling(3).mean()
                logger.info("smoothed_close 列を追加しました")
                
            # 各期間ごとの処理
            for period in [1, 2, 3]:
                # 基本となる価格変動率変数が存在するか確認
                price_change_col = f'target_price_change_pct_{period}'
                if price_change_col not in df.columns:
                    # 価格変動率を計算
                    df[price_change_col] = df['close'].pct_change(periods=period).shift(-period)
                    logger.info(f"{price_change_col} を追加しました")
                
                # 1. 平滑化した価格変動率
                smoothed_change_col = f'target_smoothed_change_{period}'
                if smoothed_change_col not in df.columns:
                    df[smoothed_change_col] = df['smoothed_close'].pct_change(periods=period).shift(-period)
                    logger.info(f"{smoothed_change_col} を追加しました")
                    
                # 2. 単純な二値分類（上昇/下落）
                binary_col = f'target_binary_{period}'
                if binary_col not in df.columns:
                    df[binary_col] = np.where(df[price_change_col] > 0, 1, 0)
                    logger.info(f"{binary_col} を追加しました")
                    
                # 3. 高閾値シグナル変数
                for threshold in [2, 3]:  # 2%, 3%
                    threshold_value = threshold / 100.0  # パーセント値を小数に変換
                    
                    # ロングシグナル
                    long_col = f'target_high_threshold_{threshold}p_long_{period}'
                    if long_col not in df.columns:
                        df[long_col] = np.where(df[price_change_col] >= threshold_value, 1, 0)
                        logger.info(f"{long_col} を追加しました")
                        
                    # ショートシグナル
                    short_col = f'target_high_threshold_{threshold}p_short_{period}'
                    if short_col not in df.columns:
                        df[short_col] = np.where(df[price_change_col] <= -threshold_value, 1, 0)
                        logger.info(f"{short_col} を追加しました")
            
            # ファイル保存
            logger.info("修正後のファイルを保存します")
            df.to_csv(features_path, index=False)
            
            # 保存したファイルを再度読み込み
            df_verify = pd.read_csv(features_path)
            still_missing = [col for col in required_targets if col not in df_verify.columns]
            if still_missing:
                logger.error(f"保存後も {len(still_missing)}個 の列が不足しています")
                for col in still_missing:
                    logger.error(f"  - {col}")
                return False
            else:
                logger.info("すべての必要な列が正常に追加され保存されました")
                return True
        else:
            logger.info("すべての必要な列が既に存在しています")
            return True
            
    except Exception as e:
        logger.exception(f"エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    success = verify_and_fix_csv()
    print("処理完了:", "成功" if success else "失敗")
