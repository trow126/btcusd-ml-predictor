#!/usr/bin/env python
"""
直接CSVファイルを修正して目標変数を追加するスクリプト
必要な変数を直接追加した後、その変数が存在するか確認します
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("direct_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("direct_fix")

def direct_fix():
    """CSVファイルを直接修正して目標変数を追加"""
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
        
        # 必要な列が存在するか確認
        if 'close' not in df.columns:
            logger.error("'close'列が見つかりません。データ形式が想定と異なります。")
            return False
            
        # 既存の目標変数を確認
        target_cols = [col for col in df.columns if col.startswith("target_")]
        logger.info(f"既存の目標変数: {len(target_cols)}個")
        
        # 修正が必要な変数のリスト
        required_targets = [
            'target_smoothed_change_1', 'target_smoothed_change_2', 'target_smoothed_change_3',
            'target_binary_1', 'target_binary_2', 'target_binary_3'
        ]
        
        # 高閾値シグナル変数
        high_threshold_targets = []
        for period in [1, 2, 3]:
            for threshold in [2, 3]:  # 2%, 3%
                for direction in ['long', 'short']:
                    high_threshold_targets.append(f'target_high_threshold_{threshold}p_{direction}_{period}')
                    
        required_targets.extend(high_threshold_targets)
        
        # 欠落している変数を特定
        missing_targets = [target for target in required_targets if target not in df.columns]
        logger.info(f"欠落している変数: {len(missing_targets)}個")
        for i, target in enumerate(missing_targets):
            logger.info(f"  {i+1}. {target}")
            
        # 変数を追加
        changes_made = False
        
        # 平滑化した価格系列の作成（必要な場合）
        if 'smoothed_close' not in df.columns:
            df['smoothed_close'] = df['close'].rolling(3).mean()
            changes_made = True
            logger.info("smoothed_close 列を追加しました")
            
        # 期間ごとの処理
        for period in [1, 2, 3]:
            # 基本となる価格変動率変数が存在するか確認
            price_change_col = f'target_price_change_pct_{period}'
            if price_change_col not in df.columns:
                # 価格変動率を計算
                df[price_change_col] = df['close'].pct_change(periods=period).shift(-period)
                changes_made = True
                logger.info(f"{price_change_col} を追加しました")
            
            # 1. 平滑化した価格変動率 - smoothed_change_*
            smoothed_change_col = f'target_smoothed_change_{period}'
            if smoothed_change_col not in df.columns:
                df[smoothed_change_col] = df['smoothed_close'].pct_change(periods=period).shift(-period)
                changes_made = True
                logger.info(f"{smoothed_change_col} を追加しました")
                
            # 2. 単純な二値分類（上昇/下落） - binary_*
            binary_col = f'target_binary_{period}'
            if binary_col not in df.columns:
                df[binary_col] = np.where(df[price_change_col] > 0, 1, 0)
                changes_made = True
                logger.info(f"{binary_col} を追加しました")
                
            # 3. 高閾値シグナル変数 - high_threshold_*
            for threshold in [2, 3]:  # 2%, 3%
                threshold_value = threshold / 100.0  # パーセント値を小数に変換
                
                # ロングシグナル
                long_col = f'target_high_threshold_{threshold}p_long_{period}'
                if long_col not in df.columns:
                    df[long_col] = np.where(df[price_change_col] >= threshold_value, 1, 0)
                    changes_made = True
                    logger.info(f"{long_col} を追加しました")
                    
                # ショートシグナル
                short_col = f'target_high_threshold_{threshold}p_short_{period}'
                if short_col not in df.columns:
                    df[short_col] = np.where(df[price_change_col] <= -threshold_value, 1, 0)
                    changes_made = True
                    logger.info(f"{short_col} を追加しました")
        
        # 追加された変数の確認
        if changes_made:
            # 目標変数の確認
            new_target_cols = [col for col in df.columns if col.startswith("target_")]
            logger.info(f"修正後の目標変数: {len(new_target_cols)}個")
            
            # 高閾値シグナル変数の確認
            high_threshold_cols = [col for col in new_target_cols if 'high_threshold' in col]
            logger.info(f"高閾値シグナル変数: {len(high_threshold_cols)}個")
            if high_threshold_cols:
                for col in high_threshold_cols[:5]:  # 最初の5つだけ表示
                    signal_count = df[col].sum()
                    signal_ratio = signal_count / len(df) * 100
                    logger.info(f"{col}: シグナル数={signal_count}個 ({signal_ratio:.2f}%)")
                    
            # バイナリ変数の確認
            binary_cols = [col for col in new_target_cols if col.startswith('target_binary_')]
            logger.info(f"binary変数: {len(binary_cols)}個")
            for col in binary_cols:
                value_counts = df[col].value_counts()
                logger.info(f"{col} の値分布: {value_counts.to_dict()}")
                
            # smoothed変数の確認
            smoothed_cols = [col for col in new_target_cols if 'smoothed_change' in col]
            logger.info(f"smoothed_change変数: {len(smoothed_cols)}個")
            for col in smoothed_cols:
                logger.info(f"{col} の統計: 平均={df[col].mean():.6f}, 最小={df[col].min():.6f}, 最大={df[col].max():.6f}")
            
            # ファイル保存
            df.to_csv(features_path, index=False)
            logger.info(f"修正されたファイルを保存しました: {features_path}")
            
            # 一度読み込み直して保存が正しく行われたか確認
            verify_df = pd.read_csv(features_path)
            missing_after = [target for target in required_targets if target not in verify_df.columns]
            if missing_after:
                logger.warning(f"保存後も {len(missing_after)}個 の変数が欠落しています")
                for target in missing_after[:5]:
                    logger.warning(f"  - {target}")
            else:
                logger.info("全ての必要な変数が正常に保存されました")
            
            return True
        else:
            logger.info("変更は不要でした - 全ての必要な変数が既に存在しています")
            return True
        
    except Exception as e:
        logger.exception(f"エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    success = direct_fix()
    print("処理完了:", "成功" if success else "失敗")
