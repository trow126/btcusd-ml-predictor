#!/usr/bin/env python
"""
BTCUSD ML Predictor - 高閾値シグナル変数の追加スクリプト
特徴量データに高閾値シグナル変数を追加するためのユーティリティスクリプト
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("add_high_threshold_signals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("add_high_threshold_signals")

def add_high_threshold_signals(
    input_path=None,
    output_path=None,
    periods=None,
    thresholds=None,
    directions=None
):
    """
    特徴量データに高閾値シグナル変数を追加する関数
    Args:
        input_path: 入力特徴量ファイルのパス
        output_path: 出力ファイルのパス（デフォルトは入力と同じ）
        periods: 予測期間のリスト（デフォルト: [1, 2, 3]）
        thresholds: 閾値のリスト（デフォルト: [0.001, 0.002, 0.003, 0.005]）
        directions: 方向のリスト（デフォルト: ["long", "short"]）
    
    Returns:
        bool: 処理が成功したかどうか
    """
    # デフォルト値の設定
    if input_path is None:
        input_path = Path("data/processed/btcusd_5m_features.csv")
    else:
        input_path = Path(input_path)
        
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
        
    if periods is None:
        periods = [1, 2, 3]
        
    if thresholds is None:
        thresholds = [0.001, 0.002, 0.003, 0.005]
        
    if directions is None:
        directions = ["long", "short"]
    
    logger.info(f"高閾値シグナル変数の追加処理を開始: 入力={input_path}, 出力={output_path}")
    logger.info(f"期間: {periods}, 閾値: {thresholds}, 方向: {directions}")
    
    try:
        # 入力ファイルの存在確認
        if not input_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_path}")
            return False
            
        # 特徴量ファイルを読み込み
        logger.info(f"特徴量ファイル読み込み中: {input_path}")
        try:
            df = pd.read_csv(input_path, parse_dates=["timestamp"])
            logger.info(f"特徴量ファイル読み込み完了: {len(df)}行, {len(df.columns)}列")
        except Exception as e:
            logger.error(f"特徴量ファイル読み込み中にエラー: {str(e)}")
            return False
            
        # 既存の高閾値シグナル変数を確認
        high_threshold_cols = [col for col in df.columns if "high_threshold" in col]
        if high_threshold_cols:
            logger.info(f"既存の高閾値シグナル変数: {len(high_threshold_cols)}個")
            logger.info(f"既存の高閾値シグナル変数を削除します")
            df = df.drop(columns=high_threshold_cols)
            logger.info(f"削除後の列数: {len(df.columns)}")
        else:
            logger.info("既存の高閾値シグナル変数はありません")
            
        # ターゲット変数（価格変動率）の確認
        target_cols = []
        for period in periods:
            # 対応する価格変動率の列名
            target_col = f"target_price_change_pct_{period}"
            if target_col in df.columns:
                target_cols.append((period, target_col))
                logger.info(f"期間{period}のターゲット列として {target_col} を使用")
            else:
                logger.warning(f"期間{period}のターゲット列 {target_col} が見つかりません")
                
        if not target_cols:
            logger.error("使用可能なターゲット列がありません")
            return False
            
        # 高閾値シグナル変数の生成
        logger.info("高閾値シグナル変数の生成を開始")
        signals_added = 0
        
        for period, target_col in target_cols:
            logger.info(f"期間{period}の変数を生成中 ({target_col}を使用)")
            target_values = df[target_col].values
            
            for threshold in thresholds:
                threshold_str = str(int(threshold * 1000))
                
                if "long" in directions:
                    # ロング専用シグナル
                    long_col = f"target_high_threshold_{threshold_str}p_long_{period}"
                    df[long_col] = np.where(target_values >= threshold, 1, 0)
                    signals_added += 1
                    
                if "short" in directions:
                    # ショート専用シグナル
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
        df.to_csv(output_path, index=False)
        logger.info(f"更新されたファイルを保存しました: {output_path}")
        
        return True
        
    except Exception as e:
        logger.exception(f"高閾値シグナル変数の追加中にエラーが発生しました: {str(e)}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # コマンドライン引数の処理
    import argparse
    parser = argparse.ArgumentParser(description="特徴量データに高閾値シグナル変数を追加するスクリプト")
    parser.add_argument("--input", type=str, default="data/processed/btcusd_5m_features.csv", help="入力特徴量ファイル")
    parser.add_argument("--output", type=str, help="出力ファイル（指定なしの場合は入力ファイルを上書き）")
    parser.add_argument("--periods", type=int, nargs="+", default=[1, 2, 3], help="予測期間")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.002, 0.003, 0.005], help="閾値")
    parser.add_argument("--directions", type=str, nargs="+", default=["long", "short"], help="方向")
    
    args = parser.parse_args()
    
    # 関数実行
    success = add_high_threshold_signals(
        input_path=args.input,
        output_path=args.output,
        periods=args.periods,
        thresholds=args.thresholds,
        directions=args.directions
    )
    
    sys.exit(0 if success else 1)
