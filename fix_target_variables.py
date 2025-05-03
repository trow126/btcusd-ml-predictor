#!/usr/bin/env python
"""
目標変数修正スクリプト - 必要な目標変数を生成します
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fix_target_variables.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fix_target_variables")

def fix_target_variables():
    """
    欠落している目標変数を生成または修正する関数
    """
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

        # 既存の目標変数を確認
        existing_targets = [col for col in df.columns if col.startswith("target_")]
        logger.info(f"既存の目標変数: {len(existing_targets)}個")
        for i, col in enumerate(existing_targets[:10]):
            logger.info(f"  {i+1}. {col}")
        if len(existing_targets) > 10:
            logger.info(f"  ...その他 {len(existing_targets)-10}個")

        # 欠落している目標変数を確認
        target_periods = [1, 2, 3]  # 標準の予測期間
        classification_threshold = 0.0005  # 分類閾値

        # 必要な目標変数の名前リスト
        required_targets = []
        for period in target_periods:
            required_targets.extend([
                f'target_smoothed_change_{period}',
                f'target_binary_{period}',
                f'target_threshold_ternary_{period}',
                f'target_threshold_binary_{period}',
                f'target_price_direction_{period}'
            ])

        # 欠落している目標変数を特定
        missing_targets = [target for target in required_targets if target not in df.columns]
        logger.info(f"欠落している目標変数: {len(missing_targets)}個")
        for i, target in enumerate(missing_targets):
            logger.info(f"  {i+1}. {target}")

        # 価格変動率変数が存在するか確認（これは基本的な目標変数）
        price_change_cols = [col for col in df.columns if col.startswith("target_price_change_pct_")]
        if not price_change_cols:
            logger.error("基本的な価格変動率変数が見つかりません。目標変数の生成に失敗する可能性があります。")

            # 価格変動率変数を生成
            logger.info("価格変動率変数を生成します")
            for period in target_periods:
                # 価格変動率（回帰用）
                target_change = df['close'].pct_change(periods=period).shift(-period)
                df[f'target_price_change_pct_{period}'] = target_change
                logger.info(f"target_price_change_pct_{period} を生成しました")

            # 更新された価格変動率変数を確認
            price_change_cols = [col for col in df.columns if col.startswith("target_price_change_pct_")]
            logger.info(f"生成された価格変動率変数: {price_change_cols}")

        # 平滑化した価格系列を計算（必要な場合）
        if not df.get('smoothed_close') is not None:
            logger.info("平滑化価格系列を生成します")
            df['smoothed_close'] = df['close'].rolling(3).mean()

        # 欠落している目標変数を生成
        targets_added = 0

        for period in target_periods:
            logger.info(f"期間 {period} の欠落目標変数を生成中...")

            # 価格変動率変数が存在するか確認
            price_change_col = f'target_price_change_pct_{period}'
            if price_change_col not in df.columns:
                logger.error(f"{price_change_col} が見つかりません。この期間の目標変数生成をスキップします。")
                continue

            # 価格変動率を取得
            target_change = df[price_change_col]

            # 1. 平滑化した価格変動率
            if f'target_smoothed_change_{period}' not in df.columns:
                # 平滑化価格系列が存在する場合
                if 'smoothed_close' in df.columns:
                    smoothed_change = df['smoothed_close'].pct_change(periods=period).shift(-period)
                    df[f'target_smoothed_change_{period}'] = smoothed_change
                    targets_added += 1
                    logger.info(f"target_smoothed_change_{period} を生成しました")
                else:
                    # 平滑化価格系列がない場合は、元の変動率を少し平滑化
                    smoothed_change = target_change.rolling(3).mean()
                    df[f'target_smoothed_change_{period}'] = smoothed_change
                    targets_added += 1
                    logger.info(f"target_smoothed_change_{period} を代替方法で生成しました")

            # 2. 単純な二値分類（上昇/下落）
            if f'target_binary_{period}' not in df.columns:
                df[f'target_binary_{period}'] = np.where(
                    target_change > 0,
                    1,  # 上昇
                    0   # 下落
                )
                targets_added += 1
                logger.info(f"target_binary_{period} を生成しました")

            # 3. 閾値ベースの3分類 (0: 下落, 1: 上昇, 2: 横ばい)
            if f'target_threshold_ternary_{period}' not in df.columns:
                df[f'target_threshold_ternary_{period}'] = np.where(
                    target_change >= classification_threshold,
                    1,  # 閾値以上の上昇
                    np.where(
                        target_change <= -classification_threshold,
                        0,  # 閾値以上の下落
                        2   # 閾値未満の変動（横ばい）
                    )
                )
                targets_added += 1
                logger.info(f"target_threshold_ternary_{period} を生成しました")

            # 4. 閾値ベースの二値分類（上昇 vs 下落、横ばいは除外）
            if f'target_threshold_binary_{period}' not in df.columns:
                df[f'target_threshold_binary_{period}'] = np.where(
                    target_change >= classification_threshold,
                    1,  # 上昇
                    np.where(
                        target_change <= -classification_threshold,
                        0,  # 下落
                        np.nan  # 横ばいはNaN (学習から除外)
                    )
                )
                targets_added += 1
                logger.info(f"target_threshold_binary_{period} を生成しました")

            # 5. 価格変動方向（-1: 下落, 0: 横ばい, 1: 上昇）- 分類モデル用
            if f'target_price_direction_{period}' not in df.columns:
                df[f'target_price_direction_{period}'] = np.where(
                    target_change >= classification_threshold,
                    1,  # 上昇
                    np.where(
                        target_change <= -classification_threshold,
                        -1,  # 下落
                        0   # 横ばい
                    )
                )
                targets_added += 1
                logger.info(f"target_price_direction_{period} を生成しました")

            # 欠落している高閾値シグナル変数を生成
            high_thresholds = [0.001, 0.002, 0.003, 0.005]  # 0.1%, 0.2%, 0.3%, 0.5%
            directions = ["long", "short"]

            for threshold in high_thresholds:
                threshold_str = str(int(threshold * 1000))  # 0.002 -> '2'

                for direction in directions:
                    target_name = f'target_high_threshold_{threshold_str}p_{direction}_{period}'

                    if target_name not in df.columns:
                        if direction == "long":
                            # ロング専用シグナル
                            df[target_name] = np.where(
                                target_change >= threshold,
                                1,  # 上昇シグナル
                                0   # シグナルなし
                            )
                        else:
                            # ショート専用シグナル
                            df[target_name] = np.where(
                                target_change <= -threshold,
                                1,  # 下落シグナル
                                0   # シグナルなし
                            )

                        targets_added += 1
                        logger.info(f"{target_name} を生成しました")

        logger.info(f"合計{targets_added}個の目標変数を追加しました")

        # 修正後の目標変数を確認
        updated_targets = [col for col in df.columns if col.startswith("target_")]
        logger.info(f"修正後の目標変数: {len(updated_targets)}個")

        # 標準目標変数のクラス分布を確認
        for period in target_periods:
            binary_col = f'target_binary_{period}'
            if binary_col in df.columns:
                binary_counts = df[binary_col].value_counts()
                logger.info(f"{binary_col} のクラス分布: {binary_counts.to_dict()}")

        # 高閾値シグナル変数のクラス分布を確認
        high_threshold_cols = [col for col in df.columns if 'high_threshold' in col]
        if high_threshold_cols:
            logger.info(f"高閾値シグナル変数: {len(high_threshold_cols)}個")
            for col in high_threshold_cols[:5]:  # 最初の5つだけ表示
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
    success = fix_target_variables()
    print("処理完了:", "成功" if success else "失敗")