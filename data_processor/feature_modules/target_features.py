# data_processor/feature_modules/target_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.target")

def generate_target_features(df: pd.DataFrame, target_periods: list, classification_threshold: float) -> pd.DataFrame:
    """
    予測対象（目標変数）を生成（改良版）
    
    Args:
        df: 入力データフレーム
        target_periods: 予測期間のリスト
        classification_threshold: 分類の閾値
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("=== 改良版予測対象の目標変数を追加しています ===")
    logger.info(f"分類閾値: {classification_threshold}, 予測期間: {target_periods}")

    features = {}
    
    # ボラティリティベースの動的閾値を計算
    volatility = df['close'].pct_change().rolling(20).std().fillna(classification_threshold)
    logger.info(f"ボラティリティの平均: {volatility.mean():.6f}")
    
    # 価格の平滑化
    smoothed_close = df['close'].rolling(3).mean()
    
    # 各予測時間軸に対する目標変数を生成
    for period in target_periods:
        logger.info(f"期間 {period} の目標変数を生成中...")
        
        # 価格変動率（回帰用）- 元のまま
        target_change = df['close'].pct_change(periods=period).shift(-period)
        features[f'target_price_change_pct_{period}'] = target_change
        
        # 平滑化した価格変動率
        smoothed_change = smoothed_close.pct_change(periods=period).shift(-period)
        features[f'target_smoothed_change_{period}'] = smoothed_change
        
        # 動的閾値を使用した3分類（従来のもの - 保持しておく）
        dynamic_threshold = volatility * 0.8  # より適切な値に調整
        # 未来のボラティリティは予測時には不明なので、現在のボラティリティを使用
        
        features[f'target_price_direction_{period}'] = np.where(
            target_change > dynamic_threshold,
            1,  # 上昇
            np.where(
                target_change < -dynamic_threshold,
                -1,  # 下落
                0   # 横ばい
            )
        )
        
        # 閾値ベースの3分類に名前を変更（より明確に）
        # 0: 下落, 1: 上昇, 2: 横ばい
        features[f'target_threshold_ternary_{period}'] = np.where(
            target_change >= classification_threshold,
            1,  # 閾値以上の上昇
            np.where(
                target_change <= -classification_threshold,
                0,  # 閾値以上の下落
                2   # 閾値未満の変動（横ばい）
            )
        )
        
        # 真の二値分類（上昇 vs 下落、横ばいは除外）
        features[f'target_threshold_binary_{period}'] = np.where(
            target_change >= classification_threshold,
            1,  # 上昇
            np.where(
                target_change <= -classification_threshold,
                0,  # 下落
                np.nan  # 横ばいはNaN (学習から除外)
            )
        )
        
        # 2クラス分類（単純な上昇/下落 - 既存のもの）
        features[f'target_binary_{period}'] = np.where(
            target_change > 0,
            1,  # 上昇
            0   # 下落
        )

    result_df = pd.DataFrame(features, index=df.index)
    
    # 生成した目標変数の統計情報をログに出力
    logger.info("=== 生成された目標変数の統計情報 ===")
    for period in target_periods:
        logger.info(f"---- 期間 {period} の統計情報 ----")
        
        # 回帰ターゲット
        price_change_col = f'target_price_change_pct_{period}'
        if price_change_col in result_df.columns:
            logger.info(f"{price_change_col} の統計: min={result_df[price_change_col].min():.6f}, max={result_df[price_change_col].max():.6f}, mean={result_df[price_change_col].mean():.6f}")
        
        # ターニングポイント3分類 
        ternary_col = f'target_threshold_ternary_{period}'
        if ternary_col in result_df.columns:
            ternary_counts = result_df[ternary_col].value_counts()
            logger.info(f"{ternary_col} のクラス分布: {ternary_counts.to_dict()}")
            
        # 真の二値分類（NaNを含む）
        binary_col = f'target_threshold_binary_{period}'
        if binary_col in result_df.columns:
            # NaNを含む値をカウント
            binary_counts_with_nan = result_df[binary_col].value_counts(dropna=False)
            logger.info(f"{binary_col} のクラス分布(NaN含む): {binary_counts_with_nan.to_dict()}")
            
            # NaNを除いた有効な値のカウント
            valid_binary = result_df[binary_col].dropna()
            binary_valid_counts = valid_binary.value_counts()
            logger.info(f"{binary_col} の有効値クラス分布: {binary_valid_counts.to_dict()}")
            logger.info(f"{binary_col} の有効値数: {len(valid_binary)} ({len(valid_binary)/len(result_df)*100:.2f}%)")
            
            # クラスバランスを確認
            if len(binary_valid_counts) > 1:
                class_ratio = binary_valid_counts.min() / binary_valid_counts.max()
                logger.info(f"{binary_col} のクラスバランス比率: {class_ratio:.4f} (1に近いほど均等)")

    logger.info(f"目標変数を含むデータフレームを生成完了。サイズ: {result_df.shape}")
    return result_df
