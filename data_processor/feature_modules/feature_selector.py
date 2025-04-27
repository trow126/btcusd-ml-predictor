# data_processor/feature_modules/feature_selector.py
import pandas as pd
import logging

logger = logging.getLogger("feature_engineering.selector")

def select_important_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    重要度の高い特徴量を選別
    
    Args:
        df: 特徴量を含むDataFrame
        
    Returns:
        DataFrame: 選別した特徴量を含むDataFrame
    """
    logger.info("重要な特徴量を選択しています")
    
    # 前回のモデルで重要度が高かった特徴量のリスト
    important_features = [
        # 基本価格・出来高データ
        'close', 'volume', 'turnover',

        # ローソク足パターン関連
        'candle_size', 'body_size', 'upper_shadow', 'lower_shadow',

        # 価格変動率関連
        'price_change_pct_1', 'price_change_pct_2', 'price_change_pct_3',
        'price_change_pct_4', 'price_change_pct_6', 'price_change_pct_10',

        # 出来高関連
        'volume_change_pct_3', 'volume_change_pct_10',
        'volume_sma_5', 'volume_sma_10', 'volume_sma_20',
        'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',

        # 移動平均関連
        'sma_diff_pct_5', 'sma_diff_pct_20', 'sma_diff_pct_50', 'sma_diff_pct_200',
        'ema_50',

        # ボリンジャーバンド関連
        'bb_width', 'bb_std', 'bb_position',

        # RSI関連
        'rsi_6', 'rsi_14', 'rsi_change_1',

        # 高値/安値からの距離関連
        'dist_from_high_5', 'dist_from_high_10', 'dist_from_high_20', 'dist_from_high_50',
        'dist_from_low_5', 'dist_from_low_10', 'dist_from_low_50',

        # MACD関連
        'macd', 'macd_signal', 'macd_hist',

        # ストキャスティクス関連
        'stoch_k', 'stoch_d',

        # 新規追加の高度な特徴量
        'atr', 'price_change_to_atr', 'volatility_change',
        'price_volume_trend', 'price_momentum', 'price_acceleration',
        'trend_strength_5', 'trend_strength_20',
        'market_efficiency_10', 'fisher_transform_rsi',
        'trend_intensity',
        'price_shock', 'ma_convergence',

        # 目標変数は常に含める
        'target_price_change_pct_1', 'target_price_change_pct_2', 'target_price_change_pct_3',
        'target_price_direction_1', 'target_price_direction_2', 'target_price_direction_3'
    ]

    # 特徴量選択（存在する列のみ）
    existing_features = [col for col in important_features if col in df.columns]
    selected_df = df[existing_features]

    logger.info(f"重要な特徴量 {len(existing_features)}個を選択しました")

    return selected_df
