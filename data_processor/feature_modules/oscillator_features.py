# data_processor/feature_modules/oscillator_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("feature_engineering.oscillator")

def generate_rsi_features(df: pd.DataFrame, rsi_periods: list) -> pd.DataFrame:
    """
    RSIに関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        rsi_periods: RSIの期間リスト
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("RSIの特徴量を追加しています")

    features = {}

    # 複数期間のRSI
    for period in rsi_periods:
        # 価格の差分
        delta = df['close'].diff()

        # 上昇幅と下落幅
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 平均上昇幅と平均下落幅
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # RSI計算
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features[f'rsi_{period}'] = rsi

        # RSIの変化率
        features[f'rsi_{period}_change'] = rsi.diff(1)

        # RSIの過買い/過売りシグナル
        features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
        features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)

    # RSIのダイバージェンス検出（価格が上昇しているがRSIが下降）
    if 'rsi_14' in features:
        features['rsi_divergence'] = np.where(
            (df['close'].diff(5) > 0) & (features['rsi_14'].diff(5) < 0),
            -1,  # ベアリッシュ・ダイバージェンス
            np.where(
                (df['close'].diff(5) < 0) & (features['rsi_14'].diff(5) > 0),
                1,  # ブリッシュ・ダイバージェンス
                0  # ダイバージェンスなし
            )
        )

    return pd.DataFrame(features, index=df.index)

def generate_macd_features(df: pd.DataFrame, macd_params: dict) -> pd.DataFrame:
    """
    MACDに関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        macd_params: MACDのパラメータ辞書
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("MACDの特徴量を追加しています")

    features = {}
    fast = macd_params["fast"]
    slow = macd_params["slow"]
    signal = macd_params["signal"]

    # 短期EMAと長期EMA
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    # MACD
    macd = ema_fast - ema_slow
    features['macd'] = macd

    # シグナル線
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    features['macd_signal'] = macd_signal

    # MACD ヒストグラム
    features['macd_hist'] = macd - macd_signal

    # MACDの変化率
    features['macd_change'] = macd.diff(1)

    # MACDのクロスシグナル
    features['macd_cross'] = np.where(
        (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)),
        1,  # 上向きクロス（買いシグナル）
        np.where(
            (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)),
            -1,  # 下向きクロス（売りシグナル）
            0  # クロスなし
        )
    )

    # MACDのゼロラインクロス
    features['macd_zero_cross'] = np.where(
        (macd > 0) & (macd.shift(1) <= 0),
        1,  # 上向きクロス
        np.where(
            (macd < 0) & (macd.shift(1) >= 0),
            -1,  # 下向きクロス
            0  # クロスなし
        )
    )

    # MACDヒストグラムの傾き
    features['macd_hist_slope'] = features['macd_hist'].diff(3)

    return pd.DataFrame(features, index=df.index)

def generate_stochastic_features(df: pd.DataFrame, stochastic_params: dict) -> pd.DataFrame:
    """
    ストキャスティクスに関する特徴量を生成
    
    Args:
        df: 入力データフレーム
        stochastic_params: ストキャスティクスのパラメータ辞書
        
    Returns:
        DataFrame: 特徴量を含むデータフレーム
    """
    logger.info("ストキャスティクスの特徴量を追加しています")

    features = {}
    k_period = stochastic_params["k"]
    d_period = stochastic_params["d"]
    slowing = stochastic_params["slowing"]

    # 期間内の最高値・最安値
    high_roll = df['high'].rolling(window=k_period).max()
    low_roll = df['low'].rolling(window=k_period).min()

    # %K（Fast）計算
    fast_k = 100 * ((df['close'] - low_roll) / (high_roll - low_roll))

    # %K（Slow）計算
    stoch_k = fast_k.rolling(window=slowing).mean()
    features['stoch_k'] = stoch_k

    # %D計算
    stoch_d = stoch_k.rolling(window=d_period).mean()
    features['stoch_d'] = stoch_d

    # ストキャスティクスの変化率
    features['stoch_k_change'] = stoch_k.diff(1)
    features['stoch_d_change'] = stoch_d.diff(1)

    # 過買い/過売りシグナル
    features['stoch_overbought'] = ((stoch_k > 80) & (stoch_d > 80)).astype(int)
    features['stoch_oversold'] = ((stoch_k < 20) & (stoch_d < 20)).astype(int)

    # クロスシグナル
    features['stoch_cross'] = np.where(
        (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)),
        1,  # 上向きクロス（買いシグナル）
        np.where(
            (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)),
            -1,  # 下向きクロス（売りシグナル）
            0  # クロスなし
        )
    )

    # ストキャスティクスの位置（0〜100の範囲での現在位置）
    features['stoch_position'] = (stoch_k + stoch_d) / 2

    return pd.DataFrame(features, index=df.index)
