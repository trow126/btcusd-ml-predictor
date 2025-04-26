# data_processor/feature_engineering_optimized.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import talib

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_engineering_optimized")

class OptimizedFeatureGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        最適化された特徴量生成クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        # デフォルト設定をロード
        self.config = self._get_default_config()

        # 渡された設定でデフォルト設定を上書き（マージ）
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                    # 辞書の場合は再帰的にマージ
                    self.config[key].update(value)
                else:
                    self.config[key] = value

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "input_dir": "data/raw",
            "input_filename": "btcusd_5m_data.csv",
            "output_dir": "data/processed",
            "output_filename": "btcusd_5m_optimized_features.csv",
            "features": {
                "price_change": True,         # 価格変動率
                "volume_change": True,        # 出来高変動率
                "moving_averages": True,      # 移動平均線
                "rsi": True,                  # RSI
                "high_low_distance": True,    # 高値/安値からの距離
                "bollinger_bands": True,      # ボリンジャーバンド
                "macd": True,                 # MACD
                "stochastic": True,           # ストキャスティクス
                "advanced_features": True     # 高度な特徴量（新規追加）
            },
            "ma_periods": [5, 10, 20, 50, 100, 200],  # 移動平均の期間
            "rsi_periods": [6, 14, 24],               # RSIの期間（複数）
            "bollinger_period": 20,                   # ボリンジャーバンドの期間
            "bollinger_std": 2,                       # ボリンジャーバンドの標準偏差
            "macd_params": {"fast": 12, "slow": 26, "signal": 9},  # MACDのパラメータ
            "stochastic_params": {"k": 14, "d": 3, "slowing": 3},  # ストキャスティクスのパラメータ
            "atr_period": 14,                         # ATRの期間
            "vwap_period": 14,                        # VWAPの期間
            "target_periods": [1, 2, 3],              # 予測対象（5分後=1, 10分後=2, 15分後=3）
            "classification_threshold": 0.0005        # 分類閾値（±0.05%）
        }

    def load_data(self) -> pd.DataFrame:
        """
        生データを読み込む

        Returns:
            DataFrame: 読み込んだデータ
        """
        input_path = Path(self.config["input_dir"]) / self.config["input_filename"]
        logger.info(f"データを {input_path} から読み込みます")

        try:
            df = pd.read_csv(input_path, index_col="timestamp", parse_dates=True)
            logger.info(f"{len(df)} 行のデータを読み込みました")
            return df
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return pd.DataFrame()

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        最適化された特徴量を生成

        Args:
            df: 入力データフレーム

        Returns:
            DataFrame: 特徴量を追加したデータフレーム
        """
        if df.empty:
            logger.warning("入力データが空です")
            return df

        # 処理前のデータをコピー
        result_df = df.copy()

        # 全ての特徴量を生成
        feature_dfs = []

        # 基本データは常に保持
        feature_dfs.append(result_df)

        # 価格変動率と重要なローソク足パターン
        if self.config["features"]["price_change"]:
            feature_dfs.append(self._generate_price_change_features(result_df))

        # 出来高変動率
        if self.config["features"]["volume_change"]:
            feature_dfs.append(self._generate_volume_change_features(result_df))

        # 移動平均線と乖離率
        if self.config["features"]["moving_averages"]:
            feature_dfs.append(self._generate_moving_average_features(result_df))

        # RSI（複数期間）
        if self.config["features"]["rsi"]:
            feature_dfs.append(self._generate_rsi_features(result_df))

        # 高値/安値からの距離
        if self.config["features"]["high_low_distance"]:
            feature_dfs.append(self._generate_high_low_distance_features(result_df))

        # ボリンジャーバンドと関連指標
        if self.config["features"]["bollinger_bands"]:
            feature_dfs.append(self._generate_bollinger_bands_features(result_df))

        # MACD
        if self.config["features"]["macd"]:
            feature_dfs.append(self._generate_macd_features(result_df))

        # ストキャスティクス
        if self.config["features"]["stochastic"]:
            feature_dfs.append(self._generate_stochastic_features(result_df))

        # 高度な特徴量（ATR, VWAP, OBV, 重要レベルとの関連など）
        if self.config["features"]["advanced_features"]:
            feature_dfs.append(self._generate_advanced_features(result_df))

        # 目標変数（ターゲット）の生成
        feature_dfs.append(self._generate_target_features(result_df))

        # 全ての特徴量を結合
        result_df = pd.concat(feature_dfs, axis=1)

        # 重複列を削除
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        # NaNを含む行の処理（最初の数百行はNaNが含まれる可能性がある）
        result_df = result_df.dropna()

        # NaNを含む行の処理（最初の数百行はNaNが含まれる可能性がある）
        result_df = result_df.dropna()

        # フィッシャー変換RSI（RSIをより線形に変換）を結合後に計算
        if 'rsi_14' in result_df.columns:
            rsi = result_df['rsi_14']
            # 0.1〜0.9の範囲に変換（極端な値を避ける）
            rsi_scaled = 0.1 + 0.8 * (rsi / 100)
            result_df['fisher_transform_rsi'] = 0.5 * np.log((1 + rsi_scaled) / (1 - rsi_scaled))
            logger.info("フィッシャー変換RSIの特徴量を追加しました")
        else:
            logger.warning("rsi_14 が存在しないため、フィッシャー変換RSIは生成されませんでした")


        # 特徴量の重要度に基づいて選別
        result_df = self._select_important_features(result_df)

        logger.info(f"特徴量を生成しました。カラム数: {len(result_df.columns)}, 行数: {len(result_df)}")

        return result_df

    def _generate_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格変動率と重要なローソク足パターンに関する特徴量を生成"""
        logger.info("価格変動率とローソク足パターンの特徴量を追加しています")

        features = {}

        # 価格変動率（現在の終値 / 過去の終値 - 1）
        for i in range(1, 11):  # 過去1〜10期間に絞る
            features[f'price_change_pct_{i}'] = df['close'].pct_change(i)

        # ローソク足の重要な形状指標
        features['candle_size'] = (df['high'] - df['low']) / df['open']  # ローソク足の大きさ
        features['body_size'] = abs(df['close'] - df['open']) / df['open']  # 実体部分の大きさ
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']  # 上ヒゲ
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']  # 下ヒゲ
        features['is_bullish'] = (df['close'] > df['open']).astype(int)  # 陽線=1, 陰線=0

        # ハイローの比率
        features['high_low_ratio'] = df['high'] / df['low']

        # 価格振れ幅（当日の変動率）
        features['price_range_pct'] = (df['high'] - df['low']) / df['open']

        # ギャップアップ/ダウン（5分足では重要度は低いが、一応）
        features['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).clip(lower=0)
        features['gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1)).clip(lower=0)

        # 出来高加重平均価格（VWAP）
        features['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).rolling(window=self.config["vwap_period"]).sum() / df['volume'].rolling(window=self.config["vwap_period"]).sum()

        # 価格モメンタム（現在値と過去平均値の差）
        for period in [5, 10, 20]:
            features[f'price_momentum_{period}'] = df['close'] - df['close'].rolling(window=period).mean()

        return pd.DataFrame(features, index=df.index)

    def _generate_volume_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高変動率に関する特徴量を生成"""
        logger.info("出来高変動率の特徴量を追加しています")

        features = {}

        # 出来高変動率
        for i in range(1, 6):  # 過去1〜5期間に絞る
            features[f'volume_change_pct_{i}'] = df['volume'].pct_change(i)

        # 出来高の移動平均
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()

        # 出来高比率（現在の出来高 / 過去N期間の平均出来高）
        for period in [5, 10, 20]:
            vol_sma = df['volume'].rolling(window=period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / vol_sma

        # 出来高の標準偏差（ボラティリティ指標）
        for period in [5, 10, 20]:
            features[f'volume_std_{period}'] = df['volume'].rolling(window=period).std()

        # 出来高モメンタム
        features['volume_momentum_5'] = df['volume'] - df['volume'].rolling(window=5).mean()

        # On-Balance Volume (OBV) - 価格変動の方向に基づいて累積される出来高
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # 価格変動と出来高の関連性
        features['price_volume_correlation'] = df['close'].rolling(window=10).corr(df['volume'])

        return pd.DataFrame(features, index=df.index)

    def _generate_moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """移動平均線と乖離率に関する特徴量を生成"""
        logger.info("移動平均線と乖離率の特徴量を追加しています")

        features = {}

        # 単純移動平均線
        sma_dict = {}
        for period in self.config["ma_periods"]:
            sma = df['close'].rolling(window=period).mean()
            features[f'sma_{period}'] = sma
            sma_dict[period] = sma

            # 乖離率（現在値が移動平均線からどれだけ離れているか）
            features[f'sma_diff_pct_{period}'] = (df['close'] - sma) / sma

        # 指数移動平均線
        ema_dict = {}
        for period in [5, 10, 20, 50]:
            ema = df['close'].ewm(span=period, adjust=False).mean()
            features[f'ema_{period}'] = ema
            ema_dict[period] = ema

        # 重み付け移動平均線（WMA）- 近い日付ほど重みが大きい
        for period in [5, 20]:
            weights = np.arange(1, period + 1)
            features[f'wma_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )

        # 移動平均線のクロスシグナル
        features['sma_5_20_cross'] = np.where(
            (sma_dict[5] > sma_dict[20]) & (sma_dict[5].shift(1) <= sma_dict[20].shift(1)),
            1,  # ゴールデンクロス
            np.where(
                (sma_dict[5] < sma_dict[20]) & (sma_dict[5].shift(1) >= sma_dict[20].shift(1)),
                -1,  # デッドクロス
                0  # クロスなし
            )
        )

        # 移動平均線の傾き
        for period in [5, 20, 50]:
            features[f'sma_{period}_slope'] = sma_dict[period].diff(5) / sma_dict[period].shift(5)

        # 複数の移動平均線の位置関係（トレンド指標）
        features['ma_trend'] = np.where(
            (sma_dict[5] > sma_dict[20]) & (sma_dict[20] > sma_dict[50]),
            1,  # 強い上昇トレンド
            np.where(
                (sma_dict[5] < sma_dict[20]) & (sma_dict[20] < sma_dict[50]),
                -1,  # 強い下降トレンド
                0  # 明確なトレンドなし
            )
        )

        return pd.DataFrame(features, index=df.index)

    def _generate_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSIに関する特徴量を生成"""
        logger.info("RSIの特徴量を追加しています")

        features = {}

        # 複数期間のRSI
        for period in self.config["rsi_periods"]:
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

    def _generate_high_low_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高値/安値からの距離に関する特徴量を生成"""
        logger.info("高値/安値からの距離の特徴量を追加しています")

        features = {}

        # 期間ごとの最高値/最安値からの距離
        for period in [5, 10, 20, 50]:
            # 過去N期間の最高値と最安値
            highest = df['high'].rolling(window=period).max()
            lowest = df['low'].rolling(window=period).min()

            # 現在値と最高値/最安値との距離（パーセンテージ）
            features[f'dist_from_high_{period}'] = (df['close'] - highest) / highest
            features[f'dist_from_low_{period}'] = (df['close'] - lowest) / lowest

            # 価格が期間の高値・安値圏にあるかどうか
            features[f'near_high_{period}'] = (features[f'dist_from_high_{period}'] > -0.005).astype(int)
            features[f'near_low_{period}'] = (features[f'dist_from_low_{period}'] < 0.005).astype(int)

        # 複数期間（5, 20, 50）の高値・安値範囲内での相対位置（0～1）
        for period in [5, 20, 50]:
            highest = df['high'].rolling(window=period).max()
            lowest = df['low'].rolling(window=period).min()
            features[f'price_position_{period}'] = (df['close'] - lowest) / (highest - lowest)

        return pd.DataFrame(features, index=df.index)

    def _generate_bollinger_bands_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボリンジャーバンドに関する特徴量を生成"""
        logger.info("ボリンジャーバンドの特徴量を追加しています")

        features = {}
        period = self.config["bollinger_period"]
        std_dev = self.config["bollinger_std"]

        # 移動平均
        bb_ma = df['close'].rolling(window=period).mean()
        features['bb_ma'] = bb_ma

        # 標準偏差
        bb_std = df['close'].rolling(window=period).std()
        features['bb_std'] = bb_std

        # ボリンジャーバンド（上限、下限）
        bb_upper = bb_ma + (bb_std * std_dev)
        bb_lower = bb_ma - (bb_std * std_dev)

        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower

        # バンド幅（ボラティリティ指標）
        features['bb_width'] = (bb_upper - bb_lower) / bb_ma

        # バンド幅の変化率
        features['bb_width_change'] = features['bb_width'].pct_change(1)

        # バンド幅の拡大/縮小シグナル
        features['bb_width_expanding'] = (features['bb_width'] > features['bb_width'].shift(5)).astype(int)

        # 価格位置（上限と下限の間での相対位置、0〜1）
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # シグナル（バンドからの超過状態）
        features['bb_upper_exceed'] = (df['close'] > bb_upper).astype(int)
        features['bb_lower_exceed'] = (df['close'] < bb_lower).astype(int)

        # ボリンジャースクイーズ（バンド幅が狭まり、大きな動きの前兆）
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(window=20).quantile(0.2)).astype(int)

        return pd.DataFrame(features, index=df.index)

    def _generate_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACDに関する特徴量を生成"""
        logger.info("MACDの特徴量を追加しています")

        features = {}
        fast = self.config["macd_params"]["fast"]
        slow = self.config["macd_params"]["slow"]
        signal = self.config["macd_params"]["signal"]

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

    def _generate_stochastic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ストキャスティクスに関する特徴量を生成"""
        logger.info("ストキャスティクスの特徴量を追加しています")

        features = {}
        k_period = self.config["stochastic_params"]["k"]
        d_period = self.config["stochastic_params"]["d"]
        slowing = self.config["stochastic_params"]["slowing"]

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

    def _generate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量を生成"""
        logger.info("高度な特徴量を追加しています")

        features = {}

        # ATR（Average True Range）- ボラティリティ指標
        atr_period = self.config["atr_period"]
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        features['atr'] = tr.rolling(window=atr_period).mean()

        # ATRの変化率
        features['atr_change'] = features['atr'].pct_change(3)

        # ATRに対する価格変動の比率（ボラティリティ調整済み価格変動）
        features['price_change_to_atr'] = df['close'].pct_change(1) / features['atr']

        # ボラティリティの加速/減速
        features['volatility_change'] = features['atr'].diff(5) / features['atr'].shift(5)

        # 価格とボリュームの関係性指標
        features['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).rolling(window=10).sum()

        # 価格の短期勢い（モメンタム）
        features['price_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

        # 価格の加速度（モメンタムの変化率）
        features['price_acceleration'] = features['price_momentum'] - features['price_momentum'].shift(1)

        # 過去N期間の価格トレンド強度（1に近いほど強いトレンド）
        for period in [5, 10, 20]:
            x = np.arange(period)
            features[f'trend_strength_{period}'] = df['close'].rolling(window=period).apply(
                lambda y: np.abs(np.corrcoef(x, y)[0, 1]), raw=True
            )

        # 市場効率係数（Market Efficiency Coefficient）
        # 1に近いほど効率的な市場（ランダムウォーク）、0に近いほど非効率（トレンド/逆行）
        for period in [10, 20]:
            features[f'market_efficiency_{period}'] = abs(df['close'].diff(period)) / (
                df['close'].diff().abs().rolling(window=period).sum()
            )

        # ボラティリティに調整された価格変動指標
        for period in [5, 10]:
            returns = df['close'].pct_change(1)
            vol = returns.rolling(window=period).std()
            features[f'vol_adjusted_change_{period}'] = returns / vol

        # 価格と移動平均のゴールデンクロス/デッドクロスの距離
        ma_20 = df['close'].rolling(window=20).mean()
        features['price_ma_20_distance'] = (df['close'] - ma_20) / ma_20

        # 暴落/急騰検出
        returns = df['close'].pct_change()
        std_returns = returns.rolling(window=20).std()
        features['price_shock'] = returns / std_returns

        # 移動平均収束/発散指標
        features['ma_convergence'] = (
            df['close'].rolling(window=10).mean() -
            df['close'].rolling(window=20).mean()
        ) / df['close']

        # ピボットポイント（前日の高値・安値・終値から計算される重要レベル）
        # 5分足では通常使わないが、日足データを取り込んで計算することも可能

        # HL2, HLC3, OHLC4（様々な価格平均）
        features['hl2'] = (df['high'] + df['low']) / 2
        features['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        features['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # トレンド強度指標（ADX風）- 直近の動きとトレンドの一致度
        features['trend_intensity'] = abs(df['close'].diff(10)) / df['close'].diff().abs().rolling(window=10).sum()

        return pd.DataFrame(features, index=df.index)

    def _generate_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """予測対象（目標変数）を生成"""
        logger.info("予測対象の目標変数を追加しています")

        features = {}
        threshold = self.config["classification_threshold"]

        # 各予測時間軸に対する目標変数を生成
        for period in self.config["target_periods"]:
            # 価格変動率（回帰用）
            target_change = df['close'].pct_change(periods=period).shift(-period)
            features[f'target_price_change_pct_{period}'] = target_change

            # 価格変動カテゴリ（3分類）
            features[f'target_price_direction_{period}'] = np.where(
                target_change > threshold,
                1,  # 上昇
                np.where(
                    target_change < -threshold,
                    -1,  # 下落
                    0   # 横ばい
                )
            )

        return pd.DataFrame(features, index=df.index)

    def _select_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        重要度の高い特徴量を選別

        Args:
            df: 特徴量を含むDataFrame

        Returns:
            DataFrame: 選別した特徴量を含むDataFrame
        """
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

    def save_features(self, df: pd.DataFrame) -> bool:
        """
        生成した特徴量をCSVファイルに保存

        Args:
            df: 保存するDataFrame

        Returns:
            bool: 保存が成功したかどうか
        """
        if df.empty:
            logger.warning("保存するデータがありません")
            return False

        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # データをCSVに保存
        output_path = output_dir / self.config["output_filename"]
        df.to_csv(output_path)
        logger.info(f"特徴量を {output_path} に保存しました。データサイズ: {df.shape}")

        return True

    def generate_feature_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        特徴量の統計情報レポートを生成

        Args:
            df: 特徴量を含むDataFrame

        Returns:
            Dict: 特徴量の統計情報
        """
        if df.empty:
            return {"status": "error", "message": "データがありません"}

        # 特徴量群ごとの数
        feature_groups = {
            "price_related": len([c for c in df.columns if 'price' in c or c in ['open', 'high', 'low', 'close']]),
            "volume_related": len([c for c in df.columns if 'volume' in c or c == 'turnover']),
            "moving_average": len([c for c in df.columns if 'sma' in c or 'ema' in c]),
            "oscillator": len([c for c in df.columns if 'rsi' in c or 'stoch' in c or 'macd' in c]),
            "volatility": len([c for c in df.columns if 'atr' in c or 'bb_' in c]),
            "trend": len([c for c in df.columns if 'trend' in c or 'ma_' in c or 'momentum' in c]),
            "target": len([c for c in df.columns if c.startswith('target_')])
        }

        # 基本統計情報
        report = {
            "status": "success",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "feature_groups": feature_groups,
            "start_date": df.index.min().isoformat(),
            "end_date": df.index.max().isoformat(),
            "missing_values": df.isna().sum().sum(),
            "target_distribution": {
                f"target_price_direction_{period}": df[f'target_price_direction_{period}'].value_counts().to_dict()
                for period in self.config["target_periods"]
            },
            "feature_correlations": {
                f"period_{period}": {
                    feature: float(df[feature].corr(df[f'target_price_direction_{period}']))
                    for feature in df.columns if not feature.startswith('target_')
                }
                for period in self.config["target_periods"]
            },
            "sample_data": {
                "first_5_rows": df.head(5),
                "last_5_rows": df.tail(5)
            }
        }

        return report

# 実行部分（外部から呼び出す場合）
def generate_optimized_features(config=None):
    """
    最適化された特徴量を生成して保存する関数

    Args:
        config: 設定辞書またはNone（デフォルト設定を使用）

    Returns:
        Tuple[pd.DataFrame, Dict]: (特徴量DataFrame, レポート)
    """
    generator = OptimizedFeatureGenerator(config)

    # データ読み込み
    df = generator.load_data()

    # 特徴量生成
    feature_df = generator.generate_features(df)

    # 特徴量保存
    generator.save_features(feature_df)

    # レポート生成
    report = generator.generate_feature_report(feature_df)

    return feature_df, report