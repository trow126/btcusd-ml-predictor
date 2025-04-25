# data_processor/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_engineering")

class FeatureGenerator:
    def __init__(self, config=None):
        """
        特徴量生成クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        self.config = config if config else self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "input_dir": "data/raw",
            "input_filename": "btcusd_5m_data.csv",
            "output_dir": "data/processed",
            "output_filename": "btcusd_5m_features.csv",
            "features": {
                "price_change": True,         # 価格変動率
                "volume_change": True,        # 出来高変動率
                "moving_averages": True,      # 移動平均線
                "rsi": True,                  # RSI
                "high_low_distance": True,    # 高値/安値からの距離
                "bollinger_bands": True,      # ボリンジャーバンド
                "macd": True,                 # MACD
                "stochastic": True           # ストキャスティクス
            },
            "ma_periods": [5, 10, 20, 50, 100, 200],  # 移動平均の期間
            "rsi_period": 14,                         # RSIの期間
            "bollinger_period": 20,                   # ボリンジャーバンドの期間
            "bollinger_std": 2,                       # ボリンジャーバンドの標準偏差
            "macd_params": {"fast": 12, "slow": 26, "signal": 9},  # MACDのパラメータ
            "stochastic_params": {"k": 14, "d": 3, "slowing": 3},  # ストキャスティクスのパラメータ
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
        特徴量を生成

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

        # 価格変動率
        if self.config["features"]["price_change"]:
            feature_dfs.append(self._generate_price_change_features(result_df))

        # 出来高変動率
        if self.config["features"]["volume_change"]:
            feature_dfs.append(self._generate_volume_change_features(result_df))

        # 移動平均線と乖離率
        if self.config["features"]["moving_averages"]:
            feature_dfs.append(self._generate_moving_average_features(result_df))

        # RSI
        if self.config["features"]["rsi"]:
            feature_dfs.append(self._generate_rsi_features(result_df))

        # 高値/安値からの距離
        if self.config["features"]["high_low_distance"]:
            feature_dfs.append(self._generate_high_low_distance_features(result_df))

        # ボリンジャーバンド
        if self.config["features"]["bollinger_bands"]:
            feature_dfs.append(self._generate_bollinger_bands_features(result_df))

        # MACD
        if self.config["features"]["macd"]:
            feature_dfs.append(self._generate_macd_features(result_df))

        # ストキャスティクス
        if self.config["features"]["stochastic"]:
            feature_dfs.append(self._generate_stochastic_features(result_df))

        # 目標変数（ターゲット）の生成
        feature_dfs.append(self._generate_target_features(result_df))

        # 全ての特徴量を結合
        result_df = pd.concat(feature_dfs, axis=1)

        # 重複列を削除（念のため）
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        # NaNを含む行の処理（最初の数百行はNaNが含まれる可能性がある）
        result_df = result_df.dropna()

        logger.info(f"特徴量を生成しました。カラム数: {len(result_df.columns)}, 行数: {len(result_df)}")

        return result_df

    def _generate_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格変動率に関する特徴量を生成"""
        logger.info("価格変動率の特徴量を追加しています")

        features = {}

        # 価格変動率（現在の終値 / 過去の終値 - 1）
        for i in range(1, 21):  # 過去1〜20期間
            features[f'price_change_pct_{i}'] = df['close'].pct_change(i)

        # ローソク足の形状
        features['candle_size'] = (df['high'] - df['low']) / df['open']  # ローソク足の大きさ
        features['body_size'] = abs(df['close'] - df['open']) / df['open']  # 実体部分の大きさ
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']  # 上ヒゲ
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']  # 下ヒゲ
        features['is_bullish'] = (df['close'] > df['open']).astype(int)  # 陽線=1, 陰線=0

        return pd.DataFrame(features, index=df.index)

    def _generate_volume_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高変動率に関する特徴量を生成"""
        logger.info("出来高変動率の特徴量を追加しています")

        features = {}

        # 出来高変動率
        for i in range(1, 11):  # 過去1〜10期間
            features[f'volume_change_pct_{i}'] = df['volume'].pct_change(i)

        # 出来高の移動平均
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()

        # 出来高比率（現在の出来高 / 過去N期間の平均出来高）
        for period in [5, 10, 20]:
            vol_sma = df['volume'].rolling(window=period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / vol_sma

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

        # 移動平均線のゴールデンクロス/デッドクロスシグナル
        features['sma_5_10_cross'] = np.where(
            (sma_dict[5] > sma_dict[10]) & (sma_dict[5].shift(1) <= sma_dict[10].shift(1)),
            1,  # ゴールデンクロス
            np.where(
                (sma_dict[5] < sma_dict[10]) & (sma_dict[5].shift(1) >= sma_dict[10].shift(1)),
                -1,  # デッドクロス
                0  # クロスなし
            )
        )

        features['sma_10_20_cross'] = np.where(
            (sma_dict[10] > sma_dict[20]) & (sma_dict[10].shift(1) <= sma_dict[20].shift(1)),
            1,  # ゴールデンクロス
            np.where(
                (sma_dict[10] < sma_dict[20]) & (sma_dict[10].shift(1) >= sma_dict[20].shift(1)),
                -1,  # デッドクロス
                0  # クロスなし
            )
        )

        return pd.DataFrame(features, index=df.index)

    def _generate_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSIに関する特徴量を生成"""
        logger.info("RSIの特徴量を追加しています")

        features = {}
        period = self.config["rsi_period"]

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
        features['rsi'] = rsi

        # RSIの過買い/過売りシグナル
        features['rsi_overbought'] = (rsi > 70).astype(int)
        features['rsi_oversold'] = (rsi < 30).astype(int)

        # RSIの変化率
        features['rsi_change_1'] = rsi.diff(1)
        features['rsi_change_3'] = rsi.diff(3)

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

            features[f'highest_{period}'] = highest
            features[f'lowest_{period}'] = lowest

            # 現在値と最高値/最安値との距離（パーセンテージ）
            features[f'dist_from_high_{period}'] = (df['close'] - highest) / highest
            features[f'dist_from_low_{period}'] = (df['close'] - lowest) / lowest

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

        # 価格位置（上限と下限の間での相対位置、0〜1）
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # シグナル（バンドからの超過状態）
        features['bb_upper_exceed'] = (df['close'] > bb_upper).astype(int)
        features['bb_lower_exceed'] = (df['close'] < bb_lower).astype(int)

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

        features['ema_fast'] = ema_fast
        features['ema_slow'] = ema_slow

        # MACD
        macd = ema_fast - ema_slow
        features['macd'] = macd

        # シグナル線
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        features['macd_signal'] = macd_signal

        # MACD ヒストグラム
        features['macd_hist'] = macd - macd_signal

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

            # 価格変動カテゴリ（分類用）
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
        feature_counts = {
            "price_change": len([c for c in df.columns if c.startswith('price_change')]),
            "volume_change": len([c for c in df.columns if c.startswith('volume')]),
            "moving_averages": len([c for c in df.columns if c.startswith('sma') or c.startswith('ema')]),
            "rsi": len([c for c in df.columns if c.startswith('rsi')]),
            "high_low_distance": len([c for c in df.columns if c.startswith('dist_from')]),
            "bollinger_bands": len([c for c in df.columns if c.startswith('bb_')]),
            "macd": len([c for c in df.columns if c.startswith('macd')]),
            "stochastic": len([c for c in df.columns if c.startswith('stoch')]),
            "target": len([c for c in df.columns if c.startswith('target')])
        }

        # 基本統計情報
        report = {
            "status": "success",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "feature_counts": feature_counts,
            "start_date": df.index.min().isoformat(),
            "end_date": df.index.max().isoformat(),
            "missing_values": df.isna().sum().sum(),
            "target_distribution": {
                f"target_price_direction_{period}": df[f'target_price_direction_{period}'].value_counts().to_dict()
                for period in self.config["target_periods"]
            },
            "sample_data": {
                "first_5_rows": df.head(5),
                "last_5_rows": df.tail(5)
            }
        }

        return report


# 実行部分（外部から呼び出す場合）
def generate_features(config=None):
    """
    特徴量を生成して保存する関数

    Args:
        config: 設定辞書またはNone（デフォルト設定を使用）

    Returns:
        Tuple[pd.DataFrame, Dict]: (特徴量DataFrame, レポート)
    """
    generator = FeatureGenerator(config)

    # データ読み込み
    df = generator.load_data()

    # 特徴量生成
    feature_df = generator.generate_features(df)

    # 特徴量保存
    generator.save_features(feature_df)

    # レポート生成
    report = generator.generate_feature_report(feature_df)

    return feature_df, report