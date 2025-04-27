# data_processor/feature_modules/feature_generator_base.py
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

class FeatureGeneratorBase:
    """特徴量生成の基本クラス"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        特徴量生成クラスの初期化

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
                "advanced_features": True     # 高度な特徴量
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
