# data_collector/bybit_collector.py
import asyncio
import pandas as pd
import datetime as dt
from pathlib import Path
import logging
import json
import time
import requests
import pybotters
from typing import Dict, Any, Optional
import aiohttp
import asyncio
from tqdm.notebook import tqdm  # Jupyter用プログレスバー


# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bybit_collector")

class BTCDataCollector:
    def __init__(self, config=None):
        """
        BTCUSDデータ収集クラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        self.config = config if config else self._get_default_config()
        self.client = None

    def _get_default_config(self):
        """デフォルト設定を返す"""
        return {
            "exchange": "bybit",
            "symbol": "BTCUSDT",  # Bybitでの表記形式
            "timeframe": "5",     # 5分足（Bybitでは数字のみ）
            "start_date": dt.datetime(2023, 1, 1),
            "end_date": dt.datetime.now(),
            "output_dir": "data/raw",
            "output_filename": "btcusd_5m_data.csv",
            "api_keys": {
                "bybit": ["YOUR_API_KEY", "YOUR_API_SECRET"]
            },
            "use_direct_api": True  # True: 直接APIを使用, False: pybottersを使用
        }

    async def setup_client(self):
        """APIクライアントをセットアップ"""
        if not self.config["use_direct_api"]:
            # pybottersクライアントを使用
            apis = {
                self.config["exchange"]: self.config["api_keys"].get(self.config["exchange"], ["", ""])
            }
            self.client = pybotters.Client(apis=apis)
            logger.info(f"pybottersクライアントをセットアップしました")
        logger.info(f"{self.config['exchange']}への接続を準備しました")

    def fetch_bybit_klines_direct(self, start_time=None, end_time=None, limit=1000):
        """
        Bybit APIを直接使用してKラインデータを取得

        Args:
            start_time: 開始時間（Unix timestamp）
            end_time: 終了時間（Unix timestamp）
            limit: 取得する最大数（最大200）

        Returns:
            DataFrameまたはNone
        """
        url = "https://api.bybit.com/v5/market/kline"

        # パラメータの準備
        params = {
            "category": "linear",
            "symbol": self.config["symbol"],
            "interval": self.config["timeframe"],
            "limit": limit
        }

        if start_time:
            params["start"] = int(start_time * 1000)  # ミリ秒に変換
        if end_time:
            params["end"] = int(end_time * 1000)      # ミリ秒に変換

        try:
            # APIリクエスト送信
            response = requests.get(url, params=params)
            response.raise_for_status()  # エラー時に例外を発生

            # レスポンスを解析
            result = response.json()

            if result["retCode"] != 0:
                logger.error(f"APIエラー: {result['retMsg']}")
                return None

            # データが存在するか確認
            if not result["result"]["list"]:
                logger.info("データがありませんでした")
                return None

            # DataFrameに変換
            df = pd.DataFrame(
                result["result"]["list"],
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
            )

            # データ型変換
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col])

            # 時間でソート
            df.sort_values("timestamp", inplace=True)

            return df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            return None

    async def fetch_bybit_klines_pybotters(self, start_time=None, end_time=None, limit=1000):
        """
        pybottersを使用してBybitからKラインデータを取得

        Args:
            start_time: 開始時間（Unix timestamp）
            end_time: 終了時間（Unix timestamp）
            limit: 取得する最大数（最大200）

        Returns:
            DataFrameまたはNone
        """
        if not self.client:
            await self.setup_client()

        # パラメータの準備
        params = {
            "category": "linear",
            "symbol": self.config["symbol"],
            "interval": self.config["timeframe"],
            "limit": str(limit)
        }

        if start_time:
            params["start"] = str(int(start_time * 1000))  # ミリ秒に変換
        if end_time:
            params["end"] = str(int(end_time * 1000))      # ミリ秒に変換

        try:
            # pybottersでAPIリクエスト
            resp = await self.client.fetch(
                "GET",
                "https://api.bybit.com/v5/market/kline",
                params=params
            )

            if resp.status != 200:
                logger.error(f"APIエラー: ステータスコード {resp.status}")
                return None

            result = resp.data

            if result["retCode"] != 0:
                logger.error(f"APIエラー: {result['retMsg']}")
                return None

            # データが存在するか確認
            if not result["result"]["list"]:
                logger.info("データがありませんでした")
                return None

            # DataFrameに変換
            df = pd.DataFrame(
                result["result"]["list"],
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
            )

            # データ型変換
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col])

            # 時間でソート
            df.sort_values("timestamp", inplace=True)

            return df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            return None

    # bybit_collector.py内の collect_historical_data メソッドを修正



    async def collect_historical_data(self):
        """
        履歴データを収集（並列処理で高速化）

        Returns:
            DataFrame: 収集したデータ
        """
        logger.info(f"{self.config['symbol']}の{self.config['timeframe']}分足データを{self.config['start_date']}から{self.config['end_date']}まで収集します")

        start_time = self.config["start_date"].timestamp()
        end_time = self.config["end_date"].timestamp()

        # 時間範囲を分割（並列処理用）
        time_chunks = []
        current_start = start_time

        # 1日ごとに分割 (Bybitの場合は1日単位で分割すると効率的)
        chunk_size = 86400  # 24時間 = 86400秒

        while current_start < end_time:
            chunk_end = min(current_start + chunk_size, end_time)
            time_chunks.append((current_start, chunk_end))
            current_start = chunk_end

        logger.info(f"データ取得を{len(time_chunks)}個のチャンクに分割しました")

        # 非同期セッションを作成（接続プールを共有）
        async with aiohttp.ClientSession() as session:
            # 並列でデータを取得
            tasks = []
            semaphore = asyncio.Semaphore(5)  # 同時に実行するリクエスト数を制限

            async def fetch_chunk(start, end):
                async with semaphore:
                    if self.config["use_direct_api"]:
                        return await self.fetch_bybit_klines_aiohttp(start, end, session)
                    else:
                        return await self.fetch_bybit_klines_pybotters(start, end)

            for i, (chunk_start, chunk_end) in enumerate(time_chunks):
                tasks.append(fetch_chunk(chunk_start, chunk_end))

            # tqdmでプログレスバーを表示しながら並列処理
            results = []
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="データ取得"):
                df = await future
                if df is not None and not df.empty:
                    results.append(df)

        # 取得したデータを結合
        if not results:
            logger.warning("データを取得できませんでした")
            return pd.DataFrame()

        result_df = pd.concat(results)

        # 重複を削除
        result_df = result_df.drop_duplicates(subset="timestamp")

        # インデックスを設定
        result_df.set_index("timestamp", inplace=True)
        result_df.sort_index(inplace=True)

        logger.info(f"データ収集完了。合計 {len(result_df)} 行のデータを取得")

        return result_df

    # aiohttp を使用した高速取得メソッドを追加
    async def fetch_bybit_klines_aiohttp(self, start_time=None, end_time=None, session=None, limit=200):
        """
        aiohttpを使用してBybit APIからKラインデータを高速取得

        Args:
            start_time: 開始時間（Unix timestamp）
            end_time: 終了時間（Unix timestamp）
            session: 再利用するaiohttp.ClientSession
            limit: 取得する最大数（最大200）

        Returns:
            DataFrameまたはNone
        """
        url = "https://api.bybit.com/v5/market/kline"

        # パラメータの準備
        params = {
            "category": "linear",
            "symbol": self.config["symbol"],
            "interval": self.config["timeframe"],
            "limit": limit
        }

        if start_time:
            params["start"] = int(start_time * 1000)  # ミリ秒に変換
        if end_time:
            params["end"] = int(end_time * 1000)      # ミリ秒に変換

        try:
            # セッションが提供されていない場合は新しく作成
            if session is None:
                async with aiohttp.ClientSession() as new_session:
                    async with new_session.get(url, params=params) as response:
                        if response.status != 200:
                            logger.error(f"APIエラー: ステータスコード {response.status}")
                            return None
                        result = await response.json()
            else:
                # 提供されたセッションを使用
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"APIエラー: ステータスコード {response.status}")
                        return None
                    result = await response.json()

            if result["retCode"] != 0:
                logger.error(f"APIエラー: {result['retMsg']}")
                return None

            # データが存在するか確認
            if not result["result"]["list"]:
                return None

            # DataFrameに変換
            df = pd.DataFrame(
                result["result"]["list"],
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
            )

            # データ型変換（警告を回避）
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col])

            return df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            return None

    def save_data(self, df):
        """
        データをCSVファイルに保存

        Args:
            df: 保存するDataFrame

        Returns:
            dict: 保存結果のサマリー
        """
        if df.empty:
            logger.warning("保存するデータがありません")
            return {"status": "error", "message": "データがありません"}

        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # データをCSVに保存
        output_path = output_dir / self.config["output_filename"]
        df.to_csv(output_path)
        logger.info(f"データを {output_path} に保存しました。データサイズ: {df.shape}")

        # 結果のサマリーを返す（数値情報）
        summary = {
            "status": "success",
            "rows": df.shape[0],
            "columns": df.shape[1],
            "start_date": df.index.min().isoformat() if not df.empty else None,
            "end_date": df.index.max().isoformat() if not df.empty else None,
            "missing_values": df.isna().sum().to_dict(),
            "data_stats": {
                "open": {
                    "min": df["open"].min() if not df.empty else None,
                    "max": df["open"].max() if not df.empty else None,
                    "mean": df["open"].mean() if not df.empty else None
                },
                "close": {
                    "min": df["close"].min() if not df.empty else None,
                    "max": df["close"].max() if not df.empty else None,
                    "mean": df["close"].mean() if not df.empty else None
                },
                "volume": {
                    "total": df["volume"].sum() if not df.empty else None,
                    "mean": df["volume"].mean() if not df.empty else None
                }
            }
        }
        return summary