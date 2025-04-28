
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# データディレクトリの作成
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# 5分足のOHLCVデータを10日分生成（約2880レコード）
start_time = datetime(2023, 1, 1)
interval = timedelta(minutes=5)
records = 2880  # 約10日分

# 乱数生成関数
def random_walk(start_price, volatility, num_records):
    """ランダムウォークで価格系列を生成"""
    price = start_price
    timestamps = [start_time + i * interval for i in range(num_records)]
    
    prices = []
    for _ in range(num_records):
        # ランダムな価格変動を生成（正規分布に近似）
        change = (np.random.random() - 0.5) * 2 * volatility * price
        price += change
        
        # 価格が0以下にならないようにする
        price = max(price, 0.01)
        
        prices.append(price)
    
    return timestamps, prices

# ビットコイン価格のサンプルデータ生成（$30,000から）
timestamps, prices = random_walk(30000, 0.005, records)  # 0.5%のボラティリティ

# OHLCV生成
data = []
for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
    open_price = price
    high_price = price * (1 + np.random.random() * 0.01)  # 最大1%高い
    low_price = price * (1 - np.random.random() * 0.01)   # 最大1%低い
    close_price = price * (1 + (np.random.random() - 0.5) * 0.015)  # ±1.5%の変動
    volume = np.random.random() * 100 + 10  # 擬似的な出来高
    turnover = volume * close_price  # 取引額
    
    data.append({
        "timestamp": timestamp,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "turnover": turnover
    })

# DataFrameに変換
df = pd.DataFrame(data)

# 保存
file_path = "data/raw/btcusd_5m_data.csv"
df.to_csv(file_path, index=False)

print(f"データファイルを生成しました: {file_path}")
print(f"レコード数: {len(df)}")
