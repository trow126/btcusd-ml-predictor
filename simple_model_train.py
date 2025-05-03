#!/usr/bin/env python
"""
シンプルなモデルトレーニングスクリプト
既存のCSVファイルを使用して直接モデルをトレーニングします
"""
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("simple_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simple_train")

def train_simple_models():
    """CSVファイルから直接モデルをトレーニング"""
    try:
        # ファイルパス設定
        features_path = Path("data/processed/btcusd_5m_features.csv")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        if not features_path.exists():
            logger.error(f"特徴量ファイルが見つかりません: {features_path}")
            return False
            
        # データ読み込み
        logger.info(f"ファイル読み込み: {features_path}")
        df = pd.read_csv(features_path)
        logger.info(f"読み込み成功: 行数={len(df)}, 列数={len(df.columns)}")
        
        # 特徴量と目標変数の確認
        target_cols = [col for col in df.columns if col.startswith("target_")]
        logger.info(f"目標変数: {len(target_cols)}個")
        
        # 必要な目標変数を確認
        binary_cols = [col for col in target_cols if col.startswith('target_binary_')]
        logger.info(f"binary変数: {len(binary_cols)}個")
        for col in binary_cols:
            value_counts = df[col].value_counts()
            logger.info(f"{col} の値分布: {value_counts.to_dict()}")
        
        # 特徴量の準備
        # 価格関連特徴量
        price_cols = [col for col in df.columns if (
            col.startswith("price_") or
            col in ["open", "high", "low", "close"] or
            "candle" in col or "body" in col
        )]
        
        # 出来高関連特徴量
        volume_cols = [col for col in df.columns if (
            "volume" in col or
            "turnover" in col
        )]
        
        # テクニカル指標関連特徴量
        technical_cols = [col for col in df.columns if (
            col.startswith("sma_") or
            col.startswith("ema_") or
            col.startswith("rsi") or
            col.startswith("bb_") or
            col.startswith("macd") or
            col.startswith("stoch_")
        )]
        
        # 特徴量を結合
        feature_cols = price_cols + volume_cols + technical_cols
        # 重複を削除
        feature_cols = list(set(feature_cols))
        # 目標変数を除外
        feature_cols = [col for col in feature_cols if not col.startswith("target_")]
        
        logger.info(f"使用する特徴量: {len(feature_cols)}個")
        
        # データをトレーニングとテストに分割
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        logger.info(f"トレーニングデータ: {len(df_train)}行, テストデータ: {len(df_test)}行")
        
        # 二値分類モデルのトレーニング
        for period in [1, 2, 3]:
            target_col = f'target_binary_{period}'
            if target_col in df.columns:
                logger.info(f"二値分類モデル（{period}期先）のトレーニングを開始")
                
                # 特徴量と目標変数
                X_train = df_train[feature_cols]
                y_train = df_train[target_col]
                X_test = df_test[feature_cols]
                y_test = df_test[target_col]
                
                # モデルトレーニング
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # モデル評価
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                logger.info(f"モデル評価: 精度={accuracy:.4f}, 適合率={precision:.4f}, 再現率={recall:.4f}, F1={f1:.4f}")
                
                # モデル保存
                model_path = models_dir / f"binary_classification_model_period_{period}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"モデルを保存しました: {model_path}")
            else:
                logger.warning(f"目標変数 {target_col} が見つかりません")
        
        # 高閾値シグナルモデルのトレーニング
        for period in [1, 2, 3]:
            for threshold in [2, 3]:  # 2%, 3%
                for direction in ['long', 'short']:
                    target_col = f'target_high_threshold_{threshold}p_{direction}_{period}'
                    if target_col in df.columns:
                        logger.info(f"高閾値シグナルモデル（{threshold}%, {direction}, {period}期先）のトレーニングを開始")
                        
                        # 特徴量と目標変数
                        X_train = df_train[feature_cols]
                        y_train = df_train[target_col]
                        X_test = df_test[feature_cols]
                        y_test = df_test[target_col]
                        
                        # クラスバランスの確認
                        signal_ratio = y_train.mean() * 100
                        logger.info(f"シグナル比率: {signal_ratio:.2f}%")
                        
                        # モデルトレーニング
                        model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=10,
                            min_samples_split=5,
                            class_weight='balanced',
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        
                        # モデル評価
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        logger.info(f"モデル評価: 精度={accuracy:.4f}, 適合率={precision:.4f}, 再現率={recall:.4f}, F1={f1:.4f}")
                        
                        # モデル保存
                        model_path = models_dir / f"high_threshold_{threshold}p_{direction}_model_period_{period}.joblib"
                        joblib.dump(model, model_path)
                        logger.info(f"モデルを保存しました: {model_path}")
                    else:
                        logger.warning(f"目標変数 {target_col} が見つかりません")
        
        logger.info("モデルトレーニングが完了しました")
        return True
        
    except Exception as e:
        logger.exception(f"エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_simple_models()
    print("処理完了:", "成功" if success else "失敗")
