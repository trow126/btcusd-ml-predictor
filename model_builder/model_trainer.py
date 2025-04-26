# model_builder/model_trainer.py
import pandas as pd
import numpy as np
import logging
import joblib
import os
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional, Union

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self, config=None):
        """
        モデルトレーニングクラス

        Args:
            config: 設定辞書またはNone（デフォルト設定を使用）
        """
        self.config = config if config else self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "input_dir": "data/processed",
            "input_filename": "btcusd_5m_features.csv",
            "output_dir": "models",
            "feature_groups": {
                "price": True,            # 価格関連特徴量
                "volume": True,           # 出来高関連特徴量
                "technical": True,        # テクニカル指標関連特徴量
            },
            "target_periods": [1, 2, 3],  # 予測対象期間 (1=5分後, 2=10分後, 3=15分後)
            "cv_splits": 5,               # 時系列交差検証の分割数
            "test_size": 0.2,             # テストデータの割合
            "regression": {               # 回帰モデル（価格変動率予測）設定
                "model_params": {
                    "objective": "regression",
                    "metric": "mae",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbose": -1
                },
                "fit_params": {
                    "num_boost_round": 1000,
                    "early_stopping_rounds": 50,
                    "verbose_eval": 100
                }
            },
            "classification": {           # 分類モデル（価格変動方向予測）設定
                "model_params": {
                    "objective": "multiclass",
                    "num_class": 5,       # 上昇/横ばい/下落の3クラス
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbose": -1
                },
                "fit_params": {
                    "num_boost_round": 1000,
                    "early_stopping_rounds": 50,
                    "verbose_eval": 100
                }
            }
        }

    def load_data(self) -> pd.DataFrame:
        """
        特徴量データを読み込む

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

    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        特徴量と目標変数を準備

        Args:
            df: 入力データフレーム

        Returns:
            Tuple: (特徴量のDict, 目標変数のDict)
        """
        if df.empty:
            logger.warning("入力データが空です")
            return {}, {}

        # 使用する特徴量を選択
        feature_cols = self._select_features(df)

        # 目標変数（各予測期間に対して）
        target_cols = {}
        for period in self.config["target_periods"]:
            # 回帰目標（価格変動率）
            target_cols[f"regression_{period}"] = f"target_price_change_pct_{period}"
            # 分類目標（価格変動方向）
            target_cols[f"classification_{period}"] = f"target_price_direction_{period}"

        # 特徴量と目標変数のDataFrameを準備
        X = df[feature_cols]
        y_dict = {}

        for target_name, target_col in target_cols.items():
            if target_col in df.columns:
                y_dict[target_name] = df[target_col]

        logger.info(f"特徴量: {len(feature_cols)}個, 目標変数: {len(y_dict)}個")

        return {"X": X}, y_dict

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        学習に使用する特徴量を選択

        Args:
            df: 入力データフレーム

        Returns:
            List[str]: 選択された特徴量名のリスト
        """
        # 特徴量グループごとの選択条件
        feature_cols = []

        # 価格関連特徴量
        if self.config["feature_groups"]["price"]:
            price_cols = [col for col in df.columns if (
                col.startswith("price_change") or
                col in ["open", "high", "low", "close"] or
                col in ["candle_size", "body_size", "upper_shadow", "lower_shadow", "is_bullish"]
            )]
            feature_cols.extend(price_cols)

        # 出来高関連特徴量
        if self.config["feature_groups"]["volume"]:
            volume_cols = [col for col in df.columns if (
                col.startswith("volume") or
                col == "turnover"
            )]
            feature_cols.extend(volume_cols)

        # テクニカル指標関連特徴量
        if self.config["feature_groups"]["technical"]:
            technical_cols = [col for col in df.columns if (
                col.startswith("sma_") or
                col.startswith("ema_") or
                col.startswith("rsi") or
                col.startswith("bb_") or
                col.startswith("macd") or
                col.startswith("stoch_") or
                col.startswith("dist_from_") or
                (col.startswith("highest_") or col.startswith("lowest_"))
            )]
            feature_cols.extend(technical_cols)

        # 目標変数を特徴量から除外
        feature_cols = [col for col in feature_cols if not col.startswith("target_")]

        # 重複を削除
        feature_cols = list(set(feature_cols))

        logger.info(f"選択された特徴量: {len(feature_cols)}個")

        return feature_cols

    def train_test_split(self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]) -> Tuple[
        Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, pd.Series]
    ]:
        """
        時系列を考慮してトレーニングデータとテストデータに分割

        Args:
            X: 特徴量DataFrame
            y_dict: 目標変数のDict

        Returns:
            Tuple: (X_train, X_test, y_train, y_test) の辞書
        """
        if X.empty:
            logger.warning("特徴量データが空です")
            return {}, {}, {}, {}

        # 時系列データなので、最後の一定割合をテストデータとする
        test_size = int(len(X) * self.config["test_size"])
        train_size = len(X) - test_size

        X_train = {"X": X.iloc[:train_size].copy()}
        X_test = {"X": X.iloc[train_size:].copy()}

        y_train = {}
        y_test = {}

        for target_name, target_series in y_dict.items():
            y_train[target_name] = target_series.iloc[:train_size].copy()
            y_test[target_name] = target_series.iloc[train_size:].copy()

        logger.info(f"トレーニングデータ: {train_size}行, テストデータ: {test_size}行")

        return X_train, X_test, y_train, y_test

    def train_regression_models(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        回帰モデル（価格変動率予測）をトレーニング

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict

        Returns:
            Dict: トレーニング結果
        """
        regression_results = {}

        # 各予測期間に対してモデルをトレーニング
        for period in self.config["target_periods"]:
            target_name = f"regression_{period}"

            if target_name not in y_train or target_name not in y_test:
                logger.warning(f"目標変数 {target_name} が見つかりません")
                continue

            logger.info(f"{period}期先の価格変動率予測モデルをトレーニングします")

            # LightGBMデータセットの作成
            lgb_train = lgb.Dataset(
                X_train["X"],
                y_train[target_name],
                feature_name=list(X_train["X"].columns),
                free_raw_data=False
            )

            lgb_valid = lgb.Dataset(
                X_test["X"],
                y_test[target_name],
                reference=lgb_train,
                feature_name=list(X_test["X"].columns),
                free_raw_data=False
            )

            # モデルパラメータ
            model_params = self.config["regression"]["model_params"].copy()

            # ブースティングラウンド数と早期停止設定
            num_boost_round = self.config["regression"]["fit_params"].get("num_boost_round", 1000)
            callbacks = []

            # 早期停止の設定
            early_stopping_rounds = self.config["regression"]["fit_params"].get("early_stopping_rounds", 50)
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))

            # 進捗表示の設定
            verbose_eval = self.config["regression"]["fit_params"].get("verbose_eval", 100)
            if verbose_eval:
                callbacks.append(lgb.log_evaluation(period=verbose_eval, show_stdv=True))

            # モデルのトレーニング
            model = lgb.train(
                params=model_params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )

            # 予測
            y_pred = model.predict(X_test["X"])

            # 評価
            mae = mean_absolute_error(y_test[target_name], y_pred)

            # 結果を保存
            regression_results[target_name] = {
                "model": model,
                "mae": mae,
                "feature_importance": {
                    name: score for name, score in zip(
                        X_train["X"].columns,
                        model.feature_importance(importance_type="gain")
                    )
                }
            }

            logger.info(f"{period}期先の価格変動率予測モデル - MAE: {mae:.6f}")

            # モデルの保存
            self._save_model(model, f"regression_model_period_{period}")

        return regression_results

    def train_classification_models(
        self, X_train: Dict[str, pd.DataFrame], X_test: Dict[str, pd.DataFrame],
        y_train: Dict[str, pd.Series], y_test: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        分類モデル（価格変動方向予測）をトレーニング

        Args:
            X_train: トレーニング特徴量のDict
            X_test: テスト特徴量のDict
            y_train: トレーニング目標変数のDict
            y_test: テスト目標変数のDict

        Returns:
            Dict: トレーニング結果
        """
        classification_results = {}

        # 各予測期間に対してモデルをトレーニング
        for period in self.config["target_periods"]:
            target_name = f"classification_{period}"

            if target_name not in y_train or target_name not in y_test:
                logger.warning(f"目標変数 {target_name} が見つかりません")
                continue

            logger.info(f"{period}期先の価格変動方向予測モデルをトレーニングします")

            # ラベルの正規化（-1, 0, 1を0, 1, 2に変換）
            y_train_norm = y_train[target_name].copy() + 2
            y_test_norm = y_test[target_name].copy() + 2

            # LightGBMデータセットの作成
            lgb_train = lgb.Dataset(
                X_train["X"],
                y_train_norm,
                feature_name=list(X_train["X"].columns),
                free_raw_data=False
            )

            lgb_valid = lgb.Dataset(
                X_test["X"],
                y_test_norm,
                reference=lgb_train,
                feature_name=list(X_test["X"].columns),
                free_raw_data=False
            )

            # モデルパラメータ
            model_params = self.config["classification"]["model_params"].copy()

            # ブースティングラウンド数と早期停止設定
            num_boost_round = self.config["classification"]["fit_params"].get("num_boost_round", 1000)
            callbacks = []

            # 早期停止の設定
            early_stopping_rounds = self.config["classification"]["fit_params"].get("early_stopping_rounds", 50)
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))

            # 進捗表示の設定
            verbose_eval = self.config["classification"]["fit_params"].get("verbose_eval", 100)
            if verbose_eval:
                callbacks.append(lgb.log_evaluation(period=verbose_eval, show_stdv=True))

            # モデルのトレーニング
            model = lgb.train(
                params=model_params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )

            # 予測
            y_pred_proba = model.predict(X_test["X"])
            y_pred = np.argmax(y_pred_proba, axis=1)

            # 予測値を元のラベル（-1, 0, 1）に戻す
            y_pred = y_pred - 2

            # 評価
            accuracy = accuracy_score(y_test[target_name], y_pred)

            # 混同行列
            cm = confusion_matrix(y_test[target_name], y_pred)

            # 分類レポート
            report = classification_report(y_test[target_name], y_pred, output_dict=True)

            # 結果を保存
            classification_results[target_name] = {
                "model": model,
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": report,
                "feature_importance": {
                    name: score for name, score in zip(
                        X_train["X"].columns,
                        model.feature_importance(importance_type="gain")
                    )
                }
            }

            logger.info(f"{period}期先の価格変動方向予測モデル - 正解率: {accuracy:.4f}")

            # モデルの保存
            self._save_model(model, f"classification_model_period_{period}")

        return classification_results
    def _save_model(self, model: Any, name: str) -> bool:
        """
        モデルを保存

        Args:
            model: 保存するモデル
            name: モデル名

        Returns:
            bool: 保存が成功したかどうか
        """
        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # モデルの保存
        model_path = output_dir / f"{name}.joblib"
        try:
            joblib.dump(model, model_path)
            logger.info(f"モデルを {model_path} に保存しました")
            return True
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            return False

    def generate_training_report(
        self, regression_results: Dict[str, Any], classification_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        トレーニング結果の要約レポートを生成

        Args:
            regression_results: 回帰モデルのトレーニング結果
            classification_results: 分類モデルのトレーニング結果

        Returns:
            Dict: トレーニング結果のレポート
        """
        report = {
            "regression": {},
            "classification": {}
        }

        # 回帰モデルの結果
        for target_name, result in regression_results.items():
            period = int(target_name.split("_")[1])

            # 上位の特徴量重要度
            top_features = sorted(
                result["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            report["regression"][f"period_{period}"] = {
                "mae": result["mae"],
                "top_features": dict(top_features)
            }

        # 分類モデルの結果
        for target_name, result in classification_results.items():
            period = int(target_name.split("_")[1])

            # 上位の特徴量重要度
            top_features = sorted(
                result["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            report["classification"][f"period_{period}"] = {
                "accuracy": result["accuracy"],
                "class_accuracy": {
                    "-1 (下落)": result["classification_report"]["-1"]["precision"],
                    "0 (横ばい)": result["classification_report"]["0"]["precision"],
                    "1 (上昇)": result["classification_report"]["1"]["precision"]
                },
                "confusion_matrix": result["confusion_matrix"].tolist(),
                "top_features": dict(top_features)
            }

        return report

# 実行部分（外部から呼び出す場合）
def train_models(config=None):
    """
    モデルのトレーニングを実行する関数

    Args:
        config: 設定辞書またはNone（デフォルト設定を使用）

    Returns:
        Dict: トレーニング結果のレポート
    """
    trainer = ModelTrainer(config)

    # データ読み込み
    df = trainer.load_data()

    # 特徴量と目標変数の準備
    X_dict, y_dict = trainer.prepare_features(df)

    # トレーニングデータとテストデータに分割
    X_train, X_test, y_train, y_test = trainer.train_test_split(X_dict["X"], y_dict)

    # 回帰モデル（価格変動率予測）のトレーニング
    regression_results = trainer.train_regression_models(X_train, X_test, y_train, y_test)

    # 分類モデル（価格変動方向予測）のトレーニング
    classification_results = trainer.train_classification_models(X_train, X_test, y_train, y_test)

    # トレーニング結果のレポート生成
    report = trainer.generate_training_report(regression_results, classification_results)

    return report