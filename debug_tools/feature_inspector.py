
# debug_tools/feature_inspector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_inspector")

def load_feature_data(file_path: str) -> pd.DataFrame:
    """特徴量データをロードする関数"""
    logger.info(f"ファイルをロード中: {file_path}")
    try:
        df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
        logger.info(f"データロード成功: {len(df)}行, {len(df.columns)}列")
        return df
    except Exception as e:
        logger.error(f"データロードエラー: {e}")
        return pd.DataFrame()

def inspect_target_variables(df: pd.DataFrame) -> None:
    """目標変数の分布と統計を確認"""
    if df.empty:
        logger.error("データが空です")
        return

    # 目標変数を抽出
    target_cols = [col for col in df.columns if col.startswith("target_")]
    if not target_cols:
        logger.warning("目標変数が見つかりません")
        return

    logger.info(f"目標変数の数: {len(target_cols)}")
    logger.info(f"目標変数のリスト: {target_cols}")

    # 目標変数の種類別にグループ化
    target_groups = {
        "regression": [col for col in target_cols if "price_change_pct" in col],
        "threshold_ternary": [col for col in target_cols if "threshold_ternary" in col],
        "threshold_binary": [col for col in target_cols if "threshold_binary" in col],
        "binary": [col for col in target_cols if "binary" in col and "threshold" not in col],
        "other": [col for col in target_cols if not any(s in col for s in 
                  ["price_change_pct", "threshold_ternary", "threshold_binary", "binary"])]
    }

    # 各グループの変数を表示
    for group_name, cols in target_groups.items():
        logger.info(f"{group_name} 変数: {cols}")

    # 各目標変数の統計情報を表示
    for col in target_cols:
        if df[col].dtype in [np.float64, np.int64]:
            # 数値型の場合は統計情報を表示
            stats = df[col].describe()
            logger.info(f"{col} の統計情報:\n{stats}")
            
            # 非数値の確認
            nan_count = df[col].isna().sum()
            logger.info(f"{col} のNaN個数: {nan_count} ({nan_count/len(df)*100:.2f}%)")
            
            # 分類変数の場合はクラス分布も表示
            if "binary" in col or "direction" in col or "threshold" in col:
                value_counts = df[col].value_counts(dropna=False)
                logger.info(f"{col} のクラス分布:\n{value_counts}")

    # 特に閾値ベース二値分類のターゲットについて詳細を表示
    threshold_binary_cols = target_groups["threshold_binary"]
    if threshold_binary_cols:
        for col in threshold_binary_cols:
            # NaNが含まれているか確認
            nan_count = df[col].isna().sum()
            valid_count = len(df[col].dropna())
            logger.info(f"{col}: 有効値 {valid_count}行, NaN {nan_count}行")
            
            # 有効な値の分布
            if valid_count > 0:
                class_dist = df[col].value_counts(dropna=True)
                logger.info(f"{col} の有効値クラス分布:\n{class_dist}")
                
                # クラスバランスを確認
                if len(class_dist) > 1:
                    ratio = class_dist.min() / class_dist.max()
                    logger.info(f"{col} のクラスバランス比率: {ratio:.4f} (1に近いほど均等)")
            else:
                logger.warning(f"{col} に有効な値がありません")

def main():
    # プロジェクトのルートディレクトリを取得
    project_root = Path(__file__).parent.parent
    
    # 特徴量ファイルのパス
    feature_file = project_root / "data" / "processed" / "btcusd_5m_features.csv"
    
    # パスの存在確認
    if not feature_file.exists():
        logger.error(f"ファイルが見つかりません: {feature_file}")
        # 代替パスを試す
        feature_file = project_root / "data" / "processed" / "btcusd_5m_optimized_features.csv"
        if not feature_file.exists():
            logger.error(f"代替ファイルも見つかりません: {feature_file}")
            return
        logger.info(f"代替ファイルを使用します: {feature_file}")
    
    # データをロードして検査
    df = load_feature_data(str(feature_file))
    if not df.empty:
        inspect_target_variables(df)
        logger.info("特徴量の検査が完了しました")
    else:
        logger.error("特徴量データが空のため、検査を実行できません")

if __name__ == "__main__":
    main()
