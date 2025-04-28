# model_builder/utils/visualization/data_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger("data_visualizer")

def visualize_feature_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None, save_path: Optional[str] = None):
    """
    特徴量の分布を可視化

    Args:
        df: 入力データフレーム
        columns: 可視化する列のリスト（Noneの場合は数値型の列を最大20個選択）
        save_path: 保存先パス（Noneの場合は表示のみ）
    """
    logger.info("特徴量の分布を可視化しています")
    
    if columns is None:
        # 数値型の列を選択
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 最大20個まで
        columns = numeric_cols[:min(20, len(numeric_cols))]
    
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            if col in df.columns:
                sns.histplot(df[col].dropna(), ax=axes[i], kde=True)
                axes[i].set_title(col)
                axes[i].set_xlabel('')
            else:
                logger.warning(f"列 {col} がデータフレームに存在しません")
    
    # 余分なサブプロットを非表示
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        # 保存先ディレクトリが存在しない場合は作成
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        logger.info(f"分布図を {save_path} に保存しました")
    else:
        plt.show()

def visualize_feature_importances(feature_importances: Dict[str, float], top_n: int = 20, save_path: Optional[str] = None):
    """
    特徴量重要度の可視化

    Args:
        feature_importances: 特徴量名と重要度のDict
        top_n: 表示する上位の特徴量数
        save_path: 保存先パス（Noneの場合は表示のみ）
    """
    logger.info(f"特徴量重要度の上位 {top_n} 個を可視化しています")
    
    # 特徴量重要度を降順でソート
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:min(top_n, len(sorted_features))]
    
    # データフレームに変換
    df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    
    # プロット
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title(f'Top {len(top_features)} Feature Importances')
    plt.tight_layout()
    
    if save_path:
        # 保存先ディレクトリが存在しない場合は作成
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        logger.info(f"特徴量重要度図を {save_path} に保存しました")
    else:
        plt.show()

def visualize_predictions(y_true: pd.Series, y_pred: pd.Series, title: str = 'Actual vs Predicted', save_path: Optional[str] = None):
    """
    予測値と実際の値の比較可視化

    Args:
        y_true: 実際の値
        y_pred: 予測値
        title: グラフのタイトル
        save_path: 保存先パス（Noneの場合は表示のみ）
    """
    logger.info(f"{title}の予測比較を可視化しています")
    
    plt.figure(figsize=(12, 6))
    
    # 時系列プロット
    plt.subplot(1, 2, 1)
    plt.plot(y_true.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'{title} - Time Series')
    plt.legend()
    
    # 散布図
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title} - Scatter Plot')
    
    # 完全な予測ラインを追加
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.tight_layout()
    
    if save_path:
        # 保存先ディレクトリが存在しない場合は作成
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        logger.info(f"予測比較図を {save_path} に保存しました")
    else:
        plt.show()

def visualize_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], title: str = 'Confusion Matrix', save_path: Optional[str] = None):
    """
    混同行列の可視化

    Args:
        confusion_matrix: 混同行列
        class_names: クラス名のリスト
        title: グラフのタイトル
        save_path: 保存先パス（Noneの場合は表示のみ）
    """
    logger.info(f"{title}の混同行列を可視化しています")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        # 保存先ディレクトリが存在しない場合は作成
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        logger.info(f"混同行列を {save_path} に保存しました")
    else:
        plt.show()

def create_evaluation_visualizations(evaluation_results: Dict[str, Any], output_dir: str = 'results/visualization'):
    """
    評価結果を視覚化して保存

    Args:
        evaluation_results: 評価結果のDict
        output_dir: 出力ディレクトリ
    """
    logger.info(f"評価結果の可視化を開始します。出力先: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 各モデルタイプに対する可視化
    for model_type, results in evaluation_results.items():
        if not results:
            continue
            
        model_dir = output_path / model_type
        model_dir.mkdir(exist_ok=True)
        
        for period_key, period_results in results.items():
            if isinstance(period_results, dict) and "error" not in period_results:
                # 特徴量重要度の可視化
                if "feature_importance" in period_results:
                    visualize_feature_importances(
                        period_results["feature_importance"],
                        save_path=str(model_dir / f"{period_key}_feature_importance.png")
                    )
                
                # 混同行列の可視化（分類モデルの場合）
                if "confusion_matrix" in period_results:
                    class_names = []
                    if model_type == "classification":
                        class_names = ["下落", "横ばい", "上昇"]
                    elif "binary" in model_type:
                        class_names = ["下落", "上昇"]
                    
                    if class_names:
                        visualize_confusion_matrix(
                            np.array(period_results["confusion_matrix"]),
                            class_names,
                            title=f"{period_key} Confusion Matrix",
                            save_path=str(model_dir / f"{period_key}_confusion_matrix.png")
                        )
    
    logger.info("評価結果の可視化が完了しました")
