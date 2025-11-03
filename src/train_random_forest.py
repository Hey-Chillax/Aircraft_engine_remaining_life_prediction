"""
随机森林训练脚本

功能:
1. 加载数据并添加统计特征
2. 训练随机森林模型
3. 评估模型性能
4. 保存模型和训练日志

作者: Augment Agent
日期: 2025
"""

import numpy as np
import pandas as pd
import pickle
import time
import json
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple

from data_loader import DataLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_random_forest(data_version: str = "full", window_size: int = 50, 
                       random_state: int = 42) -> Dict:
    """
    训练随机森林模型
    
    Args:
        data_version: 数据版本，"full"或"reduced"
        window_size: 滑动窗口大小
        random_state: 随机种子
    
    Returns:
        results: 包含模型、训练日志和评估结果的字典
    """
    logger.info("=" * 80)
    logger.info(f"开始训练随机森林模型 - 数据版本: {data_version}")
    logger.info("=" * 80)
    
    # 创建输出目录
    Path("models").mkdir(exist_ok=True)
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    Path("results/training_logs").mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    data_loader = DataLoader(random_state=random_state)
    train_df, test_df, rul_df = data_loader.load_data(data_version)
    
    # 划分训练集和验证集
    train_split, val_split = data_loader.split_train_val(train_df, val_ratio=0.2)
    
    # 准备训练数据（添加统计特征）
    logger.info("准备训练数据...")
    X_train, y_train = data_loader.prepare_rf_data(train_split, window_size)
    X_val, y_val = data_loader.prepare_rf_data(val_split, window_size)
    X_test, test_engine_ids = data_loader.prepare_rf_test_data(test_df, window_size)
    y_test = rul_df['RUL'].values
    
    logger.info(f"训练集: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"验证集: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"测试集: X={X_test.shape}, y={y_test.shape}")
    
    # 创建随机森林模型
    logger.info("创建随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    # 训练模型
    logger.info("开始训练...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 评估模型
    logger.info("评估模型...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logger.info(f"训练集 - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    logger.info(f"验证集 - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    logger.info(f"测试集 - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # 保存模型
    model_path = f"models/RandomForest_{data_version}_no_early_stopping.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"模型已保存到: {model_path}")
    
    # 保存测试集预测结果
    predictions_df = pd.DataFrame({
        'Engine_ID': test_engine_ids,
        'True_RUL': y_test,
        'Predicted_RUL': y_test_pred,
        'Error': y_test_pred - y_test,
        'Absolute_Error': np.abs(y_test_pred - y_test)
    })
    
    pred_path = f"results/predictions/RandomForest_{data_version}_no_early_stopping.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"预测结果已保存到: {pred_path}")
    
    # 保存训练日志
    training_log = {
        'model': 'RandomForest',
        'data_version': data_version,
        'early_stopping': 'no',
        'window_size': window_size,
        'n_estimators': 200,
        'max_depth': 20,
        'training_time': training_time,
        'train_rmse': float(train_rmse),
        'val_rmse': float(val_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'val_mae': float(val_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'val_r2': float(val_r2),
        'test_r2': float(test_r2),
        'n_features': X_train.shape[1]
    }
    
    log_path = f"results/training_logs/RandomForest_{data_version}_no_early_stopping.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    logger.info(f"训练日志已保存到: {log_path}")
    
    # 特征重要性分析
    feature_importance = model.feature_importances_
    logger.info(f"特征重要性前10: {np.sort(feature_importance)[-10:][::-1]}")
    
    results = {
        'model': model,
        'training_log': training_log,
        'predictions': predictions_df,
        'feature_importance': feature_importance
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练随机森林模型')
    parser.add_argument('--data_version', type=str, default='full', choices=['full', 'reduced'],
                       help='数据版本: full或reduced')
    parser.add_argument('--window_size', type=int, default=50,
                       help='滑动窗口大小')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 训练模型
    results = train_random_forest(
        data_version=args.data_version,
        window_size=args.window_size,
        random_state=args.random_state
    )
    
    logger.info("=" * 80)
    logger.info("随机森林训练完成!")
    logger.info("=" * 80)

