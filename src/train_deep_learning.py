"""
深度学习模型训练脚本（LSTM和Transformer通用）

功能:
1. 训练LSTM或Transformer模型
2. 支持早停和非早停两种模式
3. 学习率调度（余弦退火 + Warmup）
4. 保存训练日志和模型权重

作者: Augment Agent
日期: 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import time
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import DataLoader
from models import LSTMModel, TransformerModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """Warmup + 余弦退火学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        """更新学习率"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup阶段：线性增加
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # 余弦退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def train_deep_model(model_type: str = "LSTM", data_version: str = "full", 
                     use_early_stopping: bool = False, sequence_length: int = 50,
                     batch_size: int = 64, epochs: int = 300, learning_rate: float = 0.001,
                     warmup_epochs: int = 0, patience: int = 20, 
                     device: str = "cuda", random_state: int = 42) -> Dict:
    """
    训练深度学习模型（LSTM或Transformer）
    
    Args:
        model_type: 模型类型，"LSTM"或"Transformer"
        data_version: 数据版本，"full"或"reduced"
        use_early_stopping: 是否使用早停
        sequence_length: 时间窗口大小
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 初始学习率
        warmup_epochs: Warmup轮数（仅Transformer使用）
        patience: 早停耐心值
        device: 训练设备
        random_state: 随机种子
    
    Returns:
        results: 包含训练日志和评估结果的字典
    """
    logger.info("=" * 80)
    logger.info(f"开始训练{model_type}模型 - 数据版本: {data_version}, 早停: {use_early_stopping}")
    logger.info("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # 创建输出目录
    Path("models").mkdir(exist_ok=True)
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    Path("results/training_logs").mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    data_loader = DataLoader(random_state=random_state)
    train_df, test_df, rul_df = data_loader.load_data(data_version)
    
    # 划分训练集和验证集
    train_split, val_split = data_loader.split_train_val(train_df, val_ratio=0.2)
    
    # 创建时间序列窗口
    logger.info("创建时间序列窗口...")
    X_train, y_train = data_loader.create_sequences(train_split, sequence_length, stride=1)
    X_val, y_val = data_loader.create_sequences(val_split, sequence_length, stride=1)
    X_test, test_engine_ids = data_loader.create_test_sequences(test_df, sequence_length)
    y_test = rul_df['RUL'].values
    
    logger.info(f"训练集: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"验证集: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"测试集: X={X_test.shape}, y={y_test.shape}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_size = X_train.shape[2]
    logger.info(f"创建{model_type}模型（输入维度={input_size}）...")
    
    if model_type == "LSTM":
        model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=3, dropout=0.2)
    else:  # Transformer
        model = TransformerModel(
            input_size=input_size, d_model=128, nhead=8, num_layers=4,
            dim_feedforward=512, dropout=0.1, max_seq_len=100
        )
    
    model = model.to(device)
    
    # 计算参数数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数数量: {n_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    if model_type == "Transformer" and warmup_epochs > 0:
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, learning_rate, min_lr=1e-6)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 训练循环
    logger.info("开始训练...")
    start_time = time.time()
    
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'test_rmse': [],
        'learning_rate': []
    }
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_dataset)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        # 测试阶段（用于绘图）
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                test_preds.extend(outputs.cpu().numpy())
                test_targets.extend(batch_y.numpy())
        
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        
        # 更新学习率
        if isinstance(scheduler, WarmupCosineScheduler):
            current_lr = scheduler.step()
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        # 记录训练历史
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['train_rmse'].append(train_rmse)
        training_history['val_rmse'].append(val_rmse)
        training_history['test_rmse'].append(test_rmse)
        training_history['learning_rate'].append(current_lr)
        
        # 打印进度
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch [{epoch}/{epochs}] - "
                       f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
                       f"Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}, "
                       f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            es_suffix = "with_early_stopping" if use_early_stopping else "no_early_stopping"
            model_path = f"models/{model_type}_{data_version}_{es_suffix}_best.pth"
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        # 早停检查
        if use_early_stopping and patience_counter >= patience:
            logger.info(f"早停触发！最佳验证集RMSE: {best_val_rmse:.4f} (Epoch {best_epoch})")
            break
    
    training_time = time.time() - start_time
    logger.info(f"训练完成，耗时: {training_time:.2f}秒")
    logger.info(f"最佳验证集RMSE: {best_val_rmse:.4f} (Epoch {best_epoch})")
    
    # 加载最佳模型进行最终评估
    es_suffix = "with_early_stopping" if use_early_stopping else "no_early_stopping"
    model_path = f"models/{model_type}_{data_version}_{es_suffix}_best.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 最终评估
    logger.info("最终评估...")

    with torch.no_grad():
        # 训练集评估
        train_preds_final = []
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            train_preds_final.extend(outputs.cpu().numpy())
        train_rmse_final = np.sqrt(mean_squared_error(y_train, train_preds_final))
        train_mae_final = mean_absolute_error(y_train, train_preds_final)
        train_r2_final = r2_score(y_train, train_preds_final)

        # 验证集评估
        val_preds_final = []
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            val_preds_final.extend(outputs.cpu().numpy())
        val_rmse_final = np.sqrt(mean_squared_error(y_val, val_preds_final))
        val_mae_final = mean_absolute_error(y_val, val_preds_final)
        val_r2_final = r2_score(y_val, val_preds_final)

        # 测试集评估
        test_preds_final = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_preds_final.extend(outputs.cpu().numpy())
        test_preds_final = np.array(test_preds_final)
        test_rmse_final = np.sqrt(mean_squared_error(y_test, test_preds_final))
        test_mae_final = mean_absolute_error(y_test, test_preds_final)
        test_r2_final = r2_score(y_test, test_preds_final)

    logger.info(f"训练集 - RMSE: {train_rmse_final:.4f}, MAE: {train_mae_final:.4f}, R²: {train_r2_final:.4f}")
    logger.info(f"验证集 - RMSE: {val_rmse_final:.4f}, MAE: {val_mae_final:.4f}, R²: {val_r2_final:.4f}")
    logger.info(f"测试集 - RMSE: {test_rmse_final:.4f}, MAE: {test_mae_final:.4f}, R²: {test_r2_final:.4f}")

    # 保存测试集预测结果
    predictions_df = pd.DataFrame({
        'Engine_ID': test_engine_ids,
        'True_RUL': y_test,
        'Predicted_RUL': test_preds_final,
        'Error': test_preds_final - y_test,
        'Absolute_Error': np.abs(test_preds_final - y_test)
    })

    pred_path = f"results/predictions/{model_type}_{data_version}_{es_suffix}.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"预测结果已保存到: {pred_path}")

    # 保存训练历史
    history_df = pd.DataFrame(training_history)
    history_path = f"results/training_logs/{model_type}_{data_version}_{es_suffix}_history.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f"训练历史已保存到: {history_path}")

    # 保存训练日志
    training_log = {
        'model': model_type,
        'data_version': data_version,
        'early_stopping': 'yes' if use_early_stopping else 'no',
        'sequence_length': sequence_length,
        'batch_size': batch_size,
        'epochs': epochs,
        'best_epoch': best_epoch,
        'learning_rate': learning_rate,
        'warmup_epochs': warmup_epochs,
        'training_time': training_time,
        'n_parameters': n_params,
        'train_rmse': float(train_rmse_final),
        'val_rmse': float(val_rmse_final),
        'test_rmse': float(test_rmse_final),
        'train_mae': float(train_mae_final),
        'val_mae': float(val_mae_final),
        'test_mae': float(test_mae_final),
        'train_r2': float(train_r2_final),
        'val_r2': float(val_r2_final),
        'test_r2': float(test_r2_final)
    }

    log_path = f"results/training_logs/{model_type}_{data_version}_{es_suffix}.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    logger.info(f"训练日志已保存到: {log_path}")

    results = {
        'training_log': training_log,
        'training_history': training_history,
        'predictions': predictions_df
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练深度学习模型')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'Transformer'],
                       help='模型类型: LSTM或Transformer')
    parser.add_argument('--data_version', type=str, default='full', choices=['full', 'reduced'],
                       help='数据版本: full或reduced')
    parser.add_argument('--use_early_stopping', action='store_true',
                       help='是否使用早停')
    parser.add_argument('--sequence_length', type=int, default=50,
                       help='时间窗口大小')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=300,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='初始学习率')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Warmup轮数')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停耐心值')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')

    args = parser.parse_args()

    # 训练模型
    results = train_deep_model(
        model_type=args.model_type,
        data_version=args.data_version,
        use_early_stopping=args.use_early_stopping,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        device=args.device,
        random_state=args.random_state
    )

    logger.info("=" * 80)
    logger.info(f"{args.model_type}训练完成!")
    logger.info("=" * 80)

