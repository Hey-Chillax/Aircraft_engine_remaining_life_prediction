"""
可视化结果模块

功能:
1. 绘制训练曲线（Loss、RMSE、学习率）
2. 绘制模型对比图表
3. 生成预测结果散点图

作者: Augment Agent
日期: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置matplotlib参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def plot_training_curves(model_name: str, data_version: str, early_stopping: str):
    """
    绘制训练曲线
    
    Args:
        model_name: 模型名称
        data_version: 数据版本
        early_stopping: 早停策略
    """
    logger.info(f"绘制{model_name}_{data_version}_{early_stopping}的训练曲线...")
    
    # 创建输出目录
    output_dir = Path("results/training_curves")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载训练历史
    history_path = f"results/training_logs/{model_name}_{data_version}_{early_stopping}_history.csv"
    
    if not Path(history_path).exists():
        logger.warning(f"训练历史文件不存在: {history_path}")
        return
    
    history = pd.read_csv(history_path)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} - {data_version} - {early_stopping}', fontsize=16, fontweight='bold')
    
    # 1. 训练Loss曲线
    ax1 = axes[0, 0]
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练集RMSE曲线
    ax2 = axes[0, 1]
    ax2.plot(history['epoch'], history['train_rmse'], label='Train RMSE', linewidth=2, color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Training RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 验证集和测试集RMSE曲线
    ax3 = axes[1, 0]
    ax3.plot(history['epoch'], history['val_rmse'], label='Validation RMSE', linewidth=2, color='orange')
    ax3.plot(history['epoch'], history['test_rmse'], label='Test RMSE', linewidth=2, color='green')
    
    # 标注最佳epoch
    best_epoch = history.loc[history['val_rmse'].idxmin(), 'epoch']
    best_val_rmse = history['val_rmse'].min()
    ax3.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {int(best_epoch)}')
    ax3.scatter([best_epoch], [best_val_rmse], color='red', s=100, zorder=5)
    ax3.text(best_epoch, best_val_rmse, f'  {best_val_rmse:.2f}', fontsize=10, va='center')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Validation and Test RMSE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 学习率曲线
    ax4 = axes[1, 1]
    ax4.plot(history['epoch'], history['learning_rate'], label='Learning Rate', linewidth=2, color='purple')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / f"{model_name}_{data_version}_{early_stopping}_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练曲线已保存到: {output_path}")


def plot_all_models_comparison():
    """绘制所有模型的测试集RMSE对比图"""
    logger.info("绘制所有模型的测试集RMSE对比图...")
    
    # 创建输出目录
    output_dir = Path("results/training_curves")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有模型的测试集RMSE
    models_data = []
    
    for model in ['RandomForest', 'LSTM', 'Transformer']:
        for data_version in ['full', 'reduced']:
            for es in ['with_early_stopping', 'no_early_stopping']:
                log_path = f"results/training_logs/{model}_{data_version}_{es}.json"
                
                if Path(log_path).exists():
                    import json
                    with open(log_path, 'r') as f:
                        log = json.load(f)
                    
                    models_data.append({
                        'Model': model,
                        'Data': data_version,
                        'Early Stopping': 'Yes' if es == 'with_early_stopping' else 'No',
                        'Test RMSE': log['test_rmse'],
                        'Label': f"{model}\n{data_version}\n{es.replace('_', ' ')}"
                    })
    
    if not models_data:
        logger.warning("没有找到任何训练日志文件")
        return
    
    df = pd.DataFrame(models_data)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(df))
    bars = ax.bar(x, df['Test RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] * 2)
    
    # 添加数值标签
    for i, (bar, rmse) in enumerate(zip(bars, df['Test RMSE'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('Test RMSE Comparison Across All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['Model']}\n{row['Data']}\nES: {row['Early Stopping']}" 
                        for _, row in df.iterrows()], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / "all_models_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比图已保存到: {output_path}")
    
    # 找出最佳模型
    best_model = df.loc[df['Test RMSE'].idxmin()]
    logger.info(f"最佳模型: {best_model['Model']} - {best_model['Data']} - ES: {best_model['Early Stopping']} - Test RMSE: {best_model['Test RMSE']:.4f}")


def plot_prediction_scatter(model_name: str, data_version: str, early_stopping: str):
    """
    绘制预测值vs真实值散点图
    
    Args:
        model_name: 模型名称
        data_version: 数据版本
        early_stopping: 早停策略
    """
    logger.info(f"绘制{model_name}_{data_version}_{early_stopping}的预测散点图...")
    
    # 创建输出目录
    output_dir = Path("results/training_curves")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载预测结果
    pred_path = f"results/predictions/{model_name}_{data_version}_{early_stopping}.csv"
    
    if not Path(pred_path).exists():
        logger.warning(f"预测结果文件不存在: {pred_path}")
        return
    
    predictions = pd.read_csv(pred_path)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_name} - {data_version} - {early_stopping}', fontsize=14, fontweight='bold')
    
    # 1. 预测值vs真实值散点图
    ax1 = axes[0]
    ax1.scatter(predictions['True_RUL'], predictions['Predicted_RUL'], alpha=0.6, s=50)
    
    # 添加对角线
    min_val = min(predictions['True_RUL'].min(), predictions['Predicted_RUL'].min())
    max_val = max(predictions['True_RUL'].max(), predictions['Predicted_RUL'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('True RUL')
    ax1.set_ylabel('Predicted RUL')
    ax1.set_title('Predicted vs True RUL')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测误差分布直方图
    ax2 = axes[1]
    ax2.hist(predictions['Error'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (Predicted - True)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / f"{model_name}_{data_version}_{early_stopping}_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"散点图已保存到: {output_path}")


if __name__ == "__main__":
    # 为所有模型绘制训练曲线
    for model in ['LSTM', 'Transformer']:
        for data_version in ['full', 'reduced']:
            for es in ['with_early_stopping', 'no_early_stopping']:
                plot_training_curves(model, data_version, es)
                plot_prediction_scatter(model, data_version, es)
    
    # 绘制所有模型对比图
    plot_all_models_comparison()
    
    logger.info("所有可视化完成!")

