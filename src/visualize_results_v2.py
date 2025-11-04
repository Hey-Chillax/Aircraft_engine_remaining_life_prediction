"""
可视化脚本V2 - 生成第二轮训练的所有图表

功能:
1. 为每个模型生成训练曲线图（4个子图）
2. 为每个模型生成预测散点图
3. 生成所有模型的对比图表
4. 对比新旧模型性能

作者: Augment Agent
日期: 2025
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(model_name: str, history_path: str, output_dir: str):
    """
    绘制训练曲线（4个子图）
    
    Args:
        model_name: 模型名称
        history_path: 训练历史CSV路径
        output_dir: 输出目录
    """
    try:
        # 读取训练历史
        history = pd.read_csv(history_path)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Curves - {model_name}', fontsize=16, fontweight='bold')
        
        # 子图1: Train Loss
        axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: Train RMSE
        axes[0, 1].plot(history['epoch'], history['train_rmse'], 'g-', linewidth=2, label='Train RMSE')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('RMSE', fontsize=12)
        axes[0, 1].set_title('Training RMSE', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: Val/Test RMSE
        axes[1, 0].plot(history['epoch'], history['val_rmse'], 'r-', linewidth=2, label='Val RMSE')
        axes[1, 0].plot(history['epoch'], history['test_rmse'], 'orange', linewidth=2, label='Test RMSE')
        
        # 标注最佳epoch
        best_epoch_idx = history['val_rmse'].idxmin()
        best_epoch = history.loc[best_epoch_idx, 'epoch']
        best_val_rmse = history.loc[best_epoch_idx, 'val_rmse']
        axes[1, 0].axvline(x=best_epoch, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch: {int(best_epoch)}')
        axes[1, 0].scatter([best_epoch], [best_val_rmse], color='red', s=100, zorder=5)
        
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('RMSE', fontsize=12)
        axes[1, 0].set_title('Validation & Test RMSE', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: Learning Rate
        axes[1, 1].plot(history['epoch'], history['learning_rate'], 'm-', linewidth=2, label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule (Warmup + Cosine Annealing)', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, f'{model_name}_training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线图已保存: {output_path}")
        
    except Exception as e:
        logger.error(f"绘制训练曲线失败 ({model_name}): {str(e)}")


def plot_predictions(model_name: str, predictions_path: str, output_dir: str):
    """
    绘制预测结果散点图
    
    Args:
        model_name: 模型名称
        predictions_path: 预测结果CSV路径
        output_dir: 输出目录
    """
    try:
        # 读取预测结果
        predictions = pd.read_csv(predictions_path)
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Predictions - {model_name}', fontsize=16, fontweight='bold')
        
        # 子图1: 预测值 vs 真实值散点图
        axes[0].scatter(predictions['True_RUL'], predictions['Predicted_RUL'], 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 添加理想线（y=x）
        min_val = min(predictions['True_RUL'].min(), predictions['Predicted_RUL'].min())
        max_val = max(predictions['True_RUL'].max(), predictions['Predicted_RUL'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
        
        # 计算RMSE和R²
        rmse = np.sqrt(np.mean((predictions['Predicted_RUL'] - predictions['True_RUL'])**2))
        r2 = 1 - np.sum((predictions['Predicted_RUL'] - predictions['True_RUL'])**2) / \
                 np.sum((predictions['True_RUL'] - predictions['True_RUL'].mean())**2)
        
        axes[0].text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.3f}', 
                    transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[0].set_xlabel('True RUL', fontsize=12)
        axes[0].set_ylabel('Predicted RUL', fontsize=12)
        axes[0].set_title('Predicted vs True RUL', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 子图2: 预测误差分布直方图
        errors = predictions['Error']
        axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[1].axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean Error: {errors.mean():.2f}')
        
        axes[1].set_xlabel('Prediction Error (Predicted - True)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, f'{model_name}_predictions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"预测散点图已保存: {output_path}")
        
    except Exception as e:
        logger.error(f"绘制预测散点图失败 ({model_name}): {str(e)}")


def generate_all_individual_plots():
    """生成所有单个模型的图表"""
    logger.info("=" * 80)
    logger.info("开始生成所有单个模型的图表...")
    logger.info("=" * 80)
    
    # 创建输出目录
    output_dir = "results/training_curves_2"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有训练日志
    log_dir = "results/training_logs_2"
    json_files = list(Path(log_dir).glob("*.json"))
    
    logger.info(f"找到 {len(json_files)} 个训练日志文件")
    
    for json_file in json_files:
        model_name = json_file.stem
        
        # 训练曲线图
        history_path = json_file.parent / f"{model_name}_history.csv"
        if history_path.exists():
            plot_training_curves(model_name, str(history_path), output_dir)
        else:
            logger.warning(f"未找到训练历史文件: {history_path}")
        
        # 预测散点图
        predictions_path = f"results/predictions_2/{model_name}.csv"
        if Path(predictions_path).exists():
            plot_predictions(model_name, predictions_path, output_dir)
        else:
            logger.warning(f"未找到预测结果文件: {predictions_path}")
    
    logger.info(f"所有单个模型图表已生成，保存在: {output_dir}")


def plot_all_models_comparison():
    """生成所有模型的对比图表"""
    logger.info("=" * 80)
    logger.info("开始生成模型对比图表...")
    logger.info("=" * 80)

    # 创建输出目录
    output_dir = "results/training_curves_2"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 收集所有模型的结果
    log_dir = "results/training_logs_2"
    json_files = list(Path(log_dir).glob("*.json"))

    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append({
                'model_name': json_file.stem,
                'model_type': data['model'].split('_')[0],
                'model_size': data['model'].split('_')[1],
                'data_version': data['data_version'],
                'early_stopping': data['early_stopping'],
                'sequence_length': data['training_config']['sequence_length'],
                'test_rmse': data['results']['test_rmse'],
                'val_rmse': data['results']['val_rmse'],
                'train_rmse': data['results']['train_rmse'],
                'n_parameters': data['results']['n_parameters'],
                'training_time': data['results']['training_time'],
                'best_epoch': data['results']['best_epoch']
            })

    df = pd.DataFrame(results)
    df = df.sort_values('test_rmse')

    # 图1: 所有32个新模型测试集RMSE对比柱状图
    plt.figure(figsize=(20, 8))
    colors = ['green' if i < 5 else 'skyblue' for i in range(len(df))]
    plt.bar(range(len(df)), df['test_rmse'], color=colors, edgecolor='black', linewidth=0.5)
    plt.axhline(y=21.73, color='red', linestyle='--', linewidth=2, label='Round 1 Best (21.73)')
    plt.xlabel('Model Index (sorted by Test RMSE)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('All 32 New Models - Test RMSE Comparison (Round 2)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df)), range(1, len(df)+1), rotation=0, fontsize=8)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_models_rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图表已保存: all_models_rmse_comparison.png")

    # 图2: 模型大小 vs 性能散点图
    plt.figure(figsize=(12, 8))
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.scatter(subset['n_parameters'], subset['test_rmse'],
                   label=model_type, s=100, alpha=0.6, edgecolors='black', linewidth=1)
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('Model Size vs Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_size_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图表已保存: model_size_vs_performance.png")

    # 图3: 训练时间 vs 性能散点图
    plt.figure(figsize=(12, 8))
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.scatter(subset['training_time']/60, subset['test_rmse'],
                   label=model_type, s=100, alpha=0.6, edgecolors='black', linewidth=1)
    plt.xlabel('Training Time (minutes)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('Training Time vs Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_time_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图表已保存: training_time_vs_performance.png")

    # 图4: Small vs Large架构性能对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, model_type in enumerate(['LSTM', 'Transformer']):
        subset = df[df['model_type'] == model_type]
        size_comparison = subset.groupby('model_size')['test_rmse'].agg(['mean', 'std', 'min'])

        axes[i].bar(size_comparison.index, size_comparison['mean'],
                   yerr=size_comparison['std'], capsize=5,
                   color=['lightcoral', 'lightblue'], edgecolor='black', linewidth=1)
        axes[i].set_xlabel('Model Size', fontsize=12)
        axes[i].set_ylabel('Test RMSE', fontsize=12)
        axes[i].set_title(f'{model_type} - Architecture Size Comparison', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')

        # 添加最小值标注
        for idx, (size, row) in enumerate(size_comparison.iterrows()):
            axes[i].text(idx, row['min'], f"Min: {row['min']:.2f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/architecture_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图表已保存: architecture_size_comparison.png")

    # 图5: 时间窗口30 vs 50性能对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, model_type in enumerate(['LSTM', 'Transformer']):
        subset = df[df['model_type'] == model_type]
        window_comparison = subset.groupby('sequence_length')['test_rmse'].agg(['mean', 'std', 'min'])

        axes[i].bar(window_comparison.index.astype(str), window_comparison['mean'],
                   yerr=window_comparison['std'], capsize=5,
                   color=['lightgreen', 'lightyellow'], edgecolor='black', linewidth=1)
        axes[i].set_xlabel('Time Window Size', fontsize=12)
        axes[i].set_ylabel('Test RMSE', fontsize=12)
        axes[i].set_title(f'{model_type} - Time Window Comparison', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')

        # 添加最小值标注
        for idx, (window, row) in enumerate(window_comparison.iterrows()):
            axes[i].text(idx, row['min'], f"Min: {row['min']:.2f}",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_window_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("图表已保存: time_window_comparison.png")

    logger.info("所有对比图表已生成完成！")


if __name__ == "__main__":
    generate_all_individual_plots()
    plot_all_models_comparison()

