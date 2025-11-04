"""
主训练脚本V2 - 训练所有32个新模型

功能:
1. 依次训练LSTM和Transformer的Small和Large变体
2. 每个模型训练Full和Reduced两个数据版本
3. 每个版本训练带早停和不带早停两种模式
4. 每个模式训练时间窗口30和50两种情况
5. 自动生成可视化图表和总结报告

作者: Augment Agent
日期: 2025
"""

import os
import sys
import logging
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from train_deep_learning_v2 import train_deep_model_v2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数 - 训练所有新模型"""
    
    logger.info("=" * 100)
    logger.info("开始训练所有新模型 - 共32个模型")
    logger.info("=" * 100)
    
    total_start_time = time.time()
    
    # 创建所有必要的目录
    Path("models_2").mkdir(exist_ok=True)
    Path("results/predictions_2").mkdir(parents=True, exist_ok=True)
    Path("results/training_logs_2").mkdir(parents=True, exist_ok=True)
    Path("results/training_curves_2").mkdir(parents=True, exist_ok=True)
    Path("augment_caption").mkdir(exist_ok=True)
    
    # 训练计数器
    completed_models = 0
    total_models = 32
    
    # 定义训练配置
    model_configs = []
    
    # LSTM模型配置（16个）
    for model_size in ['small', 'large']:
        for data_version in ['full', 'reduced']:
            for use_early_stopping in [True, False]:
                for sequence_length in [30, 50]:
                    model_configs.append({
                        'model_type': 'LSTM',
                        'model_size': model_size,
                        'data_version': data_version,
                        'use_early_stopping': use_early_stopping,
                        'sequence_length': sequence_length,
                        'batch_size': 64,
                        'epochs': 2000,
                        'learning_rate': 0.0005,
                        'warmup_epochs': 50,
                        'patience': 30,
                        'device': 'cuda',
                        'random_state': 42
                    })
    
    # Transformer模型配置（16个）
    for model_size in ['small', 'large']:
        for data_version in ['full', 'reduced']:
            for use_early_stopping in [True, False]:
                for sequence_length in [30, 50]:
                    model_configs.append({
                        'model_type': 'Transformer',
                        'model_size': model_size,
                        'data_version': data_version,
                        'use_early_stopping': use_early_stopping,
                        'sequence_length': sequence_length,
                        'batch_size': 64,
                        'epochs': 2000,
                        'learning_rate': 0.0002,
                        'warmup_epochs': 100,
                        'patience': 30,
                        'device': 'cuda',
                        'random_state': 42
                    })
    
    # 训练所有模型
    for i, config in enumerate(model_configs, 1):
        logger.info("\n" + "=" * 100)
        logger.info(f"[{i}/{total_models}] 训练模型: {config['model_type']}-{config['model_size']} - "
                   f"{config['data_version']} - 早停: {config['use_early_stopping']} - "
                   f"时间窗口: {config['sequence_length']}")
        logger.info("=" * 100)
        
        try:
            # 训练模型
            results = train_deep_model_v2(**config)
            completed_models += 1
            logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            logger.exception(e)
            logger.info(f"跳过该模型，继续下一个...")
            continue
    
    # 总结
    total_time = time.time() - total_start_time
    
    logger.info("\n" + "=" * 100)
    logger.info("所有任务完成!")
    logger.info("=" * 100)
    logger.info(f"完成模型数: {completed_models}/{total_models}")
    logger.info(f"总耗时: {total_time/3600:.2f}小时 ({total_time/60:.2f}分钟)")
    logger.info("\n生成的文件:")
    logger.info("  - models_2/: 所有新模型权重")
    logger.info("  - results/predictions_2/: 所有新预测结果")
    logger.info("  - results/training_logs_2/: 所有新训练日志")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()

