"""
主训练脚本 - 训练所有12个模型

功能:
1. 依次训练随机森林、LSTM、Transformer模型
2. 每个模型训练完整特征和降维特征两个版本
3. 每个版本训练带早停和不带早停两种模式
4. 自动生成可视化图表和总结报告

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

from train_random_forest import train_random_forest
from train_deep_learning import train_deep_model
from visualize_results import plot_training_curves, plot_all_models_comparison, plot_prediction_scatter
from generate_report import generate_summary_report

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数 - 训练所有模型"""
    
    logger.info("=" * 100)
    logger.info("开始训练所有模型 - 共12个模型")
    logger.info("=" * 100)
    
    total_start_time = time.time()
    
    # 创建所有必要的目录
    Path("models").mkdir(exist_ok=True)
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    Path("results/training_logs").mkdir(parents=True, exist_ok=True)
    Path("results/training_curves").mkdir(parents=True, exist_ok=True)
    Path("augment_caption").mkdir(exist_ok=True)
    
    # 训练计数器
    completed_models = 0
    total_models = 12
    
    # ========== 第一部分: 训练随机森林模型 (2个模型) ==========
    logger.info("\n" + "=" * 100)
    logger.info("第一部分: 训练随机森林模型")
    logger.info("=" * 100)
    
    try:
        # 1. 随机森林 - 完整特征版本
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练随机森林 - 完整特征版本")
        train_random_forest(data_version="full", window_size=50, random_state=42)
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 2. 随机森林 - 降维特征版本
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练随机森林 - 降维特征版本")
        train_random_forest(data_version="reduced", window_size=50, random_state=42)
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
    except Exception as e:
        logger.error(f"随机森林训练失败: {str(e)}")
        logger.exception(e)
    
    # ========== 第二部分: 训练LSTM模型 (4个模型) ==========
    logger.info("\n" + "=" * 100)
    logger.info("第二部分: 训练LSTM模型")
    logger.info("=" * 100)
    
    try:
        # 3. LSTM - 完整特征 - 带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练LSTM - 完整特征 - 带早停")
        train_deep_model(
            model_type="LSTM",
            data_version="full",
            use_early_stopping=True,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.001,
            warmup_epochs=0,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("LSTM", "full", "with_early_stopping")
        plot_prediction_scatter("LSTM", "full", "with_early_stopping")
        
        # 4. LSTM - 完整特征 - 不带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练LSTM - 完整特征 - 不带早停")
        train_deep_model(
            model_type="LSTM",
            data_version="full",
            use_early_stopping=False,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.001,
            warmup_epochs=0,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("LSTM", "full", "no_early_stopping")
        plot_prediction_scatter("LSTM", "full", "no_early_stopping")
        
        # 5. LSTM - 降维特征 - 带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练LSTM - 降维特征 - 带早停")
        train_deep_model(
            model_type="LSTM",
            data_version="reduced",
            use_early_stopping=True,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.001,
            warmup_epochs=0,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("LSTM", "reduced", "with_early_stopping")
        plot_prediction_scatter("LSTM", "reduced", "with_early_stopping")
        
        # 6. LSTM - 降维特征 - 不带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练LSTM - 降维特征 - 不带早停")
        train_deep_model(
            model_type="LSTM",
            data_version="reduced",
            use_early_stopping=False,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.001,
            warmup_epochs=0,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("LSTM", "reduced", "no_early_stopping")
        plot_prediction_scatter("LSTM", "reduced", "no_early_stopping")
        
    except Exception as e:
        logger.error(f"LSTM训练失败: {str(e)}")
        logger.exception(e)
    
    # ========== 第三部分: 训练Transformer模型 (4个模型) ==========
    logger.info("\n" + "=" * 100)
    logger.info("第三部分: 训练Transformer模型")
    logger.info("=" * 100)
    
    try:
        # 7. Transformer - 完整特征 - 带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练Transformer - 完整特征 - 带早停")
        train_deep_model(
            model_type="Transformer",
            data_version="full",
            use_early_stopping=True,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.0005,
            warmup_epochs=20,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("Transformer", "full", "with_early_stopping")
        plot_prediction_scatter("Transformer", "full", "with_early_stopping")
        
        # 8. Transformer - 完整特征 - 不带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练Transformer - 完整特征 - 不带早停")
        train_deep_model(
            model_type="Transformer",
            data_version="full",
            use_early_stopping=False,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.0005,
            warmup_epochs=20,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("Transformer", "full", "no_early_stopping")
        plot_prediction_scatter("Transformer", "full", "no_early_stopping")
        
        # 9. Transformer - 降维特征 - 带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练Transformer - 降维特征 - 带早停")
        train_deep_model(
            model_type="Transformer",
            data_version="reduced",
            use_early_stopping=True,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.0005,
            warmup_epochs=20,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("Transformer", "reduced", "with_early_stopping")
        plot_prediction_scatter("Transformer", "reduced", "with_early_stopping")
        
        # 10. Transformer - 降维特征 - 不带早停
        logger.info(f"\n[{completed_models + 1}/{total_models}] 训练Transformer - 降维特征 - 不带早停")
        train_deep_model(
            model_type="Transformer",
            data_version="reduced",
            use_early_stopping=False,
            sequence_length=50,
            batch_size=64,
            epochs=300,
            learning_rate=0.0005,
            warmup_epochs=20,
            patience=20,
            device="cuda",
            random_state=42
        )
        completed_models += 1
        logger.info(f"✓ 完成 [{completed_models}/{total_models}]")
        
        # 绘制训练曲线
        plot_training_curves("Transformer", "reduced", "no_early_stopping")
        plot_prediction_scatter("Transformer", "reduced", "no_early_stopping")
        
    except Exception as e:
        logger.error(f"Transformer训练失败: {str(e)}")
        logger.exception(e)
    
    # ========== 第四部分: 生成总结报告和对比图表 ==========
    logger.info("\n" + "=" * 100)
    logger.info("第四部分: 生成总结报告和对比图表")
    logger.info("=" * 100)
    
    try:
        # 绘制所有模型对比图
        plot_all_models_comparison()
        
        # 生成总结报告
        generate_summary_report()
        
    except Exception as e:
        logger.error(f"生成报告失败: {str(e)}")
        logger.exception(e)
    
    # ========== 总结 ==========
    total_time = time.time() - total_start_time
    
    logger.info("\n" + "=" * 100)
    logger.info("所有任务完成!")
    logger.info("=" * 100)
    logger.info(f"完成模型数: {completed_models}/{total_models}")
    logger.info(f"总耗时: {total_time/3600:.2f}小时 ({total_time/60:.2f}分钟)")
    logger.info("\n生成的文件:")
    logger.info("  - models/: 所有模型权重")
    logger.info("  - results/predictions/: 所有预测结果")
    logger.info("  - results/training_logs/: 所有训练日志")
    logger.info("  - results/training_curves/: 所有训练曲线图")
    logger.info("  - augment_caption/模型训练总结报告.md: 总结报告")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()

