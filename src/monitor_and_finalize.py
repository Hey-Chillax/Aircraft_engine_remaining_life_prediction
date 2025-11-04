"""
监控训练进度并在完成后自动生成报告

功能:
1. 检查训练进程状态
2. 统计已完成的模型数量
3. 训练完成后自动生成可视化和报告

作者: Augment Agent
日期: 2025
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_training_status():
    """检查训练状态"""
    # 检查进程是否还在运行
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        
        is_running = "train_all_models_v2.py" in result.stdout
        
        # 统计已完成的模型数量
        log_dir = Path("results/training_logs_2")
        completed_models = len(list(log_dir.glob("*.json"))) if log_dir.exists() else 0
        
        return is_running, completed_models
        
    except Exception as e:
        logger.error(f"检查训练状态失败: {str(e)}")
        return False, 0


def generate_visualizations():
    """生成所有可视化图表"""
    logger.info("=" * 80)
    logger.info("开始生成可视化图表...")
    logger.info("=" * 80)
    
    try:
        # 运行可视化脚本
        result = subprocess.run(
            ["python3", "src/visualize_results_v2.py"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("可视化图表生成成功！")
        logger.info(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"可视化生成失败: {str(e)}")
        logger.error(e.stderr)
        return False


def generate_report():
    """生成总结报告"""
    logger.info("=" * 80)
    logger.info("开始生成总结报告...")
    logger.info("=" * 80)
    
    try:
        # 运行报告生成脚本
        result = subprocess.run(
            ["python3", "src/generate_report_v2.py"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("总结报告生成成功！")
        logger.info(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"报告生成失败: {str(e)}")
        logger.error(e.stderr)
        return False


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始监控训练进度...")
    logger.info("=" * 80)
    
    total_models = 32
    check_interval = 300  # 5分钟检查一次
    
    while True:
        is_running, completed_models = check_training_status()
        
        logger.info(f"训练状态: {'运行中' if is_running else '已停止'}")
        logger.info(f"已完成模型: {completed_models}/{total_models}")
        
        if not is_running and completed_models >= total_models:
            logger.info("=" * 80)
            logger.info("所有模型训练完成！开始生成最终报告...")
            logger.info("=" * 80)
            
            # 生成可视化
            if generate_visualizations():
                logger.info("✓ 可视化图表生成完成")
            else:
                logger.error("✗ 可视化图表生成失败")
            
            # 生成报告
            if generate_report():
                logger.info("✓ 总结报告生成完成")
            else:
                logger.error("✗ 总结报告生成失败")
            
            logger.info("=" * 80)
            logger.info("所有任务完成！")
            logger.info("=" * 80)
            break
        
        elif not is_running and completed_models < total_models:
            logger.warning(f"训练进程已停止，但只完成了 {completed_models}/{total_models} 个模型")
            logger.warning("请检查训练日志: training_v2_output.log")
            break
        
        else:
            # 继续等待
            logger.info(f"等待 {check_interval} 秒后再次检查...")
            time.sleep(check_interval)


if __name__ == "__main__":
    main()

