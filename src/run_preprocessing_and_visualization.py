"""
航空发动机数据预处理与可视化主程序

功能:
1. 执行完整的数据预处理流程
2. 生成所有可视化图表
3. 生成数据处理报告

作者: Augment Agent
日期: 2025
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import DataPreprocessor
from data_visualization import DataVisualizer


def main():
    """主函数 - 执行完整的数据预处理和可视化流程"""
    
    print("="*80)
    print("航空发动机剩余寿命预测 - 数据预处理与可视化")
    print("="*80)
    print()
    
    # ========== 第一部分: 数据可视化 (使用原始数据) ==========
    print("\n" + "="*80)
    print("第一部分: 数据可视化 (基于原始数据)")
    print("="*80)
    
    visualizer = DataVisualizer(data_dir="Data", output_dir="results/visualizations")
    visualizer.generate_all_visualizations()
    
    # ========== 第二部分: 数据预处理 ==========
    print("\n" + "="*80)
    print("第二部分: 数据预处理")
    print("="*80)
    
    preprocessor = DataPreprocessor(data_dir="Data")
    
    # 1. 加载数据
    print("\n[步骤 1/7] 加载数据...")
    preprocessor.load_data()
    
    # 2. 识别并移除常量特征
    print("\n[步骤 2/7] 识别并移除常量特征...")
    preprocessor.identify_constant_features()
    preprocessor.remove_constant_features()
    
    # 3. Z-score标准化
    print("\n[步骤 3/7] Z-score标准化...")
    preprocessor.normalize_features()
    preprocessor.save_normalization_params()
    
    # 4. 保存完整特征版本
    print("\n[步骤 4/7] 保存完整特征版本...")
    preprocessor.save_full_version()
    
    # 5. 识别高相关性特征
    print("\n[步骤 5/7] 识别高相关性特征...")
    preprocessor.identify_high_correlation_features(threshold=0.8)
    
    # 6. 保存降维版本
    print("\n[步骤 6/7] 保存降维特征版本...")
    preprocessor.save_reduced_version()
    
    # 7. 生成并保存报告
    print("\n[步骤 7/7] 生成数据处理报告...")
    preprocessor.save_report_markdown()
    
    # ========== 总结 ==========
    print("\n" + "="*80)
    print("所有任务完成!")
    print("="*80)
    print("\n生成的文件:")
    print("\n【可视化图表】")
    print("  - results/visualizations/01_time_series_engine_*.png")
    print("  - results/visualizations/02_correlation_heatmap.png")
    print("  - results/visualizations/03_feature_distributions_histogram.png")
    print("  - results/visualizations/03_feature_distributions_boxplot.png")
    print("\n【预处理数据】")
    print("  - Data/processed_train_full.csv (完整特征版本)")
    print("  - Data/processed_test_full.csv (完整特征版本)")
    print("  - Data/processed_train_reduced.csv (降维特征版本)")
    print("  - Data/processed_test_reduced.csv (降维特征版本)")
    print("  - Data/normalization_params.json (标准化参数)")
    print("\n【处理报告】")
    print("  - augment_caption/数据预处理报告.md")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

