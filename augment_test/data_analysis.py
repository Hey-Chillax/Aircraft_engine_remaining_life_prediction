"""
数据分析脚本 - 用于分析航空发动机剩余寿命预测数据集
"""
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_dataset():
    """分析数据集并生成详细报告"""
    
    # 读取数据
    print("正在读取数据...")
    df_train = pd.read_csv('Data/train_FD001.csv', encoding='gbk')
    df_test = pd.read_csv('Data/test_FD001.csv', encoding='gbk')
    df_rul = pd.read_csv('Data/RUL_FD001.csv', header=None, names=['RUL'])
    
    print("\n" + "="*80)
    print("航空发动机剩余寿命预测数据集分析报告")
    print("="*80)
    
    # 1. 数据集基本信息
    print("\n【1. 数据集基本信息】")
    print(f"训练集形状: {df_train.shape} (行数, 列数)")
    print(f"测试集形状: {df_test.shape} (行数, 列数)")
    print(f"RUL标签形状: {df_rul.shape} (行数, 列数)")
    
    # 2. 列名信息
    print("\n【2. 特征列信息】")
    print(f"共有 {len(df_train.columns)} 个特征列:")
    for i, col in enumerate(df_train.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 3. 训练集统计信息
    print("\n【3. 训练集统计信息】")
    print(f"唯一发动机单元数: {df_train['单元序号'].nunique()}")
    
    # 每个单元的周期数统计
    cycles_per_unit = df_train.groupby('单元序号').size()
    print(f"\n每个单元的运行周期数统计:")
    print(f"  平均值: {cycles_per_unit.mean():.2f}")
    print(f"  中位数: {cycles_per_unit.median():.2f}")
    print(f"  最大值: {cycles_per_unit.max()}")
    print(f"  最小值: {cycles_per_unit.min()}")
    print(f"  标准差: {cycles_per_unit.std():.2f}")
    
    # 4. 测试集统计信息
    print("\n【4. 测试集统计信息】")
    print(f"唯一发动机单元数: {df_test['单元序号'].nunique()}")
    
    # 每个单元的周期数统计
    test_cycles_per_unit = df_test.groupby('单元序号').size()
    print(f"\n每个单元的运行周期数统计:")
    print(f"  平均值: {test_cycles_per_unit.mean():.2f}")
    print(f"  中位数: {test_cycles_per_unit.median():.2f}")
    print(f"  最大值: {test_cycles_per_unit.max()}")
    print(f"  最小值: {test_cycles_per_unit.min()}")
    print(f"  标准差: {test_cycles_per_unit.std():.2f}")
    
    # 5. RUL标签统计
    print("\n【5. RUL标签统计信息】")
    print(f"RUL标签数量: {len(df_rul)}")
    print(f"RUL统计:")
    print(f"  平均值: {df_rul['RUL'].mean():.2f}")
    print(f"  中位数: {df_rul['RUL'].median():.2f}")
    print(f"  最大值: {df_rul['RUL'].max()}")
    print(f"  最小值: {df_rul['RUL'].min()}")
    print(f"  标准差: {df_rul['RUL'].std():.2f}")
    
    # 6. 传感器特征统计
    print("\n【6. 传感器特征统计】")
    sensor_cols = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 
                   'NF', 'NC', 'EPR', 'PS30', 'PHI', 'NRF', 'NRC', 
                   'BPR', 'FARB', 'W31', 'W32']
    
    print("\n训练集传感器特征统计:")
    print(df_train[sensor_cols].describe())
    
    # 7. 操作条件特征
    print("\n【7. 操作条件特征】")
    op_cols = ['飞行高度', '马赫数', '油门杆角度']
    print("\n训练集操作条件统计:")
    print(df_train[op_cols].describe())
    
    # 8. 数据示例
    print("\n【8. 数据示例】")
    print("\n训练集前5行:")
    print(df_train.head())
    
    print("\n测试集前5行:")
    print(df_test.head())
    
    print("\nRUL标签前10个:")
    print(df_rul.head(10))
    
    # 9. 缺失值检查
    print("\n【9. 数据质量检查】")
    print(f"\n训练集缺失值:")
    missing_train = df_train.isnull().sum()
    if missing_train.sum() == 0:
        print("  无缺失值")
    else:
        print(missing_train[missing_train > 0])
    
    print(f"\n测试集缺失值:")
    missing_test = df_test.isnull().sum()
    if missing_test.sum() == 0:
        print("  无缺失值")
    else:
        print(missing_test[missing_test > 0])
    
    # 10. 特征变化分析
    print("\n【10. 特征变化分析】")
    print("\n检查哪些特征在所有样本中保持不变:")
    constant_features = []
    for col in df_train.columns[2:]:  # 跳过单元序号和时间
        if df_train[col].nunique() == 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"  常量特征: {constant_features}")
    else:
        print("  所有特征都有变化")
    
    # 11. 单个发动机示例
    print("\n【11. 单个发动机运行示例】")
    unit_1 = df_train[df_train['单元序号'] == 1]
    print(f"\n发动机单元1的运行信息:")
    print(f"  总运行周期数: {len(unit_1)}")
    print(f"  时间范围: {unit_1['时间'].min()} - {unit_1['时间'].max()}")
    print(f"  T30温度范围: {unit_1['T30'].min():.2f} - {unit_1['T30'].max():.2f}")
    print(f"  T50温度范围: {unit_1['T50'].min():.2f} - {unit_1['T50'].max():.2f}")
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)


if __name__ == "__main__":
    analyze_dataset()

