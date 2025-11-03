"""
航空发动机数据可视化模块

功能:
1. 时间序列趋势图 - 展示关键传感器特征随时间变化的趋势
2. 特征相关性热力图 - 展示传感器特征之间的相关性
3. 特征分布图 - 展示特征的分布和训练集/测试集的差异

作者: Augment Agent
日期: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示(但图表中使用英文)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """数据可视化类"""
    
    def __init__(self, data_dir: str = "Data", output_dir: str = "results/visualizations"):
        """
        初始化数据可视化器
        
        Args:
            data_dir: 数据文件目录路径
            output_dir: 可视化结果输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_df = None
        self.test_df = None
        self.rul_df = None
        
        # 关键传感器特征
        self.key_sensors = ['NC', 'NRC', 'T50', 'T30', 'P30', 'T24', 'P15', 'PS30']
        
    def load_data(self) -> None:
        """加载原始数据"""
        print("Loading data for visualization...")
        
        self.train_df = pd.read_csv(self.data_dir / 'train_FD001.csv', encoding='gbk')
        self.test_df = pd.read_csv(self.data_dir / 'test_FD001.csv', encoding='gbk')
        self.rul_df = pd.read_csv(self.data_dir / 'RUL_FD001.csv', header=None, names=['RUL'])
        
        # 为训练集计算RUL
        max_cycles = self.train_df.groupby('单元序号')['时间'].max()
        self.train_df['RUL'] = self.train_df.apply(
            lambda row: max_cycles[row['单元序号']] - row['时间'], 
            axis=1
        )
        
        print(f"Data loaded: Train={self.train_df.shape}, Test={self.test_df.shape}")
        
    def select_representative_engines(self, n: int = 5) -> List[int]:
        """
        选择代表性发动机
        
        Args:
            n: 选择的发动机数量
            
        Returns:
            发动机单元序号列表
        """
        # 计算每个发动机的运行周期数
        engine_cycles = self.train_df.groupby('单元序号')['时间'].max().sort_values()
        
        # 选择最短、最长和中等寿命的发动机
        engines = []
        
        # 最短寿命
        engines.append(engine_cycles.index[0])
        
        # 最长寿命
        engines.append(engine_cycles.index[-1])
        
        # 中等寿命
        if n >= 3:
            engines.append(engine_cycles.index[len(engine_cycles)//2])
        
        # 25%分位
        if n >= 4:
            engines.append(engine_cycles.index[len(engine_cycles)//4])
        
        # 75%分位
        if n >= 5:
            engines.append(engine_cycles.index[3*len(engine_cycles)//4])
        
        return sorted(engines[:n])
    
    def plot_time_series_trends(self, engines: List[int] = None, save: bool = True) -> None:
        """
        绘制时间序列趋势图
        
        Args:
            engines: 发动机单元序号列表,如果为None则自动选择
            save: 是否保存图表
        """
        print("\nGenerating time series trend plots...")
        
        if engines is None:
            engines = self.select_representative_engines(n=5)
        
        print(f"Selected engines: {engines}")
        
        # 识别变化明显的传感器特征
        sensor_cols = [col for col in self.key_sensors if col in self.train_df.columns]
        
        # 计算每个传感器的标准差,选择变化最大的
        sensor_stds = {}
        for sensor in sensor_cols:
            sensor_stds[sensor] = self.train_df[sensor].std()
        
        # 按标准差排序,选择前6个
        top_sensors = sorted(sensor_stds.items(), key=lambda x: x[1], reverse=True)[:6]
        selected_sensors = [s[0] for s in top_sensors]
        
        print(f"Selected sensors (by std): {selected_sensors}")
        
        # 为每个发动机创建一个图
        for engine_id in engines:
            engine_data = self.train_df[self.train_df['单元序号'] == engine_id].copy()
            max_cycle = engine_data['时间'].max()
            
            # 创建子图
            n_sensors = len(selected_sensors)
            fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3*n_sensors))
            if n_sensors == 1:
                axes = [axes]
            
            fig.suptitle(f'Engine Unit {engine_id} - Sensor Degradation Trends (Max Cycle: {max_cycle})', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            for idx, sensor in enumerate(selected_sensors):
                ax = axes[idx]
                
                # 原始数据
                x = engine_data['时间'].values
                y = engine_data[sensor].values
                rul = engine_data['RUL'].values
                
                # 绘制原始数据点
                ax.scatter(x, y, alpha=0.5, s=20, label=f'{sensor} (Raw Data)', color='steelblue')
                
                # 多项式拟合曲线
                if len(x) > 3:
                    z = np.polyfit(x, y, 3)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(x.min(), x.max(), 300)
                    y_smooth = p(x_smooth)
                    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'{sensor} (Fitted Trend)', alpha=0.8)
                
                # 设置标签和标题
                ax.set_xlabel('Time Cycle', fontsize=11)
                ax.set_ylabel(f'{sensor} Value', fontsize=11)
                ax.set_title(f'{sensor} Degradation Pattern', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # 添加RUL信息(在右侧y轴)
                ax2 = ax.twinx()
                ax2.plot(x, rul, 'g--', linewidth=1.5, alpha=0.6, label='RUL')
                ax2.set_ylabel('Remaining Useful Life (RUL)', fontsize=11, color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.legend(loc='upper right', fontsize=9)
            
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / f'01_time_series_engine_{engine_id}.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"  Saved: {output_path}")
            
            plt.close()

        print("Time series trend plots completed.")

    def plot_correlation_heatmap(self, save: bool = True) -> None:
        """
        绘制特征相关性热力图

        Args:
            save: 是否保存图表
        """
        print("\nGenerating correlation heatmap...")

        # 选择传感器特征
        sensor_cols = [col for col in self.train_df.columns
                      if col not in ['单元序号', '时间', 'RUL']]

        # 过滤掉常量特征
        valid_sensors = []
        for col in sensor_cols:
            if self.train_df[col].std() > 0:
                valid_sensors.append(col)

        print(f"Computing correlation for {len(valid_sensors)} features...")

        # 计算相关系数矩阵
        corr_matrix = self.train_df[valid_sensors].corr()

        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 12))

        # 绘制热力图
        sns.heatmap(corr_matrix,
                   annot=True,  # 显示数值
                   fmt='.2f',   # 保留2位小数
                   cmap='coolwarm',  # 颜色映射
                   center=0,    # 中心值
                   square=True,  # 正方形单元格
                   linewidths=0.5,  # 网格线宽度
                   cbar_kws={"shrink": 0.8},  # 颜色条大小
                   vmin=-1, vmax=1,  # 值范围
                   ax=ax)

        # 设置标题和标签
        ax.set_title('Feature Correlation Heatmap (Pearson Correlation Coefficient)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)

        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / '02_correlation_heatmap.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")

        plt.close()

        print("Correlation heatmap completed.")

    def plot_feature_distributions(self, save: bool = True) -> None:
        """
        绘制特征分布图

        Args:
            save: 是否保存图表
        """
        print("\nGenerating feature distribution plots...")

        # 选择非常量特征
        sensor_cols = [col for col in self.train_df.columns
                      if col not in ['单元序号', '时间', 'RUL']]

        valid_sensors = []
        for col in sensor_cols:
            if self.train_df[col].std() > 0:
                valid_sensors.append(col)

        print(f"Plotting distributions for {len(valid_sensors)} features...")

        # 1. 绘制直方图和核密度估计
        n_features = len(valid_sensors)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, sensor in enumerate(valid_sensors):
            ax = axes[idx]

            # 训练集分布
            ax.hist(self.train_df[sensor], bins=50, alpha=0.6,
                   label='Train Set', color='steelblue', density=True)

            # 测试集分布
            if sensor in self.test_df.columns:
                ax.hist(self.test_df[sensor], bins=50, alpha=0.6,
                       label='Test Set', color='coral', density=True)

            # 核密度估计
            try:
                self.train_df[sensor].plot(kind='kde', ax=ax, linewidth=2,
                                          color='darkblue', label='Train KDE')
                if sensor in self.test_df.columns:
                    self.test_df[sensor].plot(kind='kde', ax=ax, linewidth=2,
                                             color='darkred', label='Test KDE')
            except:
                pass

            ax.set_title(f'{sensor} Distribution', fontsize=11, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Feature Distributions - Train vs Test',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save:
            output_path = self.output_dir / '03_feature_distributions_histogram.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")

        plt.close()

        # 2. 绘制箱线图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, sensor in enumerate(valid_sensors):
            ax = axes[idx]

            # 准备数据
            data_to_plot = []
            labels = []

            data_to_plot.append(self.train_df[sensor].dropna())
            labels.append('Train')

            if sensor in self.test_df.columns:
                data_to_plot.append(self.test_df[sensor].dropna())
                labels.append('Test')

            # 绘制箱线图
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)

            # 设置颜色
            colors = ['steelblue', 'coral']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_title(f'{sensor} Box Plot', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Feature Distributions - Box Plots',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save:
            output_path = self.output_dir / '03_feature_distributions_boxplot.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")

        plt.close()

        print("Feature distribution plots completed.")

    def generate_all_visualizations(self) -> None:
        """生成所有可视化图表"""
        print("="*80)
        print("Generating All Visualizations")
        print("="*80)

        # 加载数据
        self.load_data()

        # 1. 时间序列趋势图
        self.plot_time_series_trends()

        # 2. 相关性热力图
        self.plot_correlation_heatmap()

        # 3. 特征分布图
        self.plot_feature_distributions()

        print("\n" + "="*80)
        print("All visualizations completed!")
        print(f"Output directory: {self.output_dir}")
        print("="*80)


def main():
    """主函数 - 生成所有可视化图表"""
    visualizer = DataVisualizer(data_dir="Data", output_dir="results/visualizations")
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()


