"""
航空发动机数据预处理模块

功能:
1. 加载训练集、测试集和RUL标签数据
2. 过滤常量特征
3. Z-score标准化
4. 创建完整特征版本和降维特征版本数据集
5. 生成数据处理报告

作者: Augment Agent
日期: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import json


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, data_dir: str = "Data"):
        """
        初始化数据预处理器
        
        Args:
            data_dir: 数据文件目录路径
        """
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.rul_df = None
        
        # 预处理参数
        self.constant_features = []
        self.normalization_params = {}
        self.high_corr_features = []
        
        # 传感器特征列表
        self.sensor_cols = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 
                           'NF', 'NC', 'EPR', 'PS30', 'PHI', 'NRF', 'NRC', 
                           'BPR', 'FARB', 'HT_BLEED', 'NF_DMD', 'PCNFR_DMD', 
                           'W31', 'W32']
        
        # 操作条件特征
        self.op_cols = ['飞行高度', '马赫数', '油门杆角度']
        
    def load_data(self) -> None:
        """加载训练集、测试集和RUL标签数据"""
        print("正在加载数据...")
        
        # 加载训练集
        self.train_df = pd.read_csv(
            self.data_dir / 'train_FD001.csv', 
            encoding='gbk'
        )
        print(f"训练集加载完成: {self.train_df.shape}")
        
        # 加载测试集
        self.test_df = pd.read_csv(
            self.data_dir / 'test_FD001.csv', 
            encoding='gbk'
        )
        print(f"测试集加载完成: {self.test_df.shape}")
        
        # 加载RUL标签
        self.rul_df = pd.read_csv(
            self.data_dir / 'RUL_FD001.csv', 
            header=None, 
            names=['RUL']
        )
        print(f"RUL标签加载完成: {self.rul_df.shape}")
        
        # 为训练集计算RUL标签
        self._calculate_train_rul()
        
    def _calculate_train_rul(self) -> None:
        """为训练集计算RUL标签"""
        print("\n正在为训练集计算RUL标签...")
        
        # 计算每个发动机的最大周期数
        max_cycles = self.train_df.groupby('单元序号')['时间'].max()
        
        # 计算RUL = 最大周期数 - 当前周期数
        self.train_df['RUL'] = self.train_df.apply(
            lambda row: max_cycles[row['单元序号']] - row['时间'], 
            axis=1
        )
        
        print(f"训练集RUL标签计算完成")
        print(f"RUL统计: 最小值={self.train_df['RUL'].min()}, "
              f"最大值={self.train_df['RUL'].max()}, "
              f"平均值={self.train_df['RUL'].mean():.2f}")
        
    def get_data_statistics(self) -> Dict:
        """获取原始数据统计信息"""
        stats = {
            'train_shape': self.train_df.shape,
            'test_shape': self.test_df.shape,
            'train_engines': self.train_df['单元序号'].nunique(),
            'test_engines': self.test_df['单元序号'].nunique(),
            'total_features': len(self.train_df.columns),
            'sensor_features': len(self.sensor_cols),
            'operation_features': len(self.op_cols),
        }
        return stats
        
    def identify_constant_features(self, std_threshold: float = 0.0) -> List[str]:
        """
        识别常量特征
        
        Args:
            std_threshold: 标准差阈值,默认为0
            
        Returns:
            常量特征列表
        """
        print("\n正在识别常量特征...")
        
        # 跳过标识列和RUL列
        feature_cols = [col for col in self.train_df.columns 
                       if col not in ['单元序号', '时间', 'RUL']]
        
        constant_features = []
        
        for col in feature_cols:
            # 方法1: 检查唯一值数量
            nunique = self.train_df[col].nunique()
            
            # 方法2: 检查标准差
            std = self.train_df[col].std()
            
            # 如果唯一值只有1个或标准差为0,则认为是常量特征
            if nunique == 1 or std == std_threshold:
                constant_features.append(col)
                print(f"  常量特征: {col} (唯一值={nunique}, 标准差={std:.6f})")
        
        self.constant_features = constant_features
        print(f"\n共识别出 {len(constant_features)} 个常量特征")
        
        return constant_features
    
    def remove_constant_features(self) -> None:
        """移除常量特征"""
        print("\n正在移除常量特征...")
        
        if not self.constant_features:
            print("未发现常量特征,跳过移除步骤")
            return
        
        # 从训练集和测试集中移除常量特征
        self.train_df = self.train_df.drop(columns=self.constant_features)
        self.test_df = self.test_df.drop(columns=self.constant_features)
        
        # 更新传感器特征列表
        self.sensor_cols = [col for col in self.sensor_cols 
                           if col not in self.constant_features]
        
        print(f"已移除 {len(self.constant_features)} 个常量特征")
        print(f"剩余特征数: {len(self.train_df.columns)}")
        
    def normalize_features(self) -> None:
        """
        使用Z-score标准化特征
        使用训练集的均值和标准差来标准化训练集和测试集
        """
        print("\n正在进行Z-score标准化...")
        
        # 需要标准化的特征列(排除标识列和RUL)
        feature_cols = [col for col in self.train_df.columns 
                       if col not in ['单元序号', '时间', 'RUL']]
        
        # 计算训练集的均值和标准差
        for col in feature_cols:
            mean = self.train_df[col].mean()
            std = self.train_df[col].std()
            
            # 保存标准化参数
            self.normalization_params[col] = {
                'mean': float(mean),
                'std': float(std)
            }
            
            # 标准化训练集和测试集
            if std > 0:  # 避免除以0
                self.train_df[col] = (self.train_df[col] - mean) / std
                self.test_df[col] = (self.test_df[col] - mean) / std
            else:
                print(f"  警告: {col} 的标准差为0,跳过标准化")
        
        print(f"已标准化 {len(feature_cols)} 个特征")
        
    def save_normalization_params(self, output_path: str = "Data/normalization_params.json") -> None:
        """保存标准化参数到JSON文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.normalization_params, f, indent=2, ensure_ascii=False)
        
        print(f"\n标准化参数已保存到: {output_path}")
        
    def save_full_version(self, 
                         train_path: str = "Data/processed_train_full.csv",
                         test_path: str = "Data/processed_test_full.csv") -> None:
        """保存完整特征版本数据集"""
        print("\n正在保存完整特征版本数据集...")
        
        train_path = Path(train_path)
        test_path = Path(test_path)
        
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.train_df.to_csv(train_path, index=False, encoding='utf-8')
        self.test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        print(f"训练集已保存到: {train_path} (形状: {self.train_df.shape})")
        print(f"测试集已保存到: {test_path} (形状: {self.test_df.shape})")
        
    def identify_high_correlation_features(self, threshold: float = 0.8) -> List[str]:
        """
        识别高相关性特征
        
        Args:
            threshold: 相关系数阈值,默认为0.8
            
        Returns:
            需要移除的高相关性特征列表
        """
        print(f"\n正在识别高相关性特征 (阈值={threshold})...")
        
        # 只计算传感器特征的相关性
        feature_cols = [col for col in self.sensor_cols 
                       if col in self.train_df.columns]
        
        # 计算相关系数矩阵
        corr_matrix = self.train_df[feature_cols].corr().abs()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        features_to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    high_corr_pairs.append((feat1, feat2, corr_value))
                    
                    # 选择要移除的特征(保留标准差更大的)
                    std1 = self.train_df[feat1].std()
                    std2 = self.train_df[feat2].std()
                    
                    if std1 >= std2:
                        features_to_remove.add(feat2)
                        print(f"  {feat1} <-> {feat2}: {corr_value:.3f} (移除 {feat2}, std1={std1:.3f}, std2={std2:.3f})")
                    else:
                        features_to_remove.add(feat1)
                        print(f"  {feat1} <-> {feat2}: {corr_value:.3f} (移除 {feat1}, std1={std1:.3f}, std2={std2:.3f})")
        
        self.high_corr_features = list(features_to_remove)
        print(f"\n共识别出 {len(high_corr_pairs)} 对高相关性特征")
        print(f"需要移除 {len(self.high_corr_features)} 个特征: {self.high_corr_features}")

        return self.high_corr_features

    def save_reduced_version(self,
                            train_path: str = "Data/processed_train_reduced.csv",
                            test_path: str = "Data/processed_test_reduced.csv") -> None:
        """保存降维特征版本数据集"""
        print("\n正在保存降维特征版本数据集...")

        # 创建副本并移除高相关性特征
        train_reduced = self.train_df.drop(columns=self.high_corr_features, errors='ignore')
        test_reduced = self.test_df.drop(columns=self.high_corr_features, errors='ignore')

        train_path = Path(train_path)
        test_path = Path(test_path)

        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        train_reduced.to_csv(train_path, index=False, encoding='utf-8')
        test_reduced.to_csv(test_path, index=False, encoding='utf-8')

        print(f"训练集已保存到: {train_path} (形状: {train_reduced.shape})")
        print(f"测试集已保存到: {test_path} (形状: {test_reduced.shape})")
        print(f"相比完整版本,减少了 {len(self.high_corr_features)} 个特征")

    def generate_report(self) -> Dict:
        """生成数据处理报告"""
        print("\n正在生成数据处理报告...")

        report = {
            '原始数据统计': self.get_data_statistics(),
            '常量特征': {
                '数量': len(self.constant_features),
                '特征列表': self.constant_features
            },
            '标准化参数': {
                '特征数量': len(self.normalization_params),
                '参数示例': {k: v for k, v in list(self.normalization_params.items())[:3]}
            },
            '高相关性特征': {
                '数量': len(self.high_corr_features),
                '特征列表': self.high_corr_features
            },
            '数据集版本': {
                '完整版特征数': len(self.train_df.columns),
                '降维版特征数': len(self.train_df.columns) - len(self.high_corr_features)
            }
        }

        return report

    def save_report_markdown(self, output_path: str = "augment_caption/数据预处理报告.md") -> None:
        """保存数据处理报告为Markdown格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 航空发动机数据预处理报告\n\n")
            f.write("---\n\n")

            # 原始数据统计
            f.write("## 1. 原始数据统计信息\n\n")
            stats = report['原始数据统计']
            f.write(f"- **训练集形状**: {stats['train_shape']}\n")
            f.write(f"- **测试集形状**: {stats['test_shape']}\n")
            f.write(f"- **训练集发动机数量**: {stats['train_engines']}\n")
            f.write(f"- **测试集发动机数量**: {stats['test_engines']}\n")
            f.write(f"- **总特征数**: {stats['total_features']}\n")
            f.write(f"- **传感器特征数**: {stats['sensor_features']}\n")
            f.write(f"- **操作条件特征数**: {stats['operation_features']}\n\n")

            # 常量特征
            f.write("## 2. 移除的常量特征\n\n")
            const_info = report['常量特征']
            f.write(f"共识别并移除了 **{const_info['数量']}** 个常量特征:\n\n")
            if const_info['特征列表']:
                for feat in const_info['特征列表']:
                    f.write(f"- {feat}\n")
            else:
                f.write("- 无常量特征\n")
            f.write("\n")

            # 标准化参数
            f.write("## 3. Z-score标准化参数\n\n")
            norm_info = report['标准化参数']
            f.write(f"共对 **{norm_info['特征数量']}** 个特征进行了Z-score标准化。\n\n")
            f.write("标准化公式: `z = (x - mean) / std`\n\n")
            f.write("**参数示例** (前3个特征):\n\n")
            f.write("| 特征 | 均值 (mean) | 标准差 (std) |\n")
            f.write("|------|-------------|-------------|\n")
            for feat, params in norm_info['参数示例'].items():
                f.write(f"| {feat} | {params['mean']:.4f} | {params['std']:.4f} |\n")
            f.write("\n")
            f.write("完整的标准化参数已保存到: `Data/normalization_params.json`\n\n")

            # 高相关性特征
            f.write("## 4. 移除的高相关性特征\n\n")
            corr_info = report['高相关性特征']
            f.write(f"使用相关系数阈值 **≥ 0.8** 识别高相关性特征。\n\n")
            f.write(f"共识别并移除了 **{corr_info['数量']}** 个高相关性特征:\n\n")
            if corr_info['特征列表']:
                for feat in corr_info['特征列表']:
                    f.write(f"- {feat}\n")
            else:
                f.write("- 无高相关性特征需要移除\n")
            f.write("\n")
            f.write("**特征选择策略**:\n")
            f.write("- 对于高相关的特征对,保留标准差更大的特征\n")
            f.write("- 标准差更大意味着特征变化更明显,包含更多信息\n\n")

            # 数据集版本对比
            f.write("## 5. 数据集版本对比\n\n")
            version_info = report['数据集版本']
            f.write("| 版本 | 特征数量 | 文件路径 |\n")
            f.write("|------|----------|----------|\n")
            f.write(f"| 完整特征版本 | {version_info['完整版特征数']} | `Data/processed_train_full.csv` / `Data/processed_test_full.csv` |\n")
            f.write(f"| 降维特征版本 | {version_info['降维版特征数']} | `Data/processed_train_reduced.csv` / `Data/processed_test_reduced.csv` |\n")
            f.write("\n")

            # 总结
            f.write("## 6. 预处理流程总结\n\n")
            f.write("数据预处理按以下顺序执行:\n\n")
            f.write("1. **数据加载**: 加载训练集、测试集和RUL标签\n")
            f.write("2. **RUL计算**: 为训练集计算RUL标签 (最大周期数 - 当前周期数)\n")
            f.write("3. **常量特征过滤**: 移除标准差为0或唯一值为1的特征\n")
            f.write("4. **Z-score标准化**: 使用训练集的均值和标准差标准化所有特征\n")
            f.write("5. **保存完整版本**: 保存包含所有预处理后特征的数据集\n")
            f.write("6. **高相关性特征识别**: 识别相关系数≥0.8的特征对\n")
            f.write("7. **保存降维版本**: 移除高相关性特征后保存数据集\n\n")

            f.write("---\n\n")
            f.write("**报告生成时间**: 2025年\n")
            f.write("**生成工具**: Augment Agent\n")

        print(f"数据处理报告已保存到: {output_path}")


def main():
    """主函数 - 执行完整的数据预处理流程"""
    print("="*80)
    print("航空发动机数据预处理")
    print("="*80)

    # 创建预处理器
    preprocessor = DataPreprocessor(data_dir="Data")

    # 1. 加载数据
    preprocessor.load_data()

    # 2. 识别并移除常量特征
    preprocessor.identify_constant_features()
    preprocessor.remove_constant_features()

    # 3. Z-score标准化
    preprocessor.normalize_features()
    preprocessor.save_normalization_params()

    # 4. 保存完整特征版本
    preprocessor.save_full_version()

    # 5. 识别高相关性特征并保存降维版本
    preprocessor.identify_high_correlation_features(threshold=0.8)
    preprocessor.save_reduced_version()

    # 6. 生成并保存报告
    preprocessor.save_report_markdown()

    print("\n" + "="*80)
    print("数据预处理完成!")
    print("="*80)


if __name__ == "__main__":
    main()


