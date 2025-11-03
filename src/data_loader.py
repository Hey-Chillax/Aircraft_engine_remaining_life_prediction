"""
数据加载和预处理模块

功能:
1. 加载训练集和测试集数据
2. 构建时间序列窗口（用于LSTM和Transformer）
3. 提取统计特征（用于随机森林）
4. 数据划分（训练集/验证集）

作者: Augment Agent
日期: 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, data_dir: str = "Data", random_state: int = 42):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
            random_state: 随机种子
        """
        self.data_dir = data_dir
        self.random_state = random_state
        
    def load_data(self, data_version: str = "full") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载训练集、测试集和RUL标签
        
        Args:
            data_version: 数据版本，"full"或"reduced"
        
        Returns:
            train_df: 训练集DataFrame
            test_df: 测试集DataFrame
            rul_df: 测试集RUL标签DataFrame
        """
        logger.info(f"加载{data_version}版本数据...")
        
        if data_version == "full":
            train_path = f"{self.data_dir}/processed_train_full.csv"
            test_path = f"{self.data_dir}/processed_test_full.csv"
        else:
            train_path = f"{self.data_dir}/processed_train_reduced.csv"
            test_path = f"{self.data_dir}/processed_test_reduced.csv"
        
        rul_path = f"{self.data_dir}/RUL_FD001.csv"
        
        # 加载数据
        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
        rul_df = pd.read_csv(rul_path, encoding='utf-8', header=None, names=['RUL'])
        
        logger.info(f"训练集形状: {train_df.shape}")
        logger.info(f"测试集形状: {test_df.shape}")
        logger.info(f"RUL标签形状: {rul_df.shape}")
        
        return train_df, test_df, rul_df
    
    def split_train_val(self, train_df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按发动机单元划分训练集和验证集
        
        Args:
            train_df: 训练集DataFrame
            val_ratio: 验证集比例
        
        Returns:
            train_split: 训练集
            val_split: 验证集
        """
        logger.info(f"按发动机单元划分训练集和验证集（验证集比例={val_ratio}）...")
        
        # 获取所有发动机单元序号
        engine_ids = train_df['单元序号'].unique()
        
        # 划分发动机单元
        train_engines, val_engines = train_test_split(
            engine_ids, 
            test_size=val_ratio, 
            random_state=self.random_state
        )
        
        # 根据发动机单元划分数据
        train_split = train_df[train_df['单元序号'].isin(train_engines)].copy()
        val_split = train_df[train_df['单元序号'].isin(val_engines)].copy()
        
        logger.info(f"训练集: {len(train_engines)}台发动机, {len(train_split)}条记录")
        logger.info(f"验证集: {len(val_engines)}台发动机, {len(val_split)}条记录")
        
        return train_split, val_split
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 50, 
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        为LSTM和Transformer创建时间序列窗口
        
        Args:
            df: 输入DataFrame
            sequence_length: 窗口大小
            stride: 滑动步长
        
        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,) - 窗口最后一个时间点的RUL
        """
        logger.info(f"创建时间序列窗口（窗口大小={sequence_length}, 步长={stride}）...")
        
        X, y = [], []
        
        # 获取特征列（排除单元序号、时间、RUL）
        feature_cols = [col for col in df.columns if col not in ['单元序号', '时间', 'RUL']]
        
        # 为每台发动机创建窗口
        for engine_id in df['单元序号'].unique():
            engine_data = df[df['单元序号'] == engine_id].sort_values('时间')
            
            # 提取特征和标签
            features = engine_data[feature_cols].values
            rul = engine_data['RUL'].values
            
            # 滑动窗口
            for i in range(0, len(features) - sequence_length + 1, stride):
                X.append(features[i:i+sequence_length])
                y.append(rul[i+sequence_length-1])  # 窗口最后一个时间点的RUL
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        logger.info(f"生成{len(X)}个样本, 特征形状: {X.shape}")
        
        return X, y
    
    def create_test_sequences(self, test_df: pd.DataFrame, sequence_length: int = 50) -> Tuple[np.ndarray, List[int]]:
        """
        为测试集每台发动机创建最后一个窗口
        
        Args:
            test_df: 测试集DataFrame
            sequence_length: 窗口大小
        
        Returns:
            X_test: (n_engines, sequence_length, n_features)
            engine_ids: 发动机单元序号列表
        """
        logger.info(f"为测试集创建最后一个窗口（窗口大小={sequence_length}）...")
        
        X_test = []
        engine_ids = []
        
        # 获取特征列
        feature_cols = [col for col in test_df.columns if col not in ['单元序号', '时间', 'RUL']]
        
        for engine_id in sorted(test_df['单元序号'].unique()):
            engine_data = test_df[test_df['单元序号'] == engine_id].sort_values('时间')
            features = engine_data[feature_cols].values
            
            # 取最后sequence_length个周期
            if len(features) >= sequence_length:
                X_test.append(features[-sequence_length:])
            else:
                # 如果不足sequence_length个周期，用0填充
                padded = np.zeros((sequence_length, features.shape[1]), dtype=np.float32)
                padded[-len(features):] = features
                X_test.append(padded)
            
            engine_ids.append(engine_id)
        
        X_test = np.array(X_test, dtype=np.float32)
        
        logger.info(f"测试集: {len(X_test)}台发动机, 特征形状: {X_test.shape}")
        
        return X_test, engine_ids
    
    def add_statistical_features(self, df: pd.DataFrame, window_size: int = 50) -> pd.DataFrame:
        """
        为随机森林添加滑动窗口统计特征
        
        Args:
            df: 输入DataFrame
            window_size: 滑动窗口大小
        
        Returns:
            df_with_stats: 添加统计特征后的DataFrame
        """
        logger.info(f"添加滑动窗口统计特征（窗口大小={window_size}）...")
        
        df_with_stats = df.copy()
        
        # 获取传感器特征列（排除单元序号、时间、RUL）
        sensor_cols = [col for col in df.columns if col not in ['单元序号', '时间', 'RUL']]
        
        # 为每台发动机计算统计特征
        for engine_id in df['单元序号'].unique():
            mask = df_with_stats['单元序号'] == engine_id
            engine_data = df_with_stats[mask].sort_values('时间')
            
            for col in sensor_cols:
                # 滑动窗口均值
                df_with_stats.loc[mask, f'{col}_mean'] = engine_data[col].rolling(
                    window=window_size, min_periods=1).mean().values
                
                # 滑动窗口标准差
                df_with_stats.loc[mask, f'{col}_std'] = engine_data[col].rolling(
                    window=window_size, min_periods=1).std().fillna(0).values
                
                # 滑动窗口最大值
                df_with_stats.loc[mask, f'{col}_max'] = engine_data[col].rolling(
                    window=window_size, min_periods=1).max().values
                
                # 滑动窗口最小值
                df_with_stats.loc[mask, f'{col}_min'] = engine_data[col].rolling(
                    window=window_size, min_periods=1).min().values
        
        # 统计新增特征数
        new_feature_count = len(df_with_stats.columns) - len(df.columns)
        logger.info(f"新增{new_feature_count}个统计特征, 总特征数: {len(df_with_stats.columns)}")
        
        return df_with_stats
    
    def prepare_rf_data(self, df: pd.DataFrame, window_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        为随机森林准备数据（添加统计特征）
        
        Args:
            df: 输入DataFrame
            window_size: 滑动窗口大小
        
        Returns:
            X: (n_samples, n_features)
            y: (n_samples,)
        """
        # 添加统计特征
        df_with_stats = self.add_statistical_features(df, window_size)
        
        # 提取特征和标签
        feature_cols = [col for col in df_with_stats.columns if col not in ['单元序号', '时间', 'RUL']]
        X = df_with_stats[feature_cols].values
        y = df_with_stats['RUL'].values
        
        logger.info(f"随机森林数据准备完成: X形状={X.shape}, y形状={y.shape}")
        
        return X, y
    
    def prepare_rf_test_data(self, test_df: pd.DataFrame, window_size: int = 50) -> Tuple[np.ndarray, List[int]]:
        """
        为随机森林准备测试集数据（每台发动机取最后一个时间点）
        
        Args:
            test_df: 测试集DataFrame
            window_size: 滑动窗口大小
        
        Returns:
            X_test: (n_engines, n_features)
            engine_ids: 发动机单元序号列表
        """
        # 添加统计特征
        df_with_stats = self.add_statistical_features(test_df, window_size)
        
        X_test = []
        engine_ids = []
        
        # 获取特征列
        feature_cols = [col for col in df_with_stats.columns if col not in ['单元序号', '时间', 'RUL']]
        
        # 为每台发动机取最后一个时间点
        for engine_id in sorted(test_df['单元序号'].unique()):
            engine_data = df_with_stats[df_with_stats['单元序号'] == engine_id].sort_values('时间')
            last_row = engine_data[feature_cols].iloc[-1].values
            X_test.append(last_row)
            engine_ids.append(engine_id)
        
        X_test = np.array(X_test, dtype=np.float32)
        
        logger.info(f"随机森林测试集准备完成: {len(X_test)}台发动机, 特征形状: {X_test.shape}")
        
        return X_test, engine_ids

