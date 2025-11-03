"""
模型定义模块

功能:
1. LSTM模型定义
2. Transformer模型定义
3. 位置编码实现

作者: Augment Agent
日期: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class LSTMModel(nn.Module):
    """LSTM模型用于RUL预测"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层: 128 → 64 → 32 → 1
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            output: (batch_size,)
        """
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 全连接层
        out = self.fc1(last_output)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out.squeeze()  # (batch_size,)


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            dropout: Dropout比例
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer模型用于RUL预测"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 512, 
                 dropout: float = 0.1, max_seq_len: int = 100):
        """
        初始化Transformer模型
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            max_seq_len: 最大序列长度
        """
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出层
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            output: (batch_size,)
        """
        # 输入投影
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, seq, d_model)
        
        # 取最后一个时间步
        x = x[:, -1, :]  # (batch, d_model)
        
        # 输出层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze()  # (batch,)


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试LSTM模型
    print("=" * 80)
    print("测试LSTM模型")
    print("=" * 80)
    
    lstm_model = LSTMModel(input_size=17, hidden_size=128, num_layers=3, dropout=0.2)
    print(f"LSTM模型参数数量: {count_parameters(lstm_model):,}")
    
    # 测试前向传播
    x = torch.randn(32, 50, 17)  # (batch_size, seq_len, input_size)
    output = lstm_model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n" + "=" * 80)
    print("测试Transformer模型")
    print("=" * 80)
    
    transformer_model = TransformerModel(
        input_size=17, d_model=128, nhead=8, num_layers=4, 
        dim_feedforward=512, dropout=0.1
    )
    print(f"Transformer模型参数数量: {count_parameters(transformer_model):,}")
    
    # 测试前向传播
    output = transformer_model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

