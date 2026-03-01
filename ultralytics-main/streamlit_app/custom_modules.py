"""
自定义模块补丁 - 添加缺失的ECA模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import math

# 定义ECA模块 - 高效通道注意力机制
class ECA(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, channels=64, gamma=2, b=1):
        super().__init__()
        try:
            # 确保channels参数是有效的
            channels = int(channels) if channels else 64
            
            # 计算kernel大小
            t = int(abs((math.log(channels, 2) + b) / gamma))
            k = t if t % 2 else t + 1
            
            # 创建必需的层
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # 确保卷积层被正确创建和命名
            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
            
            # 调试信息
            print(f"ECA模块初始化成功，channels={channels}, kernel_size={k}")
            print(f"ECA.conv属性: {self.conv}")
            
            # 添加备用属性，以防止某些名称不匹配的情况
            self.k = k
            self.gate_channels = channels
            self.conv1d = self.conv  # 备用名称
        except Exception as e:
            print(f"ECA模块初始化错误: {e}")
            # 确保基本属性存在，即使出错
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            self.conv1d = self.conv
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            # 调试信息
            if not hasattr(self, 'conv'):
                print("错误: ECA模块缺少conv属性")
                # 如果找不到conv属性，动态创建它
                self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False).to(x.device)
            
            # 标准ECA前向传播
            y = self.avg_pool(x)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            return x * y.expand_as(x)
        except Exception as e:
            print(f"ECA前向传播错误: {e}")
            # 出错时返回输入，不做任何操作
            return x

# 将ECA模块注入到ultralytics.nn.modules.conv命名空间中
def patch_ultralytics():
    try:
        import ultralytics.nn.modules.conv as conv_module
        
        # 检查是否已存在ECA模块
        if not hasattr(conv_module, 'ECA'):
            # 设置ECA模块
            setattr(conv_module, 'ECA', ECA)
            print("成功添加ECA模块到ultralytics")
            
            # 添加额外检查
            if hasattr(conv_module, 'ECA'):
                test_instance = conv_module.ECA(64)
                if hasattr(test_instance, 'conv'):
                    print("ECA模块conv属性验证成功")
                else:
                    print("警告: ECA模块没有conv属性")
            
            return True
        else:
            print("ECA模块已存在")
            # 检查现有模块是否有conv属性
            test_instance = conv_module.ECA(64)
            if hasattr(test_instance, 'conv'):
                print("现有ECA模块conv属性验证成功")
            else:
                print("警告: 现有ECA模块没有conv属性，尝试替换...")
                # 强制替换现有模块
                setattr(conv_module, 'ECA', ECA)
            return True
    except Exception as e:
        print(f"无法添加ECA模块: {e}")
        return False 