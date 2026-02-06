"""
RMS Layer Normalization 模块 (RMSNorm Module)

实现 Root Mean Square Layer Normalization，用于 Transformer 各层的归一化。
支持两种模式：
  1. 纯 RMSNorm: 仅对输入做归一化
  2. Fused Add + RMSNorm: 先将 hidden_states 与 residual 相加，再做归一化
     这种融合操作避免了单独的残差相加步骤，减少内存访问

调用链:
  - Qwen3DecoderLayer.forward():
    - input_layernorm(hidden_states) 或 input_layernorm(hidden_states, residual)
    - post_attention_layernorm(hidden_states, residual)
  - Qwen3Model.forward():
    - self.norm(hidden_states, residual)  # 最终层归一化

使用的库函数:
  - torch.Tensor.pow(2): 逐元素平方
  - torch.Tensor.mean(dim=-1, keepdim=True): 沿最后一维求均值
  - torch.rsqrt: 求平方根的倒数, rsqrt(x) = 1/sqrt(x)
  - torch.Tensor.mul_: 原地逐元素乘法
  - torch.Tensor.float / to(dtype): 类型转换（先转 float32 计算，再转回原始 dtype）
  - torch.compile: 编译优化
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    与标准 LayerNorm 不同，RMSNorm 不减去均值（不做中心化），仅通过 RMS 值归一化，
    计算量更小且效果相近。

    数学公式:
      RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    参数:
      hidden_size: 归一化维度大小（即最后一维的大小）
      eps: 防止除零的小常数，默认 1e-6

    权重:
      self.weight: shape [hidden_size], 可学习的缩放参数，初始化为全 1
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        纯 RMSNorm 前向传播（无残差相加）。

        调用链:
          Qwen3DecoderLayer.forward() -> input_layernorm(hidden_states)  # 第一层时 residual=None
          实际路由: RMSNorm.forward(x, residual=None) -> rms_forward(x)

        Tensor 处理:
          输入: x, shape [..., hidden_size] (例如 [num_tokens, hidden_size])
            -> float(): 转为 float32 保证数值精度
            -> pow(2).mean(-1, keepdim=True): 计算 RMS^2, shape [..., 1]
            -> mul_(rsqrt(var + eps)): 归一化, shape [..., hidden_size]
            -> to(orig_dtype).mul_(weight): 转回原始 dtype 并乘以缩放参数
          输出: shape [..., hidden_size], 归一化后的 tensor

        使用的库函数:
          - torch.Tensor.float(): 转为 float32
          - torch.Tensor.pow(2): 逐元素平方
          - torch.Tensor.mean(dim=-1, keepdim=True): 沿最后一维求均值
          - torch.rsqrt(x): 计算 1/sqrt(x)
          - torch.Tensor.mul_(x): 原地乘法
          - torch.Tensor.to(dtype): 类型转换
        """
        orig_dtype = x.dtype
        x = x.float()
        # 计算 RMS: sqrt(mean(x^2))，这里用 var 表示 mean(x^2)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # x = x / sqrt(var + eps)
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原始精度并乘以可学习缩放参数
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合残差相加 + RMSNorm 前向传播。

        先计算 x = x + residual，再对结果做 RMSNorm，同时返回相加后的值作为新的 residual。
        这种融合避免了单独存储中间结果，减少了一次内存读写。

        调用链:
          Qwen3DecoderLayer.forward() -> input_layernorm(hidden_states, residual)
                                       -> post_attention_layernorm(hidden_states, residual)
          Qwen3Model.forward() -> self.norm(hidden_states, residual)
          实际路由: RMSNorm.forward(x, residual) -> add_rms_forward(x, residual)

        Tensor 处理:
          输入: x, shape [num_tokens, hidden_size]
                residual, shape [num_tokens, hidden_size]
            -> float().add_(residual.float()): x = x + residual (float32 精度)
            -> residual = x.to(orig_dtype): 新 residual = x + old_residual (原始精度)
            -> pow(2).mean(-1, keepdim=True): 计算 RMS^2
            -> mul_(rsqrt(var + eps)): 归一化
            -> to(orig_dtype).mul_(weight): 缩放
          输出: (normalized_x, new_residual)
            - normalized_x: shape [num_tokens, hidden_size], 归一化后的值
            - new_residual: shape [num_tokens, hidden_size], x + old_residual 的值

        使用的库函数:
          - torch.Tensor.float(): 转为 float32
          - torch.Tensor.add_(other): 原地加法
          - torch.rsqrt(x): 计算 1/sqrt(x)
          - torch.Tensor.mul_(x): 原地乘法
          - torch.Tensor.to(dtype): 类型转换
        """
        orig_dtype = x.dtype
        # 融合残差相加: x = x + residual (float32 精度计算)
        x = x.float().add_(residual.float())
        # 保存相加结果作为新的 residual（转回原始精度）
        residual = x.to(orig_dtype)
        # RMSNorm 归一化
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        RMSNorm 统一入口，根据是否有 residual 选择不同的计算路径。

        调用链:
          - 被 Qwen3DecoderLayer.forward() 和 Qwen3Model.forward() 调用

        参数:
          x:        输入 tensor, shape [num_tokens, hidden_size]
          residual: 残差 tensor, shape [num_tokens, hidden_size] 或 None

        返回:
          - 若 residual=None: 返回归一化后的 x, shape [num_tokens, hidden_size]
          - 若 residual 不为 None: 返回 (normalized_x, new_residual) 元组
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
