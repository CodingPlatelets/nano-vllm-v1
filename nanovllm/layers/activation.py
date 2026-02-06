"""
激活函数模块 (Activation Module)

实现 SwiGLU 激活函数（SiLU + Gate 机制），用于 Qwen3 MLP 中的门控激活。

调用链:
  Qwen3MLP.forward() -> SiluAndMul.forward()

使用的库函数:
  - torch.Tensor.chunk: 将 tensor 沿最后一维拆分为两半
  - torch.nn.functional.silu: SiLU (Swish) 激活函数, silu(x) = x * sigmoid(x)
  - torch.compile: 将 forward 编译为优化的 Triton/CUDA kernel
"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SwiGLU 激活函数实现：将输入沿最后一维拆分为两半，前半部分通过 SiLU 激活，
    然后与后半部分逐元素相乘。

    在 Transformer MLP 中的角色:
      gate_up_proj 输出 [gate, up] 两部分 -> SiluAndMul 计算 silu(gate) * up

    数学公式:
      SwiGLU(x) = SiLU(x_gate) * x_up
      其中 x_gate = x[..., :d], x_up = x[..., d:], d = x.shape[-1] // 2
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU 前向传播。

        调用链:
          Qwen3MLP.forward() -> self.act_fn(gate_up) -> SiluAndMul.forward()

        Tensor 处理:
          输入: x, shape [num_tokens, 2 * intermediate_size / tp_size]
            -> chunk(2, -1) 拆分为 x (gate) 和 y (up), 各 shape [num_tokens, intermediate_size / tp_size]
            -> F.silu(x) * y, 输出 shape [num_tokens, intermediate_size / tp_size]

        使用的库函数:
          - torch.Tensor.chunk(2, -1): 沿最后一维将 tensor 均匀拆分为 2 份
          - F.silu(x): 计算 SiLU 激活, silu(x) = x * sigmoid(x)
          - 逐元素乘法 (*): gate 值与 up 值相乘
        """
        # x: [num_tokens, 2 * intermediate_size / tp_size]
        # 拆分为 gate 和 up 两部分
        x, y = x.chunk(2, -1)
        # silu(gate) * up -> [num_tokens, intermediate_size / tp_size]
        return F.silu(x) * y
