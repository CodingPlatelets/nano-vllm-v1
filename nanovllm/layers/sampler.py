"""
采样器模块 (Sampler Module)

实现基于温度缩放的随机采样，使用 Gumbel-max 技巧高效采样。
仅在 rank 0 进程中执行（张量并行时其他 rank 不需要采样）。

调用链:
  ModelRunner.run() -> self.sampler(logits, temperatures)

使用的库函数:
  - torch.Tensor.float(): 转为 float32
  - torch.Tensor.div_(x): 原地除法（温度缩放）
  - torch.softmax: 计算 softmax 概率分布
  - torch.empty_like: 创建同形状空 tensor
  - torch.Tensor.exponential_(1): 原地填充标准指数分布随机数
  - torch.Tensor.clamp_min_(x): 原地截断最小值
  - torch.Tensor.argmax(dim=-1): 取最大值索引
  - torch.compile: 编译优化
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """
    Token 采样器，根据 logits 和温度参数采样下一个 token。

    使用 Gumbel-max 技巧代替显式的多项式采样 (torch.multinomial)，
    性能更好且可以被 torch.compile 编译优化。

    Gumbel-max 技巧:
      等价于从 Categorical(softmax(logits / temperature)) 分布采样。
      实现方式: argmax(probs / exponential_noise)
      其中 exponential_noise ~ Exp(1)
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        根据 logits 和温度采样 token。

        调用链:
          ModelRunner.run() -> self.sampler(logits, temperatures) -> token_ids

        Tensor 处理:
          输入:
            logits:       shape [num_seqs, vocab_size], 模型输出的原始 logits
            temperatures: shape [num_seqs], 每个序列的采样温度
          步骤:
            1. logits.float(): 转为 float32 精度
            2. div_(temperatures.unsqueeze(1)): 温度缩放, shape [num_seqs, vocab_size]
               temperature 越大, 分布越平坦(更随机); 越小, 越接近 greedy
            3. softmax(logits, dim=-1): 转为概率分布, shape [num_seqs, vocab_size]
            4. Gumbel-max 采样:
               - exponential_(1): 生成标准指数分布噪声
               - clamp_min_(1e-10): 避免除零
               - probs / noise: 等价于加 Gumbel 噪声
               - argmax(dim=-1): 取概率最高的 token
          输出:
            sample_tokens: shape [num_seqs], 采样得到的 token id

        使用的库函数:
          - torch.Tensor.float(): 转为 float32
          - torch.Tensor.div_(temperatures.unsqueeze(dim=1)): 温度缩放
            unsqueeze(1) 将 [num_seqs] 扩展为 [num_seqs, 1] 以广播
          - torch.softmax(logits, dim=-1): 沿词表维度计算 softmax
          - torch.empty_like(probs).exponential_(1): 生成与 probs 同形状的指数分布噪声
          - torch.Tensor.clamp_min_(1e-10): 截断最小值防止除零
          - torch.Tensor.argmax(dim=-1): 取最大值索引得到采样 token
        """
        # 温度缩放: logits / temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # softmax 转为概率分布
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max 技巧: argmax(probs / Exp(1)) 等价于从 Categorical(probs) 采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
