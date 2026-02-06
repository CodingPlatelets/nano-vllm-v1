"""
旋转位置编码模块 (Rotary Position Embedding / RoPE)

实现 RoPE (Rotary Position Embedding)，为 query 和 key 注入位置信息。
RoPE 通过旋转变换将绝对位置编码转化为相对位置编码，使得内积只依赖于相对位置。

调用链:
  Qwen3Attention.__init__() -> get_rope() -> _get_rope_cached() -> RotaryEmbedding()
  Qwen3Attention.forward() -> self.rotary_emb(positions, q, k)
                            -> RotaryEmbedding.forward() -> apply_rotary_emb()

使用的库函数:
  - torch.arange: 生成等差序列
  - torch.einsum("i,j -> ij", ...): 外积，计算位置与频率的乘积
  - torch.Tensor.cos / sin: 三角函数
  - torch.chunk: 沿指定维度拆分 tensor
  - torch.cat: 沿指定维度拼接 tensor
  - torch.compile: 编译优化 forward 方法
  - functools.lru_cache: 缓存 RotaryEmbedding 实例（所有层共享同一个）
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对输入 tensor 应用旋转位置编码。

    数学公式 (对每个位置 p 的每对相邻维度 [2i, 2i+1]):
      y[2i]   = x[2i]   * cos(p * theta_i) - x[2i+1] * sin(p * theta_i)
      y[2i+1] = x[2i+1] * cos(p * theta_i) + x[2i]   * sin(p * theta_i)

    调用链:
      RotaryEmbedding.forward() -> apply_rotary_emb(query, cos, sin)
                                 -> apply_rotary_emb(key, cos, sin)

    Tensor 处理:
      输入: x, shape [num_tokens, num_heads, head_dim]
            cos, shape [num_tokens, 1, head_dim // 2]
            sin, shape [num_tokens, 1, head_dim // 2]
        -> chunk(2, -1): 将 x 拆分为 x1, x2, 各 shape [num_tokens, num_heads, head_dim // 2]
        -> y1 = x1 * cos - x2 * sin
        -> y2 = x2 * cos + x1 * sin
        -> cat((y1, y2), -1): 拼接回 [num_tokens, num_heads, head_dim]
      输出: shape [num_tokens, num_heads, head_dim], 与输入相同

    使用的库函数:
      - torch.chunk(x, 2, dim=-1): 将 head_dim 维度拆分为两半
      - torch.Tensor.float(): 转 float32 计算（避免 bf16/fp16 精度损失）
      - torch.cat((y1, y2), dim=-1): 拼接两半结果
      - torch.Tensor.to(dtype): 转回原始精度
    """
    # 将 head_dim 拆为前半和后半，分别对应旋转矩阵的实部和虚部
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 应用二维旋转变换
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼接回完整的 head_dim
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) 实现。

    预计算所有位置的 cos/sin 值并缓存为 buffer，推理时通过 positions 索引查表。
    所有 Attention 层共享同一个 RotaryEmbedding 实例（通过 lru_cache 实现）。

    数学原理:
      对于位置 p 和频率维度 i:
        theta_i = 1 / (base^(2i/d))  其中 d = head_dim
        cos_cache[p, i] = cos(p * theta_i)
        sin_cache[p, i] = sin(p * theta_i)

    参数:
      head_size: 每个注意力头的维度大小 (head_dim)
      rotary_dim: 旋转编码的维度（必须等于 head_size）
      max_position_embeddings: 支持的最大位置数
      base: 频率基数，控制不同维度的旋转频率, 默认 10000 (Qwen3 使用 1000000)

    缓存:
      cos_sin_cache: shape [max_position, 1, head_dim], 预计算的 [cos, sin] 拼接缓存
                     第二维为 1 是为了广播到多个 head
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 计算频率向量: theta_i = 1 / (base^(2i/d)), shape [head_dim // 2]
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 位置序列: [0, 1, 2, ..., max_position - 1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积: freqs[p, i] = p * theta_i, shape [max_position, head_dim // 2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()  # shape [max_position, head_dim // 2]
        sin = freqs.sin()  # shape [max_position, head_dim // 2]
        # 拼接 cos 和 sin: shape [max_position, head_dim], 然后增加 head 维度
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)  # [max_position, 1, head_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 query 和 key 应用旋转位置编码。

        调用链:
          Qwen3Attention.forward() -> self.rotary_emb(positions, q, k)
          -> RotaryEmbedding.forward() -> apply_rotary_emb(query, cos, sin)
                                        -> apply_rotary_emb(key, cos, sin)

        Tensor 处理:
          输入:
            positions: shape [num_tokens], 每个 token 的位置索引
            query: shape [num_tokens, num_heads, head_dim]
            key:   shape [num_tokens, num_kv_heads, head_dim]
          中间:
            cos_sin = cos_sin_cache[positions]: 根据位置索引查表
              shape [num_tokens, 1, head_dim], 1 会广播到 num_heads
            cos, sin: 各 shape [num_tokens, 1, head_dim // 2]
          输出:
            query: shape [num_tokens, num_heads, head_dim], 旋转后的 query
            key:   shape [num_tokens, num_kv_heads, head_dim], 旋转后的 key

        使用的库函数:
          - 索引操作 self.cos_sin_cache[positions]: 按位置查表
          - torch.Tensor.chunk(2, dim=-1): 拆分 cos 和 sin
          - apply_rotary_emb(): 应用旋转变换
        """
        # 根据 position 索引从预计算缓存中取出对应的 cos/sin 值
        cos_sin = self.cos_sin_cache[positions]  # [num_tokens, 1, head_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)  # 各 [num_tokens, 1, head_dim // 2]
        # 对 query 和 key 分别应用旋转编码
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """
    创建并缓存 RotaryEmbedding 实例。

    使用 lru_cache(1) 确保所有 Attention 层共享同一个 RotaryEmbedding 实例，
    因为它们的 head_size、rotary_dim、max_position、base 参数相同。

    调用链:
      get_rope() -> _get_rope_cached() -> RotaryEmbedding()

    使用的库函数:
      - functools.lru_cache(1): 缓存最近一次调用的结果
    """
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取 RotaryEmbedding 实例的工厂函数。

    调用链:
      Qwen3Attention.__init__() -> get_rope(head_dim, ...) -> _get_rope_cached()

    参数:
      head_size:    每个注意力头的维度大小
      rotary_dim:   旋转编码维度（必须等于 head_size）
      max_position: 最大支持位置数
      base:         频率基数 (Qwen3 使用 1000000)
      rope_scaling: RoPE 缩放配置（当前仅支持 "default" 类型，即不缩放）

    返回:
      RotaryEmbedding 实例
    """
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", "default")
        assert rope_type == "default", f"rope_type '{rope_type}' is not supported yet"
    return _get_rope_cached(head_size, rotary_dim, max_position, base)
