"""
注意力机制模块 (Attention Module)

实现 Paged Attention，包含:
  1. Triton kernel: 将新的 K/V 写入 Paged KV Cache
  2. Flash Attention: 计算注意力输出（支持两种模式）

支持两种执行路径:
  - Prefill / 非 CUDA Graph Decode: 使用 flash_attn_varlen_func (变长序列)
  - CUDA Graph Decode: 使用 flash_attn_with_kvcache (固定 batch_size)

调用链:
  Qwen3Attention.forward()
    -> self.attn(q, k, v)
    -> Attention.forward()
       -> store_kvcache()          # Triton kernel: 写入 KV Cache
       -> flash_attn_varlen_func() # 或 flash_attn_with_kvcache()

使用的库函数:
  - triton.jit / triton.language: Triton JIT 编译 GPU kernel
  - flash_attn.flash_attn_varlen_func: 变长序列 Flash Attention（prefill + decode）
  - flash_attn.flash_attn_with_kvcache: 带 KV Cache 的 Flash Attention（CUDA Graph decode）
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton GPU kernel: 将新的 key/value 写入 Paged KV Cache 的指定 slot。

    调用链:
      store_kvcache() -> store_kvcache_kernel[(N,)](...)
      每个 Triton program 处理一个 token 的 K 和 V

    工作原理:
      1. 每个 program (idx) 处理一个新 token
      2. 从 slot_mapping 读取该 token 对应的物理 slot 号
      3. 从 key/value tensor 读取该 token 的 K/V 数据
      4. 写入 KV Cache 的对应 slot 位置

    参数 (Triton kernel 参数):
      key_ptr:          key tensor 的基地址
      key_stride:       key tensor 第 0 维的 stride (= num_heads * head_dim)
      value_ptr:        value tensor 的基地址
      value_stride:     value tensor 第 0 维的 stride
      k_cache_ptr:      k_cache 的基地址
      v_cache_ptr:      v_cache 的基地址
      slot_mapping_ptr: slot_mapping 的基地址
      D:                每个 token 的 K/V 数据大小 (= num_kv_heads * head_dim), 编译时常量

    使用的 Triton 库函数:
      - tl.program_id(0): 获取当前 program 的 ID (对应第几个 token)
      - tl.load(ptr + offsets): 从 GPU 显存读取数据
      - tl.store(ptr + offsets, data): 向 GPU 显存写入数据
      - tl.arange(0, D): 生成 [0, 1, ..., D-1] 的索引序列
    """
    # 当前 program 处理第 idx 个新 token
    idx = tl.program_id(0)
    # 读取该 token 对应的物理 slot 号
    slot = tl.load(slot_mapping_ptr + idx)
    # slot == -1 表示该位置不需要写入（如 CUDA Graph 的 padding 位置）
    if slot == -1: return
    # 计算 key/value 在源 tensor 中的偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 读取该 token 的 K 和 V (所有 head 的数据是连续的)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 计算在 KV Cache 中的目标偏移
    cache_offsets = slot * D + tl.arange(0, D)
    # 写入 KV Cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    """
    将新 token 的 key 和 value 写入 Paged KV Cache（调用 Triton kernel）。

    调用链:
      Attention.forward() -> store_kvcache(k, v, k_cache, v_cache, slot_mapping)
      -> store_kvcache_kernel[(N,)](...): 启动 N 个 Triton program

    Tensor 处理:
      输入:
        key:          shape [N, num_kv_heads, head_dim], 新 token 的 key
        value:        shape [N, num_kv_heads, head_dim], 新 token 的 value
        k_cache:      shape [num_blocks * block_size, num_kv_heads, head_dim], 展平的 key 缓存
                      (实际存储为 [num_blocks, block_size, num_kv_heads * head_dim] 视图)
        v_cache:      shape 同 k_cache, 展平的 value 缓存
        slot_mapping: shape [N], 每个新 token 在缓存中的 slot 索引

      Triton kernel 中将 key/value 视为 [N, D] (D = num_kv_heads * head_dim) 的二维布局，
      按 slot_mapping 写入缓存的对应行。

    使用的库函数:
      - store_kvcache_kernel[(N,)](...): 启动 Triton kernel，N 个 program 并行
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 每个 token 的 K/V 数据总大小
    # 内存布局断言：确保最内层维度连续
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 N 个 Triton program，每个处理一个 token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                               k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    Paged Attention 实现，核心注意力计算层。

    每个 Transformer 层中有一个 Attention 实例。
    在 ModelRunner.allocate_kv_cache() 中，k_cache 和 v_cache 会被替换为
    分配好的 KV Cache tensor 的对应层切片。

    支持三种执行模式:
      1. Warmup（无 KV Cache）: 直接用 flash_attn_varlen_func 计算
      2. Prefill / 非 CUDA Graph Decode: 先写入 KV Cache，再用 flash_attn_varlen_func
         从 KV Cache 读取所有 K/V 计算注意力
      3. CUDA Graph Decode: 先写入 KV Cache，再用 flash_attn_with_kvcache 计算

    属性:
      num_heads:   当前 rank 的 query head 数量
      head_dim:    每个 head 的维度
      scale:       attention 缩放因子 (1/sqrt(head_dim))
      num_kv_heads: 当前 rank 的 KV head 数量 (GQA 时 < num_heads)
      k_cache:     key 缓存 tensor, 初始为空, 后由 ModelRunner 设置
      v_cache:     value 缓存 tensor, 初始为空, 后由 ModelRunner 设置
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始化为空 tensor，ModelRunner.allocate_kv_cache() 会替换为实际的 KV Cache 切片
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        注意力前向传播：写入 KV Cache + 计算注意力输出。

        调用链:
          Qwen3Attention.forward()
            -> self.attn(q, k, v)
            -> Attention.forward(q, k, v)
               -> store_kvcache()              # 写入 KV Cache
               -> flash_attn_varlen_func()     # 或 flash_attn_with_kvcache()

        Tensor 处理:
          输入:
            q: shape [num_tokens, num_heads, head_dim]
            k: shape [num_tokens, num_kv_heads, head_dim]
            v: shape [num_tokens, num_kv_heads, head_dim]

          === 路径 1: CUDA Graph Decode (cu_seqlens_q is None) ===
            前提: 每个序列只有 1 个新 token (decode 阶段)
            1. store_kvcache(k, v, ...): 将新 K/V 写入 Paged KV Cache
            2. q.view(batch_size, 1, num_heads, head_dim): 重塑为 [B, 1, H, D]
            3. flash_attn_with_kvcache(q, k_cache, v_cache, ...):
               通过 block_table 和 cache_seqlens 从 Paged KV Cache 读取历史 K/V
               输出: shape [batch_size, 1, num_heads, head_dim]
            4. view(-1, num_heads, head_dim): 恢复为 [num_tokens, num_heads, head_dim]

          === 路径 2: Prefill / 非 CUDA Graph Decode (cu_seqlens_q is not None) ===
            2a. block_tables is not None (有 KV Cache):
              1. store_kvcache(k, v, ...): 将新 K/V 写入 Paged KV Cache
              2. k, v = k_cache, v_cache: 用完整的 KV Cache 替换 k, v
              3. flash_attn_varlen_func(q, k_cache, v_cache, ...):
                 通过 block_table 从 Paged KV Cache 读取 K/V
                 支持变长序列（通过 cu_seqlens_q/k 指定边界）
            2b. block_tables is None (warmup, 无 KV Cache):
              1. flash_attn_varlen_func(q, k, v, ...): 直接用输入的 k, v 计算

          输出: o, shape [num_tokens, num_heads, head_dim]

        使用的库函数:
          - store_kvcache(): 自定义 Triton kernel 写入 KV Cache
          - flash_attn_with_kvcache(): Flash Attention 2 的 KV Cache decode 版本
            参数: q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal
          - flash_attn_varlen_func(): Flash Attention 2 的变长序列版本
            参数: q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                   softmax_scale, causal, block_table
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 步骤 1: 将新 token 的 K/V 写入 Paged KV Cache
        # （仅在 KV Cache 已分配时执行，warmup 阶段跳过）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.cu_seqlens_q is None:
            # === CUDA Graph Decode 路径 ===
            # cu_seqlens 不可用（CUDA Graph 不支持动态 shape），使用 flash_attn_with_kvcache
            batch_size = context.block_tables.size(0)
            # q 重塑为 [batch_size, seqlen=1, num_heads, head_dim]
            q = q.view(batch_size, 1, self.num_heads, self.head_dim)
            # 从 Paged KV Cache 中通过 block_table 读取历史 K/V 并计算注意力
            o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                        cache_seqlens=context.context_lens,
                                        block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
            # 恢复为 [num_tokens, num_heads, head_dim]
            o = o.view(-1, self.num_heads, self.head_dim)
        else:
            # === Prefill / 非 CUDA Graph Decode 路径 ===
            if context.block_tables is not None:
                # 有 Paged KV Cache: 用 KV Cache 替换 k, v
                # flash_attn_varlen_func 将通过 block_table 从 KV Cache 读取
                k, v = k_cache, v_cache
            # 变长序列 Flash Attention（支持 batch 内不同长度的序列）
            o = flash_attn_varlen_func(q, k, v,
                                        cu_seqlens_q=context.cu_seqlens_q,
                                        cu_seqlens_k=context.cu_seqlens_k,
                                        max_seqlen_q=context.max_seqlen_q,
                                        max_seqlen_k=context.max_seqlen_k,
                                        softmax_scale=self.scale, causal=True,
                                        block_table=context.block_tables)
        return o
