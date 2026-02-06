"""
全局上下文模块 (Global Context Module)

本模块通过全局变量 _CONTEXT 在模型前向传播过程中传递 attention 所需的元数据，
避免在模型各层之间显式传递大量参数。

调用链:
  - ModelRunner.prepare_model_input() 调用 set_context() 设置上下文
  - Attention.forward() / ParallelLMHead.forward() 调用 get_context() 读取上下文
  - ModelRunner.run() 在推理结束后调用 reset_context() 清理上下文
  - ModelRunner.capture_cudagraph() 在捕获 CUDA Graph 时也会调用 set_context()/reset_context()

使用的库函数:
  - dataclasses.dataclass: 用于定义 Context 数据类
  - torch.Tensor: 上下文中存储的 tensor 类型
"""

from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    全局上下文数据类，存储 Flash Attention 和 KV Cache 所需的元数据。

    在模型前向传播过程中，Attention 层和 LM Head 层需要访问这些元数据，
    但这些信息不适合作为模型 forward() 的参数传递（因为 CUDA Graph 捕获时需要固定签名）。
    因此通过全局变量传递。

    字段说明:
      - cu_seqlens_q: shape [batch_size + 1], Flash Attention 变长序列的 query 累计长度前缀和
                      例如 [0, 5, 12] 表示第一个序列 query 长度为 5，第二个为 7
                      在 CUDA Graph decode 阶段为 None（此时使用 flash_attn_with_kvcache）
      - cu_seqlens_k: shape [batch_size + 1], Flash Attention 变长序列的 key 累计长度前缀和
      - max_seqlen_q: 当前 batch 中最长 query 序列长度（Flash Attention 需要）
      - max_seqlen_k: 当前 batch 中最长 key 序列长度（Flash Attention 需要）
      - slot_mapping:  shape [total_new_tokens], 每个新 token 在 KV Cache 物理块中的 slot 索引
                       用于 Triton kernel store_kvcache 将新的 K/V 写入正确的缓存位置
      - context_lens:  shape [batch_size], 每个序列的上下文长度（cached + new tokens）
                       仅在 CUDA Graph decode 阶段使用（flash_attn_with_kvcache 需要）
      - block_tables:  shape [batch_size, max_num_blocks], 每个序列的物理块号表
                       用于 Paged Attention 从 KV Cache 中读取数据
                       为 None 表示 warmup 阶段（无 KV Cache）
      - seq_need_compute_logits: shape [num_seqs_need_logits], 需要计算 logits 的序列索引
                                 用于 ParallelLMHead 仅对完整序列计算 logits（节省计算）
    """
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    seq_need_compute_logits: torch.Tensor | None = None


# 模块级全局变量，存储当前推理步骤的上下文
_CONTEXT = Context()


def get_context():
    """
    获取当前全局上下文。

    调用链:
      - 被 Attention.forward() 调用，获取 slot_mapping / cu_seqlens / block_tables 等
      - 被 ParallelLMHead.forward() 调用，获取 cu_seqlens_q 和 seq_need_compute_logits
      - 被 ModelRunner.run() 调用，获取 seq_need_compute_logits
      - 被 ModelRunner.prepare_sample() 调用，获取 seq_need_compute_logits 过滤温度

    返回:
      Context 实例，包含当前推理步骤的所有元数据
    """
    return _CONTEXT


def set_context(cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, seq_need_compute_logits=None):
    """
    设置全局上下文，在每个推理步骤开始时由 ModelRunner.prepare_model_input() 调用。

    调用链:
      - ModelRunner.prepare_model_input() -> set_context(): 正常推理时设置完整上下文
      - ModelRunner.capture_cudagraph() -> set_context(): CUDA Graph 捕获时设置部分上下文

    参数:
      cu_seqlens_q:  query 累计序列长度, shape [batch_size + 1]
      cu_seqlens_k:  key 累计序列长度, shape [batch_size + 1]
      max_seqlen_q:  最大 query 序列长度 (int)
      max_seqlen_k:  最大 key 序列长度 (int)
      slot_mapping:  KV Cache slot 映射, shape [total_new_tokens]
      context_lens:  各序列上下文长度, shape [batch_size]
      block_tables:  物理块号表, shape [batch_size, max_num_blocks]
      seq_need_compute_logits: 需要计算 logits 的序列索引, shape [num_seqs]
    """
    global _CONTEXT
    _CONTEXT = Context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables, seq_need_compute_logits)


def reset_context():
    """
    重置全局上下文为空的 Context，在每个推理步骤结束后由 ModelRunner.run() 调用。

    调用链:
      - ModelRunner.run() -> reset_context(): 推理完成后清理
      - ModelRunner.capture_cudagraph() -> reset_context(): 每个 batch size 的 graph 捕获完成后清理
    """
    global _CONTEXT
    _CONTEXT = Context()
