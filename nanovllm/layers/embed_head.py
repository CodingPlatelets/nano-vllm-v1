"""
词嵌入与语言模型头模块 (Embedding & LM Head Module)

实现支持张量并行的词嵌入层 (VocabParallelEmbedding) 和语言模型输出头 (ParallelLMHead)。

词表并行策略:
  将词表沿 vocab 维度切分到各 rank，每个 rank 只存储 vocab_size/tp_size 个嵌入向量。
  - Embedding: 各 rank 查表后 all_reduce 聚合
  - LM Head: 各 rank 计算部分 logits 后 gather 到 rank 0 拼接

调用链:
  Qwen3Model.forward():
    -> self.embed_tokens(input_ids)  -> VocabParallelEmbedding.forward()
  Qwen3ForCausalLM.compute_logits():
    -> self.lm_head(hidden_states)   -> ParallelLMHead.forward()

使用的库函数:
  - torch.nn.functional.embedding: 查表获取嵌入向量
  - torch.nn.functional.linear: 线性变换 (LM Head)
  - torch.distributed.all_reduce: 跨 rank 求和
  - torch.distributed.gather: 将各 rank 的 tensor 收集到 rank 0
  - torch.cat: 拼接 tensor
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行嵌入层：将词表均匀切分到各 rank。

    每个 rank 存储词表的一个分区:
      rank 0: token_id [0, vocab_size/tp)
      rank 1: token_id [vocab_size/tp, 2*vocab_size/tp)
      ...

    前向传播:
      1. 对于不属于当前 rank 分区的 token_id，将其 mask 为 0
      2. 用 F.embedding 查表（只在本 rank 的分区中查）
      3. 将不属于本 rank 的嵌入结果置零
      4. all_reduce 求和，得到完整的嵌入结果

    参数:
      num_embeddings: 词表总大小
      embedding_dim:  嵌入向量维度 (hidden_size)

    权重:
      self.weight: shape [num_embeddings / tp_size, embedding_dim]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 每个 rank 的词表分区大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前 rank 负责的词表索引范围 [vocab_start_idx, vocab_end_idx)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        按词表分区加载嵌入权重。

        调用链: load_model() -> weight_loader(param, loaded_weight)

        Tensor 处理:
          loaded_weight: shape [vocab_size, embedding_dim] (完整词表嵌入)
          -> narrow(0, tp_rank * shard_size, shard_size): 截取当前 rank 的分区
          -> copy_ 到 param.data: shape [vocab_size / tp_size, embedding_dim]

        使用的库函数:
          - torch.Tensor.narrow(0, start, length): 沿词表维度截取
          - torch.Tensor.copy_: 复制数据
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        词表并行嵌入查表。

        调用链:
          Qwen3Model.forward() -> self.embed_tokens(input_ids)
          -> VocabParallelEmbedding.forward(input_ids)

        Tensor 处理:
          输入: x (input_ids), shape [num_tokens]
          tp_size == 1 (单卡):
            -> F.embedding(x, weight): shape [num_tokens, embedding_dim]
          tp_size > 1 (多卡):
            1. mask = (x >= start) & (x < end): 标记属于当前 rank 的 token
            2. x = mask * (x - start): 将 id 转为本地偏移（不属于的置 0）
            3. y = F.embedding(x, weight): 查表, shape [num_tokens, embedding_dim]
            4. y = mask.unsqueeze(1) * y: 不属于本 rank 的嵌入置零
            5. all_reduce(y): 跨 rank 求和得到完整嵌入
          输出: shape [num_tokens, embedding_dim]

        使用的库函数:
          - F.embedding(x, weight): 查表, x 中的索引映射到 weight 的对应行
          - dist.all_reduce(y): NCCL 跨 rank 求和
        """
        if self.tp_size > 1:
            # 标记哪些 token_id 属于当前 rank 的词表分区
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将 token_id 转为本地偏移索引（不属于当前 rank 的置 0，查表结果后续会置零）
            x = mask * (x - self.vocab_start_idx)
        # 嵌入查表
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 将不属于当前 rank 的嵌入结果置零（mask 广播到 embedding_dim 维度）
            y = mask.unsqueeze(1) * y
            # 跨 rank 求和，得到完整的嵌入向量
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型输出头：将 hidden_states 投影到词表维度，得到 logits。

    继承自 VocabParallelEmbedding，复用其词表分片逻辑和权重加载方法。
    可以与 embed_tokens 共享权重 (tie_word_embeddings)。

    与嵌入层的区别:
      - 嵌入层: token_id -> embedding (查表)
      - LM Head: hidden_states -> logits (线性变换)

    张量并行策略:
      各 rank 计算 logits 的一个分区（对应自己负责的词表部分），
      然后通过 gather 收集到 rank 0 并拼接为完整的 logits。
      只有 rank 0 拥有完整 logits（用于采样），其他 rank 返回 None。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias  # LM Head 不使用 bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        计算语言模型 logits。

        调用链:
          Qwen3ForCausalLM.compute_logits(hidden_states)
            -> self.lm_head(hidden_states)
            -> ParallelLMHead.forward(hidden_states)

        Tensor 处理:
          输入: x (hidden_states), shape [total_tokens, hidden_size]
          步骤:
            1. 从 context 获取 cu_seqlens_q，计算每个序列最后一个 token 的索引
               last_indices = cu_seqlens_q[1:] - 1, shape [batch_size]
            2. 如果有 seq_need_compute_logits（不是所有序列都需要），进一步过滤
               last_indices = last_indices[seq_need_compute_logits]
            3. x = x[last_indices]: 只取需要的 token, shape [num_seqs, hidden_size]
            4. logits = F.linear(x, weight): 投影到词表, shape [num_seqs, vocab_size / tp_size]
            5. tp_size > 1 时:
               - gather 到 rank 0: 收集所有 rank 的 logits
               - cat(all_logits, -1): 拼接为 [num_seqs, vocab_size]
               - rank 0 返回完整 logits，其他 rank 返回 None
          输出:
            - rank 0: shape [num_seqs, vocab_size], 完整 logits
            - 其他 rank: None

        使用的库函数:
          - get_context(): 获取 cu_seqlens_q 和 seq_need_compute_logits
          - torch.Tensor.__getitem__(indices): 高级索引，取出指定行
          - torch.Tensor.contiguous(): 确保内存连续
          - F.linear(x, weight): 线性变换 x @ weight.T, shape [num_seqs, vocab_per_rank]
          - dist.gather(tensor, gather_list, dst=0): 将各 rank 的 tensor 收集到 rank 0
          - torch.cat(tensors, dim=-1): 沿词表维度拼接
        """
        context = get_context()
        # 计算每个序列最后一个 token 在 flattened 输入中的索引
        # cu_seqlens_q = [0, len_1, len_1+len_2, ...], 所以 cu_seqlens_q[1:]-1 就是每个序列的最后位置
        last_indices = context.cu_seqlens_q[1:] - 1
        # 如果只有部分序列需要计算 logits（如 chunked prefill 时未完成的序列不需要）
        if context.seq_need_compute_logits.numel():
            last_indices = last_indices[context.seq_need_compute_logits]
        # 只取需要计算 logits 的 token 的 hidden_states
        x = x[last_indices].contiguous()  # [num_seqs, hidden_size]
        # 线性投影到词表空间（注意这里 weight 是 [vocab_per_rank, hidden_size]）
        logits = F.linear(x, self.weight)  # [num_seqs, vocab_size / tp_size]
        if self.tp_size > 1:
            # 多卡: 将各 rank 的部分 logits 收集到 rank 0
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # rank 0 拼接为完整 logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
