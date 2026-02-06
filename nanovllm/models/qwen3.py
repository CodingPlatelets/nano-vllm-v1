"""
Qwen3 模型实现 (Qwen3 Model Implementation)

完整实现 Qwen3 因果语言模型（Causal Language Model），包含:
  - Qwen3Attention:   多头注意力（支持 GQA 和张量并行）
  - Qwen3MLP:         门控 MLP（SwiGLU 激活）
  - Qwen3DecoderLayer: 单个 Transformer 解码器层（Attention + MLP + RMSNorm + Residual）
  - Qwen3Model:       完整的 Transformer 模型主体（Embedding + N 层 DecoderLayer + Final Norm）
  - Qwen3ForCausalLM: 带语言模型头的完整模型（Model + LM Head）

模型架构:
  input_ids -> Embedding -> [DecoderLayer_0, ..., DecoderLayer_N] -> Final RMSNorm -> LM Head -> logits

调用链:
  ModelRunner.__init__():
    -> Qwen3ForCausalLM(hf_config) -> Qwen3Model(config) -> N * Qwen3DecoderLayer(config)
  ModelRunner.run_model():
    -> model(input_ids, positions) -> Qwen3ForCausalLM.forward() -> Qwen3Model.forward()
    -> model.compute_logits(hidden_states) -> Qwen3ForCausalLM.compute_logits()

使用的库函数:
  - torch.nn.Module: 所有模型组件的基类
  - torch.nn.ModuleList: 存储 DecoderLayer 列表
  - torch.distributed.get_world_size(): 获取张量并行大小
  - transformers.Qwen3Config: HuggingFace 的 Qwen3 配置类
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 多头注意力层，支持 Grouped Query Attention (GQA) 和张量并行。

    GQA: query heads 可以多于 KV heads，多个 query heads 共享同一组 KV heads。
    例如 Qwen3-14B: 40 个 query heads, 8 个 KV heads (每 5 个 query 共享 1 个 KV)

    子模块:
      - qkv_proj:    QKVParallelLinear, 将 hidden_states 投影为 Q, K, V (合并为一次矩阵乘法)
      - o_proj:      RowParallelLinear, 将注意力输出投影回 hidden_size (需要 all_reduce)
      - rotary_emb:  RotaryEmbedding, 旋转位置编码
      - attn:        Attention, Paged Attention 计算核心
      - q_norm/k_norm: RMSNorm (仅当 qkv_bias=False 时存在，即 Qwen3 默认配置)

    参数:
      hidden_size:  隐藏层维度
      num_heads:    总 query head 数量
      num_kv_heads: 总 KV head 数量 (GQA)
      max_position: 最大位置编码长度
      head_dim:     每个 head 的维度 (默认 hidden_size / num_heads)
      rms_norm_eps: RMSNorm 的 epsilon
      qkv_bias:     QKV 投影是否使用 bias (Qwen3 默认 False)
      rope_theta:   RoPE 频率基数
      rope_scaling: RoPE 缩放配置
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        # 总 head 数量和每个 rank 分到的 head 数量
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 每个 rank 的 Q/KV 输出大小
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放因子: 1 / sqrt(head_dim)
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # QKV 合并投影: hidden_size -> (num_heads + 2*num_kv_heads) * head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影: num_heads * head_dim -> hidden_size (行并行, 需要 all_reduce)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 旋转位置编码 (所有层共享同一实例)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # Paged Attention 计算核心
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3 特有: 当不使用 QKV bias 时，对 Q 和 K 做 RMSNorm (QK-Norm)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Qwen3 注意力前向传播。

        调用链:
          Qwen3DecoderLayer.forward()
            -> self.self_attn(positions, hidden_states)
            -> Qwen3Attention.forward()
               -> qkv_proj() -> split -> view -> q_norm/k_norm -> rotary_emb -> attn -> o_proj

        Tensor 处理:
          输入:
            positions:     shape [num_tokens], 每个 token 的位置索引
            hidden_states: shape [num_tokens, hidden_size]

          1. qkv_proj(hidden_states):
             shape [num_tokens, hidden_size] -> [num_tokens, q_size + 2 * kv_size]
             使用 QKVParallelLinear (F.linear)

          2. split([q_size, kv_size, kv_size], dim=-1):
             -> q: [num_tokens, q_size], k: [num_tokens, kv_size], v: [num_tokens, kv_size]
             使用 torch.Tensor.split

          3. view 重塑为多头格式:
             q: [num_tokens, q_size] -> [num_tokens, num_heads, head_dim]
             k: [num_tokens, kv_size] -> [num_tokens, num_kv_heads, head_dim]
             v: [num_tokens, kv_size] -> [num_tokens, num_kv_heads, head_dim]

          4. q_norm / k_norm (如果不使用 qkv_bias):
             对 q, k 的 head_dim 维度做 RMSNorm (QK-Norm, Qwen3 特有)

          5. rotary_emb(positions, q, k):
             对 q, k 应用 RoPE 旋转位置编码
             输入输出 shape 不变

          6. attn(q, k, v):
             Paged Attention (Flash Attention + KV Cache)
             -> o: [num_tokens, num_heads, head_dim]

          7. o.flatten(1, -1):
             [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads * head_dim]

          8. o_proj(o):
             [num_tokens, num_heads * head_dim] -> [num_tokens, hidden_size]
             使用 RowParallelLinear (F.linear + all_reduce)

          输出: shape [num_tokens, hidden_size]
        """
        # 1. QKV 投影: [num_tokens, hidden_size] -> [num_tokens, q_size + 2*kv_size]
        qkv = self.qkv_proj(hidden_states)
        # 2. 拆分 Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 3. 重塑为多头格式: [num_tokens, size] -> [num_tokens, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 4. Qwen3 QK-Norm: 对 Q 和 K 做 RMSNorm (仅当不使用 qkv_bias 时)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 5. 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 6. Paged Attention: [num_tokens, num_heads, head_dim]
        o = self.attn(q, k, v)
        # 7+8. 展平 + 输出投影: [num_tokens, num_heads * head_dim] -> [num_tokens, hidden_size]
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 门控 MLP 层，使用 SwiGLU 激活函数。

    结构: gate_up_proj -> SiluAndMul -> down_proj
    数学公式: MLP(x) = down_proj(silu(gate_proj(x)) * up_proj(x))

    gate_proj 和 up_proj 合并为 gate_up_proj 以减少矩阵乘法次数:
      gate_up_proj: hidden_size -> 2 * intermediate_size (列并行)
      down_proj:    intermediate_size -> hidden_size (行并行, 需要 all_reduce)

    参数:
      hidden_size:      隐藏层维度
      intermediate_size: MLP 中间层维度 (通常 = hidden_size * 8/3 取整到某个倍数)
      hidden_act:       激活函数类型 (必须为 "silu")
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # gate_proj + up_proj 合并: hidden_size -> 2 * intermediate_size / tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # gate 和 up 各占 intermediate_size
            bias=False,
        )
        # down_proj: intermediate_size / tp_size -> hidden_size (行并行)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """
        Qwen3 MLP 前向传播。

        调用链:
          Qwen3DecoderLayer.forward()
            -> self.mlp(hidden_states)
            -> Qwen3MLP.forward()

        Tensor 处理:
          输入: x, shape [num_tokens, hidden_size]

          1. gate_up_proj(x):
             [num_tokens, hidden_size] -> [num_tokens, 2 * intermediate_size / tp_size]
             使用 MergedColumnParallelLinear (F.linear)

          2. act_fn(gate_up):
             SwiGLU: silu(gate) * up
             [num_tokens, 2 * intermediate_size / tp_size] -> [num_tokens, intermediate_size / tp_size]
             使用 SiluAndMul (F.silu + 逐元素乘)

          3. down_proj(x):
             [num_tokens, intermediate_size / tp_size] -> [num_tokens, hidden_size]
             使用 RowParallelLinear (F.linear + all_reduce)

          输出: shape [num_tokens, hidden_size]
        """
        # 1. 合并的 gate+up 投影
        gate_up = self.gate_up_proj(x)  # [num_tokens, 2 * intermediate_size / tp]
        # 2. SwiGLU 激活: silu(gate) * up
        x = self.act_fn(gate_up)  # [num_tokens, intermediate_size / tp]
        # 3. 下投影 + all_reduce
        x = self.down_proj(x)  # [num_tokens, hidden_size]
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 单个 Transformer 解码器层。

    结构 (Pre-Norm + Residual):
      x -> input_layernorm -> self_attn -> post_attention_layernorm -> mlp -> (output, residual)

    残差连接策略:
      使用 "fused add + RMSNorm" 优化内存访问:
      - 第一层 (residual=None): norm(x), residual = x
      - 后续层 (residual != None): norm(x + residual), residual = x + old_residual
      这样避免了单独的残差相加操作

    子模块:
      - self_attn:                 Qwen3Attention (多头注意力)
      - mlp:                       Qwen3MLP (门控 MLP)
      - input_layernorm:           RMSNorm (注意力前的归一化)
      - post_attention_layernorm:  RMSNorm (MLP 前的归一化)
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单个解码器层的前向传播。

        调用链:
          Qwen3Model.forward()
            -> for layer in self.layers: layer(positions, hidden_states, residual)
            -> Qwen3DecoderLayer.forward()

        Tensor 处理:
          输入:
            positions:     shape [num_tokens]
            hidden_states: shape [num_tokens, hidden_size]
            residual:      shape [num_tokens, hidden_size] 或 None (第一层)

          === 第一层 (residual is None) ===
            1. hidden_states, residual = layernorm(hidden_states), hidden_states
               -> hidden_states = RMSNorm(hidden_states): 归一化
               -> residual = 原始 hidden_states (嵌入输出)

          === 后续层 (residual is not None) ===
            1. hidden_states, residual = layernorm(hidden_states, residual)
               -> residual = hidden_states + old_residual (融合加法)
               -> hidden_states = RMSNorm(residual)

          2. hidden_states = self_attn(positions, hidden_states)
             -> 注意力计算, shape 不变 [num_tokens, hidden_size]

          3. hidden_states, residual = post_layernorm(hidden_states, residual)
             -> residual = hidden_states + residual (融合加法)
             -> hidden_states = RMSNorm(residual)

          4. hidden_states = mlp(hidden_states)
             -> MLP 计算, shape 不变 [num_tokens, hidden_size]

          输出: (hidden_states, residual), 各 shape [num_tokens, hidden_size]
        """
        # 1. 注意力前归一化 (Pre-Norm)
        if residual is None:
            # 第一层: 直接归一化，原始输入作为 residual
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层: fused add + RMSNorm (hidden_states + residual -> norm -> new_residual)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 2. 多头注意力
        hidden_states = self.self_attn(positions, hidden_states)
        # 3. MLP 前归一化 (fused add + RMSNorm)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 4. 门控 MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 Transformer 模型主体（不包含 LM Head）。

    结构: Embedding -> N * DecoderLayer -> Final RMSNorm

    子模块:
      - embed_tokens: VocabParallelEmbedding, 词表并行嵌入层
      - layers:       nn.ModuleList[Qwen3DecoderLayer], N 个解码器层
      - norm:         RMSNorm, 最终层归一化
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Qwen3 模型主体前向传播。

        调用链:
          Qwen3ForCausalLM.forward()
            -> self.model(input_ids, positions)
            -> Qwen3Model.forward()

        Tensor 处理:
          输入:
            input_ids: shape [num_tokens], token id 序列（batch 内所有序列拼接）
            positions: shape [num_tokens], 对应的位置索引

          1. embed_tokens(input_ids):
             [num_tokens] -> [num_tokens, hidden_size]
             使用 VocabParallelEmbedding (F.embedding + all_reduce)

          2. for layer in layers: layer(positions, hidden_states, residual)
             反复执行 DecoderLayer，hidden_states 和 residual 交替更新
             每层 shape 保持 [num_tokens, hidden_size]

          3. norm(hidden_states, residual):
             最终的 fused add + RMSNorm
             -> hidden_states = RMSNorm(hidden_states + residual)

          输出: hidden_states, shape [num_tokens, hidden_size]
        """
        # 1. 词嵌入查表
        hidden_states = self.embed_tokens(input_ids)  # [num_tokens, hidden_size]
        residual = None
        # 2. 逐层前向传播
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 3. 最终层归一化
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型（Causal LM）：Model + LM Head。

    顶层模型类，被 ModelRunner 直接实例化和调用。

    packed_modules_mapping:
      定义权重加载时的合并映射关系，供 load_model() 使用:
        "q_proj"    -> ("qkv_proj", "q"):  q_proj 权重加载到 qkv_proj 的 Q 部分
        "k_proj"    -> ("qkv_proj", "k"):  k_proj 权重加载到 qkv_proj 的 K 部分
        "v_proj"    -> ("qkv_proj", "v"):  v_proj 权重加载到 qkv_proj 的 V 部分
        "gate_proj" -> ("gate_up_proj", 0): gate_proj 权重加载到 gate_up_proj 的第 0 分片
        "up_proj"   -> ("gate_up_proj", 1): up_proj 权重加载到 gate_up_proj 的第 1 分片

    子模块:
      - model:   Qwen3Model, Transformer 主体
      - lm_head: ParallelLMHead, 语言模型输出头

    权重共享 (tie_word_embeddings):
      如果配置启用，lm_head.weight 与 embed_tokens.weight 共享同一份数据
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 权重绑定: lm_head 和 embed_tokens 共享同一份权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        模型前向传播（不包含 LM Head），返回 hidden_states。

        调用链:
          ModelRunner.run_model()
            -> self.model(input_ids, positions) [直接调用]
            或 CUDA Graph replay 后取 outputs

        Tensor 处理:
          输入:
            input_ids: shape [num_tokens]
            positions: shape [num_tokens]
          输出:
            hidden_states: shape [num_tokens, hidden_size]

        注意: forward 和 compute_logits 分开是为了支持 CUDA Graph —
              CUDA Graph 只捕获 forward（固定 shape），compute_logits 在 graph 外执行。
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算语言模型 logits（在 CUDA Graph 外执行）。

        调用链:
          ModelRunner.run_model()
            -> self.model.compute_logits(hidden_states)
            或 self.model.compute_logits(graph_vars["outputs"][:bs])

        Tensor 处理:
          输入: hidden_states, shape [num_tokens, hidden_size] (或 CUDA Graph 输出)
          -> lm_head(hidden_states):
             内部取每个序列最后一个 token 的 hidden_states
             -> F.linear 投影到词表 -> gather/cat 拼接
          输出:
            - rank 0: shape [num_seqs, vocab_size], 完整 logits
            - 其他 rank: None

        使用的库函数:
          - ParallelLMHead.forward(): 参见 embed_head.py
        """
        return self.lm_head(hidden_states)
