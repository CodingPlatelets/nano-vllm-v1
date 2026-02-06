"""
全局配置模块 (Global Configuration Module)

定义 nano-vllm 引擎的所有配置参数。
在 LLMEngine.__init__() 中创建，贯穿整个引擎的生命周期。

调用链:
  LLMEngine.__init__()
    -> Config(model, **kwargs)
    -> Config.__post_init__()
       -> AutoConfig.from_pretrained(model)  # 加载 HuggingFace 模型配置

使用的库函数:
  - dataclasses.dataclass: 定义配置数据类
  - transformers.AutoConfig.from_pretrained: 从模型路径/ID 加载 HF 配置
"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    nano-vllm 引擎全局配置。

    字段说明:
      model:                   模型路径或 HuggingFace 模型 ID (如 "Qwen/Qwen3-14B")
      max_num_batched_tokens:  每轮推理的最大 token 预算 (默认 16384)
                               控制 Scheduler 每步最多处理多少 token
      max_num_seqs:            最大并发序列数 (默认 512)
                               限制 running 队列中的序列数量
      max_model_len:           最大模型上下文长度 (默认 40960)
                               会与 HF 配置的 max_position_embeddings 取较小值
      gpu_memory_utilization:  GPU 显存利用率 (默认 0.9)
                               ModelRunner 据此计算 KV Cache 可用显存
      tensor_parallel_size:    张量并行大小 (默认 1, 最大 8)
                               决定使用多少个 GPU
      enforce_eager:           是否禁用 CUDA Graph (默认 False)
                               True 时不捕获 CUDA Graph，适合调试
      hf_config:               HuggingFace 模型配置 (AutoConfig 实例)
                               在 __post_init__ 中自动加载
      eos:                     EOS token id (默认 -1)
                               在 LLMEngine.__init__ 中由 tokenizer 设置
      kvcache_block_size:      KV Cache 块大小 (默认 256 tokens/block)
                               必须是 256 的倍数
      num_kvcache_blocks:      KV Cache 物理块总数 (默认 -1, 由 ModelRunner 计算)
                               在 allocate_kv_cache() 中根据剩余显存自动计算
      chunked_prefill:         是否启用 Chunked Prefill (默认 False)
                               启用后长 prefill 会被分块处理，与 decode 交替执行
    """
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 40960
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    chunked_prefill: bool = False

    def __post_init__(self):
        """
        配置后处理：验证参数并加载 HuggingFace 模型配置。

        调用链:
          Config(model, **kwargs) -> __post_init__()

        步骤:
          1. 验证 kvcache_block_size 是 256 的倍数
          2. 验证 tensor_parallel_size 在 [1, 8] 范围内
          3. 从模型路径/ID 加载 HuggingFace 配置
          4. 将 max_model_len 限制为不超过模型的 max_position_embeddings

        使用的库函数:
          - AutoConfig.from_pretrained(self.model): 从模型路径或 HF Hub 加载配置
        """
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 加载 HuggingFace 模型配置（包含 hidden_size, num_layers, num_heads 等）
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 将 max_model_len 限制为不超过模型支持的最大位置
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
