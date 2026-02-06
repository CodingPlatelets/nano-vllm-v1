# Nano-vLLM 架构文档

## 1. 项目概述

Nano-vLLM 是一个约 1200 行的轻量级 vLLM v1 推理引擎实现，支持以下优化技术：

- **Paged Attention**: 类似 OS 虚拟内存的 KV Cache 分页管理
- **Prefix Caching**: 基于 hash 的前缀缓存复用
- **Chunked Prefill**: 长 prefill 分块处理，与 decode 交替执行
- **Tensor Parallelism**: 多 GPU 张量并行
- **CUDA Graph**: 捕获 decode 阶段的 CUDA 操作，消除 Python 开销
- **torch.compile**: 编译优化小型算子 (RMSNorm, RoPE, SiluAndMul, Sampler)

当前实现了 Qwen3 模型（支持 GQA）。

---

## 2. 项目结构

```
nanovllm/
├── __init__.py              # 包入口，导出 LLM 和 SamplingParams
├── llm.py                   # LLM 类（LLMEngine 的别名）
├── config.py                # Config 全局配置
├── sampling_params.py       # SamplingParams 采样参数
├── engine/
│   ├── llm_engine.py        # LLMEngine: 顶层引擎，组装调度器+模型+分词器
│   ├── model_runner.py      # ModelRunner: 模型加载、推理执行、CUDA Graph、多进程通信
│   ├── scheduler.py         # Scheduler: vLLM v1 调度策略
│   ├── sequence.py          # Sequence: 请求状态管理
│   └── block_manager.py     # BlockManager: KV Cache 物理块分配
├── layers/
│   ├── attention.py         # Attention: Triton KV Cache 写入 + Flash Attention
│   ├── linear.py            # 张量并行 Linear 层 (Column/Row/QKV/Merged)
│   ├── activation.py        # SiluAndMul (SwiGLU 激活)
│   ├── embed_head.py        # VocabParallelEmbedding + ParallelLMHead
│   ├── layernorm.py         # RMSNorm (含 fused add+norm)
│   ├── rotary_embedding.py  # RoPE 旋转位置编码
│   └── sampler.py           # Sampler (Gumbel-max 采样)
├── models/
│   └── qwen3.py             # Qwen3 完整模型实现
└── utils/
    ├── context.py           # 全局 Context (传递 Attention 元数据)
    └── loader.py            # Safetensors 权重加载器
```

---

## 3. 完整调用链

### 3.1 初始化流程

```
用户代码: LLM(model_id, enforce_eager=False, tensor_parallel_size=1)
│
└─► LLMEngine.__init__(model, **kwargs)
    │
    ├─► Config(model, **kwargs)
    │   └─► AutoConfig.from_pretrained(model)     # 加载 HF 配置
    │
    ├─► [多卡] mp.Process(target=ModelRunner, args=(config, rank, event))
    │   └─► worker 进程启动，进入 ModelRunner.__init__()
    │
    ├─► ModelRunner(config, rank=0, events)        # 主进程
    │   ├─► dist.init_process_group("nccl")        # 初始化 NCCL
    │   ├─► torch.cuda.set_device(rank)            # 绑定 GPU
    │   ├─► Qwen3ForCausalLM(hf_config)            # 构建模型
    │   │   └─► Qwen3Model → N × Qwen3DecoderLayer
    │   │       每层: Qwen3Attention + Qwen3MLP + 2×RMSNorm
    │   ├─► load_model(model, path)                # 加载 safetensors 权重
    │   ├─► warmup_model()                         # 预热（记录峰值显存）
    │   ├─► allocate_kv_cache()                    # 根据剩余显存分配 KV Cache
    │   ├─► capture_cudagraph()                    # 捕获不同 bs 的 CUDA Graph
    │   └─► [多卡] SharedMemory 通信初始化
    │
    ├─► AutoTokenizer.from_pretrained(model)       # 加载分词器
    │
    └─► Scheduler(config)                          # 创建调度器
        └─► BlockManager(num_blocks, block_size)   # 创建块管理器
```

### 3.2 推理流程

```
用户代码: llm.generate(prompts, sampling_params)
│
├─► add_request(prompt, sp) × N                   # 添加所有请求
│   ├─► tokenizer.encode(prompt)                   # 分词
│   ├─► Sequence(token_ids, sampling_params)       # 创建序列
│   └─► scheduler.add(seq)                         # 加入 waiting 队列
│
└─► while not is_finished():
    └─► step()                                     # 每轮推理
        │
        ├─► scheduler.schedule()                   # 调度决策
        │   ├─► 阶段1: 调度 running 队列
        │   │   ├─► block_manager.can_append()     # 检查 KV Cache 容量
        │   │   ├─► block_manager.may_append()     # 分配/更新块
        │   │   └─► [不够] preempt() 抢占队尾序列
        │   │       └─► block_manager.deallocate() # 释放 KV Cache
        │   └─► 阶段2: 调度 waiting 队列 (无 preempt 时)
        │       ├─► block_manager.get_token_layout()  # 分析 Prefix Cache
        │       ├─► block_manager.can_allocate()      # 检查容量
        │       └─► block_manager.allocate()          # 分配块
        │
        ├─► model_runner.call("run", seqs)         # GPU 推理
        │   ├─► [多卡] write_shm → event.set()    # 通知 worker
        │   └─► ModelRunner.run(seqs)
        │       ├─► prepare_model_input(seqs)      # 准备输入 tensor
        │       │   └─► set_context(...)           # 设置全局 Context
        │       ├─► prepare_sample(seqs)           # 准备温度参数
        │       ├─► run_model(input_ids, positions) # 模型前向
        │       │   └─► (见下方 Tensor 数据流)
        │       ├─► sampler(logits, temperatures)  # 采样
        │       └─► reset_context()                # 清理
        │
        └─► scheduler.postprocess(seqs, token_ids) # 后处理
            ├─► seq.append_token(token_id)         # 追加 token
            ├─► [完成] block_manager.deallocate()  # 释放 KV Cache
            └─► [未完成] 更新 num_cached_tokens    # 本轮 token 变为已缓存
```

---

## 4. Tensor 数据流

### 4.1 模型前向传播 (非 CUDA Graph)

以 batch 中有多个序列、总共 T 个新 token 为例：

```
input_ids: [T]                    # 所有序列新 token 拼接
positions: [T]                    # 对应位置索引

┌─────────────────────────────────────────────────────────────────┐
│ VocabParallelEmbedding                                          │
│   F.embedding(input_ids, weight) + all_reduce                   │
│   [T] → [T, hidden_size]                                        │
└─────────────────────────────────────────────────────────────────┘
                        ↓ hidden_states [T, H]
                        
═══ 重复 N 层 Qwen3DecoderLayer ═══

┌─────────────────────────────────────────────────────────────────┐
│ RMSNorm (input_layernorm)                                       │
│   fused add + RMSNorm (hidden_states + residual)                │
│   [T, H] → [T, H] (normalized), [T, H] (new residual)          │
└─────────────────────────────────────────────────────────────────┘
                        ↓ [T, H]

┌─────────────────────────────────────────────────────────────────┐
│ Qwen3Attention                                                   │
│                                                                  │
│ 1. qkv_proj (QKVParallelLinear, F.linear)                       │
│    [T, H] → [T, q_size + 2×kv_size]                             │
│                                                                  │
│ 2. split → q [T, q_size], k [T, kv_size], v [T, kv_size]       │
│                                                                  │
│ 3. view → q [T, num_heads, head_dim]                            │
│           k [T, num_kv_heads, head_dim]                          │
│           v [T, num_kv_heads, head_dim]                          │
│                                                                  │
│ 4. q_norm / k_norm (RMSNorm on head_dim)                        │
│                                                                  │
│ 5. rotary_emb (RoPE)                                            │
│    cos_sin_cache[positions] → apply rotation                     │
│    shape 不变                                                    │
│                                                                  │
│ 6. Attention.forward:                                            │
│    a. store_kvcache (Triton kernel):                             │
│       k, v → 写入 KV Cache 的 slot_mapping 指定位置              │
│    b. flash_attn_varlen_func:                                   │
│       q + KV Cache → o [T, num_heads, head_dim]                 │
│       (通过 block_table 从 Paged KV Cache 读取)                  │
│                                                                  │
│ 7. o.flatten(1,-1) → [T, num_heads × head_dim]                  │
│                                                                  │
│ 8. o_proj (RowParallelLinear, F.linear + all_reduce)             │
│    [T, num_heads × head_dim] → [T, H]                           │
└─────────────────────────────────────────────────────────────────┘
                        ↓ [T, H]

┌─────────────────────────────────────────────────────────────────┐
│ RMSNorm (post_attention_layernorm)                               │
│   fused add + RMSNorm                                           │
│   [T, H] → [T, H], [T, H]                                       │
└─────────────────────────────────────────────────────────────────┘
                        ↓ [T, H]

┌─────────────────────────────────────────────────────────────────┐
│ Qwen3MLP                                                        │
│                                                                  │
│ 1. gate_up_proj (MergedColumnParallelLinear, F.linear)           │
│    [T, H] → [T, 2 × intermediate_size / tp]                     │
│                                                                  │
│ 2. SiluAndMul: silu(gate) × up                                  │
│    [T, 2×I/tp] → [T, I/tp]                                      │
│                                                                  │
│ 3. down_proj (RowParallelLinear, F.linear + all_reduce)          │
│    [T, I/tp] → [T, H]                                           │
└─────────────────────────────────────────────────────────────────┘
                        ↓ [T, H]
                        
═══ 结束 N 层循环 ═══

┌─────────────────────────────────────────────────────────────────┐
│ RMSNorm (final norm)                                             │
│   fused add + RMSNorm                                           │
│   [T, H] → [T, H]                                               │
└─────────────────────────────────────────────────────────────────┘
                        ↓ hidden_states [T, H]
                        
┌─────────────────────────────────────────────────────────────────┐
│ ParallelLMHead (compute_logits)                                  │
│                                                                  │
│ 1. 取每个序列最后一个 token:                                      │
│    last_indices = cu_seqlens_q[1:] - 1                           │
│    [可选] last_indices = last_indices[seq_need_compute_logits]    │
│    x = hidden_states[last_indices] → [B, H]                     │
│                                                                  │
│ 2. F.linear(x, weight) → [B, vocab_size / tp]                   │
│                                                                  │
│ 3. [多卡] dist.gather → torch.cat → [B, vocab_size] (rank 0)    │
└─────────────────────────────────────────────────────────────────┘
                        ↓ logits [B, vocab_size]
                        
┌─────────────────────────────────────────────────────────────────┐
│ Sampler                                                          │
│                                                                  │
│ 1. logits / temperature → 温度缩放                               │
│ 2. softmax → 概率分布                                            │
│ 3. Gumbel-max: argmax(probs / Exp(1)) → token_ids [B]           │
└─────────────────────────────────────────────────────────────────┘
                        ↓ token_ids [B]
```

### 4.2 CUDA Graph Decode 路径

在 decode 阶段（每个序列 1 个 token），使用 CUDA Graph 加速：

```
1. 将 input_ids, positions, slot_mapping, context_lens, block_tables
   复制到 CUDA Graph 的预分配变量中

2. graph.replay()  → 回放捕获的 CUDA 操作
   内部执行: model(input_ids, positions) → hidden_states
   Attention 使用 flash_attn_with_kvcache (非 varlen 版本)

3. compute_logits(outputs[:bs]) → logits (在 graph 外执行)

4. sampler(logits, temperatures) → token_ids
```

---

## 5. 关键机制

### 5.1 Paged Attention

类似操作系统的虚拟内存分页机制：

- **物理块**: KV Cache 被划分为固定大小的块 (默认 256 tokens/block)
- **Block Table**: 每个序列维护一个逻辑块到物理块的映射表
- **按需分配**: 序列增长时才分配新块，避免预分配浪费
- **Triton Kernel**: `store_kvcache_kernel` 通过 `slot_mapping` 将新 K/V 写入正确的物理位置
- **Flash Attention**: 通过 `block_table` 参数从分散的物理块中读取 K/V

```
逻辑视图:   [token_0, ..., token_255] [token_256, ..., token_511] ...
                    ↓                          ↓
Block Table:    [block_5]                  [block_12]
                    ↓                          ↓
物理视图:   KV Cache[block_5]            KV Cache[block_12]
```

### 5.2 Prefix Caching

基于 hash 的前缀缓存复用机制：

- **Hash 链**: 每个块的 hash 依赖于前一个块的 hash + 本块的 token 内容
  - `hash(block_i) = xxhash64(hash(block_{i-1}) || token_ids_i)`
- **Cache 查找**: 新序列分配时，遍历块并查 `hash_to_block_id`
  - 命中: 复用已有块（增加 ref_count 或从 free 重新分配）
  - 未命中: 分配新块
- **惰性清理**: 块释放后不清除 hash，仍可被后续序列命中复用
  - 只有当空闲块被重新分配给不同内容时，旧 hash 才被清除

### 5.3 Chunked Prefill

将长 prefill 分块处理，避免阻塞 decode 请求：

- **启用条件**: `config.chunked_prefill = True`
- **分块逻辑**: `num_new_tokens = min(num_new_tokens, token_budget)`
  - 每轮只处理 `token_budget` 允许的部分
  - 未处理完的 token 在下一轮继续（通过 `num_cached_tokens` 跟踪进度）
- **好处**: 新请求的 prefill 不会独占整个 step，decode 请求可以并行处理

### 5.4 Tensor Parallelism

多 GPU 间分片权重的并行策略：

- **Column Parallel** (按输出维度分片):
  - 用于 QKV 投影、gate_up 投影
  - 各 rank 独立计算部分输出，不需要通信
  - 子类: `QKVParallelLinear`, `MergedColumnParallelLinear`

- **Row Parallel** (按输入维度分片):
  - 用于 o_proj、down_proj
  - 各 rank 计算部分内积后 `all_reduce` 求和
  - bias 只在 rank 0 加上

- **Vocab Parallel** (按词表维度分片):
  - Embedding: 各 rank 查表后 `all_reduce`
  - LM Head: 各 rank 计算后 `gather` 到 rank 0

- **通信模式**: Column → Row 配对，一次 `all_reduce` per pair

```
Column Parallel:  x [N, H] → weight_shard [H/tp, H] → y_shard [N, H/tp]
                                                        ↓
Row Parallel:     y_shard [N, H/tp] → weight_shard [H, H/tp] → z_partial [N, H]
                                                                  ↓ all_reduce
                                                                z [N, H]
```

### 5.5 CUDA Graph

捕获 decode 阶段的 GPU 操作，消除 Python 开销：

- **捕获**: 为 batch_size = [1, 2, 4, 8, 16, 32, ..., max_bs] 各捕获一个 graph
- **回放**: 运行时选 ≥ 实际 bs 的最小 graph，复制输入到预分配变量，replay
- **限制**: 只捕获 `model.forward()`，不包含 `compute_logits()`
  - 因为 `compute_logits` 需要动态索引 (`seq_need_compute_logits`)
- **Context 区别**: CUDA Graph 模式下 `cu_seqlens_q = None`
  - Attention 使用 `flash_attn_with_kvcache` 而非 `flash_attn_varlen_func`

### 5.6 多进程通信

rank 0 和 worker 之间通过 SharedMemory + Event 通信：

```
rank 0:                           rank 1, 2, ...:
  call("run", seqs)                 loop():
    write_shm:                        event.wait()       ← 阻塞等待
      pickle.dumps(data)             read_shm:
      shm.buf = data                   pickle.loads(shm.buf)
      event.set() ─────────────────►  event.clear()
    run(seqs)                         call("run", seqs)
                                        run(seqs)
```

---

## 6. 外部库依赖一览

| 库 | 用途 | 使用位置 |
|---|---|---|
| `torch` | 张量计算、GPU 管理、分布式通信 | 全局 |
| `torch.nn.functional.linear` | 线性变换 | linear.py, embed_head.py |
| `torch.nn.functional.embedding` | 词嵌入查表 | embed_head.py |
| `torch.nn.functional.silu` | SiLU 激活函数 | activation.py |
| `torch.softmax` | Softmax 概率分布 | sampler.py |
| `torch.compile` | 编译优化算子 | layernorm, rope, activation, sampler |
| `torch.distributed` | NCCL 分布式通信 (all_reduce, gather, barrier) | linear.py, embed_head.py, model_runner.py |
| `torch.cuda.CUDAGraph` | CUDA Graph 捕获和回放 | model_runner.py |
| `torch.multiprocessing` | 多进程管理 (spawn) | llm_engine.py |
| `triton` | GPU kernel 编写 (store_kvcache) | attention.py |
| `flash_attn` | Flash Attention 2 (varlen + kvcache) | attention.py |
| `transformers` | AutoConfig, AutoTokenizer, Qwen3Config | config.py, llm_engine.py, qwen3.py |
| `safetensors` | 模型权重文件读取 | loader.py |
| `xxhash` | 快速哈希 (Prefix Caching) | block_manager.py |
| `numpy` | token_ids 转字节串 (供 xxhash) | block_manager.py |
| `tqdm` | 进度条显示 | llm_engine.py |
| `pickle` | 序列化 (SharedMemory 通信) | model_runner.py |

---

## 7. 快速使用

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(model_id, enforce_eager=False, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello!"}],
        tokenize=False, add_generation_prompt=True,
    )
]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

运行方式（项目使用 uv 管理）：
```bash
uv run python example.py
```
