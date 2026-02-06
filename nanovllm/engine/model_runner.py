"""
模型执行器模块 (Model Runner Module)

负责模型的加载、初始化、推理执行和多进程协调。
是连接调度器 (Scheduler) 和模型 (Qwen3ForCausalLM) 的桥梁。

核心职责:
  1. 初始化分布式环境和模型
  2. 分配 KV Cache 显存
  3. 捕获 CUDA Graph 加速 decode 阶段
  4. 将调度器输出的序列信息转换为模型输入 tensor
  5. 执行模型推理和采样
  6. 多进程间通过共享内存通信

调用链:
  LLMEngine.__init__():
    -> mp.Process(target=ModelRunner, ...)  # 启动 worker 进程 (rank > 0)
    -> ModelRunner(config, 0, events)       # 主进程 (rank 0)
  LLMEngine.step():
    -> model_runner.call("run", seqs)
    -> ModelRunner.run(seqs)
       -> prepare_model_input(seqs) -> run_model(input_ids, positions) -> sampler(logits, temperatures)

使用的库函数:
  - pickle.dumps / loads: 序列化/反序列化（SharedMemory 通信）
  - torch.distributed.init_process_group: 初始化 NCCL 分布式通信
  - torch.cuda.set_device: 设置当前 GPU
  - torch.cuda.CUDAGraph: CUDA Graph 捕获和回放
  - torch.cuda.mem_get_info / memory_stats: 查询显存信息
  - torch.inference_mode: 推理模式（禁用梯度计算）
  - multiprocessing.shared_memory.SharedMemory: 进程间共享内存
  - multiprocessing.synchronize.Event: 进程间事件同步
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型执行器，管理模型生命周期和推理执行。

    职责:
      1. 初始化: 分布式环境 -> 模型构建 -> 权重加载 -> warmup -> KV Cache 分配 -> CUDA Graph 捕获
      2. 推理: 准备输入 -> 模型前向 -> 采样
      3. 通信: rank 0 通过 SharedMemory 广播方法调用给 worker rank

    进程模型:
      - rank 0 (主进程): 由 LLMEngine 直接创建，拥有调度器和分词器
      - rank > 0 (worker 进程): 由 mp.Process 启动，进入 loop() 等待指令

    通信协议:
      rank 0 调用 call(method, *args):
        -> write_shm: 序列化 [method_name, *args] 写入共享内存
        -> event.set(): 通知所有 worker
        -> 执行本地方法
      rank > 0 在 loop() 中:
        -> event.wait(): 等待通知
        -> read_shm: 从共享内存读取方法名和参数
        -> 执行对应方法

    属性:
      config:        全局配置
      block_size:    KV Cache 块大小
      enforce_eager: 是否禁用 CUDA Graph
      world_size:    张量并行大小
      rank:          当前进程的 rank
      event:         进程间事件 (rank 0 持有 list[Event], rank > 0 持有单个 Event)
      model:         Qwen3ForCausalLM 模型实例
      sampler:       Sampler 采样器实例
      kv_cache:      KV Cache tensor
      graphs:        CUDA Graph 字典 {batch_size: CUDAGraph}
      graph_vars:    CUDA Graph 的输入/输出变量
      shm:           SharedMemory 共享内存实例
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型执行器。

        调用链:
          LLMEngine.__init__():
            -> ModelRunner(config, rank=0, events)  # 主进程
            -> mp.Process(target=ModelRunner, args=(config, rank, event))  # worker

        初始化步骤:
          1. dist.init_process_group("nccl"): 初始化 NCCL 分布式通信
          2. torch.cuda.set_device(rank): 绑定 GPU
          3. Qwen3ForCausalLM(hf_config): 构建模型（在 GPU 上，使用模型精度）
          4. load_model(model, path): 加载 safetensors 权重
          5. warmup_model(): 预热模型（记录峰值显存，用于后续 KV Cache 分配）
          6. allocate_kv_cache(): 根据剩余显存分配 KV Cache
          7. capture_cudagraph(): 捕获不同 batch size 的 CUDA Graph
          8. 创建 SharedMemory（仅多卡时）
          9. rank > 0 进入 loop() 等待指令

        参数:
          config: 全局配置
          rank:   当前进程的 rank (0 = 主进程)
          event:  rank 0 传入 list[Event], rank > 0 传入单个 Event
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 1. 初始化 NCCL 分布式通信
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 2. 绑定 GPU
        torch.cuda.set_device(rank)
        # 临时切换默认 dtype 和 device 为模型配置
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)  # 例如 bfloat16
        torch.set_default_device("cuda")
        # 3. 构建模型（参数在 GPU 上创建，使用 hf_config 指定的 dtype）
        self.model = Qwen3ForCausalLM(hf_config)
        # 4. 从 safetensors 文件加载权重
        load_model(self.model, config.model)
        # 5. 创建采样器
        self.sampler = Sampler()
        # 6. 预热（记录峰值显存）
        self.warmup_model()
        # 7. 分配 KV Cache
        self.allocate_kv_cache()
        # 8. 捕获 CUDA Graph (除非 enforce_eager)
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复默认 dtype 和 device
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 9. 多卡时设置共享内存通信
        if self.world_size > 1:
            if rank == 0:
                # 尝试清理已存在的共享内存
                try:
                    existing_shm = SharedMemory(name="nanovllm", create=False)
                    existing_shm.close()
                    existing_shm.unlink()  # 标记删除
                    print(f"Cleaned up existing shared memory: nanovllm")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error cleaning up shared memory: {e}")

                # 创建 1MB 共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等待所有 worker 就绪
            else:
                dist.barrier()
                # worker 连接到已创建的共享内存
                self.shm = SharedMemory(name="nanovllm")
                # worker 进入事件循环，等待 rank 0 的指令
                self.loop()

    def exit(self):
        """
        清理资源并退出。

        调用链:
          LLMEngine.exit() -> model_runner.call("exit")
          -> rank 0 和所有 worker 都会执行此方法

        清理步骤:
          1. 关闭共享内存
          2. barrier 同步
          3. rank 0 删除共享内存
          4. 释放 CUDA Graph 资源
          5. 等待 GPU 操作完成
          6. 销毁分布式进程组
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        Worker 进程的事件循环。

        仅 rank > 0 的 worker 进程执行。
        不断从共享内存读取方法调用指令并执行，直到收到 "exit" 命令。

        调用链:
          ModelRunner.__init__() (rank > 0) -> self.loop()
          -> while True: read_shm() -> call(method, *args) -> if "exit": break
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取方法调用指令。

        仅 rank > 0 的 worker 调用。
        等待 event 信号，然后从共享内存中反序列化方法名和参数。

        调用链: loop() -> read_shm()

        通信格式:
          shm.buf[0:4]:     数据长度 n (4 字节, 小端序)
          shm.buf[4:n+4]:   pickle 序列化的 [method_name, *args]

        使用的库函数:
          - self.event.wait(): 等待 rank 0 的通知
          - int.from_bytes(bytes, "little"): 读取数据长度
          - pickle.loads(bytes): 反序列化
          - self.event.clear(): 清除事件标志

        返回:
          (method_name, args) 元组
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 阻塞等待 rank 0 的信号
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        向共享内存写入方法调用指令。

        仅 rank 0 调用。序列化方法名和参数，写入共享内存，然后通知所有 worker。

        调用链: call() -> write_shm(method_name, *args) (rank 0 时)

        通信格式:
          同 read_shm

        使用的库函数:
          - pickle.dumps([method_name, *args]): 序列化
          - int.to_bytes(4, "little"): 写入数据长度
          - event.set(): 通知 worker
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 通知所有 worker
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        调用指定方法，并在多卡时同步到所有 worker。

        调用链:
          LLMEngine.step() -> model_runner.call("run", seqs)
          LLMEngine.exit() -> model_runner.call("exit")

        执行逻辑:
          1. rank 0: 先通过 write_shm 广播给所有 worker
          2. 所有 rank: 执行对应的本地方法

        参数:
          method_name: 要调用的方法名 (如 "run", "exit")
          *args:       方法参数

        返回:
          方法的返回值
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        模型预热：用最大输入尺寸运行一次模型，记录峰值显存。

        目的:
          1. 触发 PyTorch 的内存分配器预分配
          2. 触发 torch.compile 的编译
          3. 记录峰值显存，用于后续计算 KV Cache 可用空间

        调用链:
          ModelRunner.__init__() -> self.warmup_model()
          -> 创建 max_num_batched_tokens 个 dummy token -> self.run(seqs)

        使用的库函数:
          - torch.cuda.empty_cache(): 清空 CUDA 缓存
          - torch.cuda.reset_peak_memory_stats(): 重置峰值显存统计
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 计算 warmup 的序列数和长度
        num_seqs = max(min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs), 1)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_new_tokens = max_model_len
        # 运行一次模型（结果不使用，只关注显存占用）
        self.run(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据剩余 GPU 显存分配 KV Cache。

        调用链:
          ModelRunner.__init__() -> self.allocate_kv_cache()

        计算逻辑:
          1. 查询 GPU 显存信息:
             - total: GPU 总显存
             - used = total - free: 已使用的显存
             - peak: warmup 期间的峰值显存分配
             - current: 当前显存分配
          2. 每个 block 的字节数 = 2(K+V) * num_layers * block_size * num_kv_heads * head_dim * dtype_size
          3. 可用显存 = total * gpu_memory_utilization - used - peak + current
          4. num_blocks = 可用显存 // block_bytes

        KV Cache tensor 布局:
          shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
          - dim 0: K 和 V (0=K, 1=V)
          - dim 1: 模型层数
          - dim 2: 物理块数
          - dim 3: 每块 token 数
          - dim 4: KV head 数
          - dim 5: head 维度

        分配后，将 kv_cache 的各层切片赋值给对应 Attention 模块的 k_cache/v_cache。

        使用的库函数:
          - torch.cuda.mem_get_info(): 查询可用和总显存
          - torch.cuda.memory_stats(): 查询详细显存统计
          - torch.empty(...): 分配 KV Cache tensor
        """
        config = self.config
        hf_config = config.hf_config
        # 查询 GPU 显存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 计算每个 KV Cache block 的大小
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 每个 block: 2(K,V) * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 计算可分配的块数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 分配 KV Cache tensor: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 将 KV Cache 切片赋值给每个 Attention 层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # [num_blocks, block_size, num_kv_heads, head_dim]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        将序列的 block_table 列表转换为 padded 的 2D tensor。

        调用链:
          prepare_model_input() -> prepare_block_tables(seqs)

        Tensor 处理:
          输入: 每个 seq 的 block_table (长度可能不同)
          -> 用 -1 padding 到最大长度
          -> 转为 torch.tensor, shape [batch_size, max_num_blocks]
          -> pin_memory().cuda(non_blocking=True): 异步传输到 GPU

        使用的库函数:
          - torch.tensor(..., pin_memory=True): 创建 pinned memory tensor
          - .cuda(non_blocking=True): 异步 H2D 传输
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables


    def prepare_model_input(self, seqs: list[Sequence]):
        """
        将调度后的序列列表转换为模型输入 tensor，并设置全局 Context。

        这是推理流水线的关键环节，将序列的逻辑信息转换为 Flash Attention 需要的物理信息。

        调用链:
          ModelRunner.run(seqs)
            -> prepare_model_input(seqs)
            -> set_context(...)

        输入转换:
          对每个序列:
            1. input_ids: 取 [num_cached_tokens, num_context_tokens) 范围的 token
               (只取需要新计算的 token，跳过已缓存的)
            2. positions: 对应的位置索引
            3. cu_seqlens_q: query 累计长度 (新计算的 token 数)
            4. cu_seqlens_k: key 累计长度 (cached + new token 数)
            5. slot_mapping: 新 token 在 KV Cache 中的 slot 位置
               遍历从 num_cached_blocks 开始的所有块，计算每个 token 的物理 slot
            6. context_lens: 每个序列的上下文长度
            7. seq_need_compute_logits: 标记哪些序列需要计算 logits
               条件: 序列的所有 token 都已被处理 (len == cached + new) 且有 block_table

        如果 cu_seqlens_k > cu_seqlens_q（有 Prefix Cache 或正在 decode），
        则需要 block_tables 让 Flash Attention 从 KV Cache 中读取。

        所有 tensor 都通过 pin_memory + non_blocking 异步传输到 GPU。

        Tensor 输出:
          input_ids: shape [total_new_tokens], 所有序列新 token 的拼接
          positions: shape [total_new_tokens], 对应的位置索引

        使用的库函数:
          - torch.tensor(..., pin_memory=True): 创建 pinned memory tensor
          - .cuda(non_blocking=True): 异步 H2D 传输
          - set_context(...): 设置全局 Context
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        context_lens = []
        seq_need_compute_logits = []
        for seq_index, seq in enumerate(seqs):
            # 判断序列是否需要计算 logits:
            # 条件: 所有 token 都已处理完毕 (len == cached + new) 且有 KV Cache
            if len(seq) == seq.num_cached_tokens + seq.num_new_tokens and seq.block_table:
                seq_need_compute_logits.append(seq_index)
            context_lens.append(seq.num_context_tokens)
            # 只取需要新计算的 token (跳过已缓存的)
            input_ids.extend(seq[seq.num_cached_tokens: seq.num_context_tokens])
            positions.extend(list(range(seq.num_cached_tokens, seq.num_context_tokens)))
            seqlen_q = seq.num_new_tokens
            seqlen_k = seq.num_context_tokens
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup 时没有 block_table
                continue
            # 计算 slot_mapping: 新 token 在 KV Cache 中的物理位置
            for i in range(seq.num_cached_blocks, len(seq.block_table)):
                if i == seq.num_cached_blocks:
                    # 第一个新块: 从块内的 cached 偏移位置开始
                    start = seq.block_table[i] * self.block_size + seq.num_cached_tokens % seq.block_size
                else:
                    # 后续块: 从块的起始位置开始
                    start = seq.block_table[i] * self.block_size
                if i == len(seq.block_table) - 1:
                    # 最后一个块: 可能不满
                    end = seq.block_table[i] * self.block_size + seq.num_context_tokens % self.block_size \
                        if seq.num_context_tokens % self.block_size != 0 \
                            else (seq.block_table[i] + 1) * self.block_size
                else:
                    # 中间块: 整块
                    end = (seq.block_table[i] + 1) * self.block_size
                slot_mapping.extend(list(range(start, end)))
        # 如果有 Prefix Cache 或正在 decode (K 总长 > Q 总长)，需要 block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        # 所有 tensor 通过 pinned memory 异步传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        seq_need_compute_logits = torch.tensor(seq_need_compute_logits, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 设置全局 Context，供 Attention 层和 LM Head 读取
        set_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, seq_need_compute_logits)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样所需的温度参数。

        调用链:
          ModelRunner.run(seqs) -> prepare_sample(seqs) (仅 rank 0)

        Tensor 处理:
          1. 收集每个序列的 temperature
          2. 如果有 seq_need_compute_logits，只保留需要的序列的温度
          输出: temperatures, shape [num_seqs_need_logits] (float32, GPU)

        使用的库函数:
          - torch.tensor(..., pin_memory=True).cuda(non_blocking=True): 异步传输到 GPU
          - temperatures[indices]: 高级索引过滤
        """
        context = get_context()
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        # 只保留需要计算 logits 的序列的温度
        if context.seq_need_compute_logits.numel():
            temperatures = temperatures[context.seq_need_compute_logits]
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """
        执行模型前向传播，返回 logits。

        支持两种模式:
          1. 非 CUDA Graph: 直接调用 model.forward() + compute_logits()
          2. CUDA Graph: 将输入复制到 graph 变量 -> replay -> compute_logits()

        CUDA Graph 使用条件:
          - enforce_eager == False
          - 有 block_tables (不是 warmup)
          - input_ids 数量 == block_tables 数量 (decode: 每个序列恰好 1 个 token)
          - batch_size <= 512

        调用链:
          ModelRunner.run(seqs)
            -> run_model(input_ids, positions)
            -> model(input_ids, positions) + compute_logits(hidden_states)
               或 CUDA Graph replay + compute_logits(outputs)

        Tensor 处理:
          === 非 CUDA Graph ===
            input_ids [total_tokens] + positions [total_tokens]
            -> model(input_ids, positions): hidden_states [total_tokens, hidden_size]
            -> compute_logits(hidden_states): logits [num_seqs, vocab_size]

          === CUDA Graph ===
            1. 复制输入到 graph 变量:
               graph_vars["input_ids"][:bs] = input_ids
               graph_vars["positions"][:bs] = positions
               graph_vars["slot_mapping"][:bs] = context.slot_mapping (其余填 -1)
               graph_vars["context_lens"][:bs] = context.context_lens (其余填 0)
               graph_vars["block_tables"][:bs] = context.block_tables (其余填 0)
            2. graph.replay(): 回放捕获的 CUDA 操作
            3. compute_logits(outputs[:bs]): 取有效输出计算 logits

        使用的库函数:
          - torch.inference_mode(): 禁用梯度计算
          - graph.replay(): 回放 CUDA Graph
          - model() / model.compute_logits(): Qwen3ForCausalLM 的前向传播

        返回:
          logits: shape [num_seqs, vocab_size] (rank 0) 或 None (其他 rank)
        """
        context = get_context()
        # 判断是否使用 CUDA Graph (仅 decode 阶段: 每个序列 1 个 token)
        use_cuda_graph = (
            not self.enforce_eager
            and context.block_tables is not None
            and input_ids.size(0) == context.block_tables.size(0)
            and input_ids.size(0) <= 512
        )
        if not use_cuda_graph:
            # === 非 CUDA Graph: 直接执行 ===
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # === CUDA Graph: 复制输入 -> replay -> 取输出 ===
            bs = input_ids.size(0)
            # 找到最小的不小于 bs 的预捕获 batch size
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 将实际输入复制到 graph 的输入变量中
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # slot_mapping: 有效位置填实际值，其余填 -1 (Triton kernel 会跳过 -1)
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # context_lens: 有效位置填实际值，其余填 0
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # block_tables: 有效位置填实际值，其余填 0
            graph_vars["block_tables"].zero_()
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 回放 CUDA Graph
            graph.replay()
            # 取有效输出计算 logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence]) -> list[int]:
        """
        完整的推理流水线：准备输入 -> 模型前向 -> 采样。

        这是 LLMEngine.step() 调用的核心方法。

        调用链:
          LLMEngine.step()
            -> model_runner.call("run", seqs)
            -> ModelRunner.run(seqs)
               -> prepare_model_input(seqs)  # 准备输入 tensor + 设置 Context
               -> prepare_sample(seqs)       # 准备采样温度 (仅 rank 0)
               -> run_model(input_ids, positions)  # 模型前向
               -> sampler(logits, temperatures)     # 采样 (仅 rank 0)
               -> reset_context()            # 清理 Context

        参数:
          seqs: 调度后的序列列表

        返回:
          (token_ids, seq_need_compute_logits) 元组
          - token_ids: 采样得到的 token id 列表 (仅 rank 0, 其他 rank 为 None)
          - seq_need_compute_logits: 需要计算 logits 的序列索引 tensor
        """
        # 1. 准备模型输入 (设置全局 Context)
        input_ids, positions = self.prepare_model_input(seqs)
        # 2. 准备采样温度 (仅 rank 0 需要)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 3. 模型前向传播
        logits = self.run_model(input_ids, positions)
        # 4. 采样 (仅 rank 0)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 5. 获取哪些序列需要 logits（用于后续 postprocess）
        seq_need_compute_logits = get_context().seq_need_compute_logits
        # 6. 清理 Context
        reset_context()
        return token_ids, seq_need_compute_logits

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        为不同 batch size 捕获 CUDA Graph，加速 decode 阶段。

        CUDA Graph 将一系列 CUDA 操作捕获为一个"图"，
        replay 时整个图作为一次 kernel 启动执行，消除了 Python 开销和 kernel 启动延迟。

        只捕获 model.forward()（不包含 compute_logits），因为:
          1. forward 的输入 shape 在 decode 时固定（每个序列 1 个 token）
          2. compute_logits 需要动态索引 (seq_need_compute_logits)，不适合 CUDA Graph

        捕获的 batch size: [1, 2, 4, 8, 16, 32, ..., max_bs]
        运行时选择 >= 实际 bs 的最小 graph。

        调用链:
          ModelRunner.__init__() -> capture_cudagraph() (enforce_eager=False 时)

        步骤:
          1. 分配固定大小的输入/输出 tensor (最大 batch size)
          2. 对每个 batch size (从大到小):
             a. set_context: 设置 decode 模式的 Context (无 cu_seqlens)
             b. warmup: 运行一次确保所有 kernel 已编译
             c. capture: 在 torch.cuda.graph() 上下文中捕获
             d. 共享 graph_pool（所有 bs 共用一个内存池）
          3. 保存 graph_vars 供 run_model 使用

        使用的库函数:
          - torch.cuda.CUDAGraph(): 创建 CUDA Graph 实例
          - torch.cuda.graph(graph, pool): 捕获上下文管理器
          - graph.pool(): 获取 CUDA Graph 内存池
          - graph.replay(): 回放 CUDA Graph
          - torch.cuda.synchronize(): 等待所有 CUDA 操作完成
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 分配固定大小的输入/输出 tensor（这些 tensor 在 CUDA Graph 捕获和回放时被复用）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 需要捕获的 batch size 列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获（大的先分配内存，小的复用）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置 decode 模式的 Context（无 cu_seqlens，Attention 将使用 flash_attn_with_kvcache）
            set_context(slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # warmup: 确保所有 kernel 已编译
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # capture: 捕获 CUDA 操作
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 第一个 graph 创建内存池，后续 graph 共享
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存 graph 变量，供 run_model 使用
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
