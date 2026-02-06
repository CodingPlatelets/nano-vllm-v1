"""
调度器模块 (Scheduler Module)

实现 vLLM v1 风格的请求调度策略，管理 waiting 和 running 两个队列，
决定每轮推理哪些序列参与计算。

调度策略:
  1. 优先调度 running 队列中的序列（已在运行的请求优先继续）
  2. 如果 KV Cache 不足，从 running 队列尾部 preempt（抢占）序列
  3. 如果没有发生 preempt，再从 waiting 队列中调度新请求
  4. token_budget 控制每轮最多处理的 token 数量

支持 Chunked Prefill:
  当 enable_chunked=True 时，长 prefill 可以被分块处理，
  每轮只处理 token_budget 允许的部分，下一轮继续。

调用链:
  LLMEngine.__init__()  -> Scheduler(config)
  LLMEngine.add_request() -> scheduler.add(seq)
  LLMEngine.step():
    -> scheduler.schedule()    # 调度决策
    -> model_runner.call("run", seqs)  # 模型推理
    -> scheduler.postprocess() # 后处理（追加 token / 完成序列）
  LLMEngine.is_finished() -> scheduler.is_finished()

使用的库函数:
  - collections.deque: waiting 和 running 队列（支持高效的两端操作）
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    请求调度器，管理推理请求的生命周期。

    核心职责:
      1. 维护 waiting/running 两个队列
      2. 每轮决定哪些序列参与推理 (schedule)
      3. 协调 BlockManager 进行 KV Cache 分配/释放
      4. 推理后处理：追加 token、判断完成、更新缓存状态

    属性:
      enable_chunked:         是否启用 Chunked Prefill
      max_model_len:          最大模型上下文长度
      max_num_seqs:           最大并发序列数
      max_num_batched_tokens: 每轮最大 token 预算
      eos:                    EOS token id
      block_manager:          KV Cache 块管理器
      waiting:                等待调度的序列队列 (deque)
      running:                正在运行的序列队列 (deque)
    """

    def __init__(self, config: Config):
        """
        初始化调度器。

        调用链:
          LLMEngine.__init__() -> Scheduler(config)

        参数:
          config: 全局配置，包含调度参数和 KV Cache 配置
        """
        self.enable_chunked = config.chunked_prefill
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        判断所有请求是否已完成。

        调用链: LLMEngine.is_finished() -> scheduler.is_finished()

        返回: True 如果 waiting 和 running 队列都为空
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新请求到 waiting 队列。

        调用链: LLMEngine.add_request() -> scheduler.add(seq)

        参数:
          seq: 新创建的 Sequence 实例

        断言: 序列长度不能超过 max_model_len - 1（留 1 个位置给生成的 token）
        """
        assert len(seq) <= self.max_model_len - 1, "Sequence length exceeds max_model_len"
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        执行一轮调度，决定哪些序列参与本轮推理。

        调度算法 (vLLM v1 风格):

        === 阶段 1: 调度 running 队列 ===
        遍历 running 队列中的每个序列:
          1. 计算需要处理的新 token 数:
             num_new_tokens = 总 token 数 - 已缓存 token 数
             如果启用 chunked prefill，限制为 min(num_new_tokens, token_budget)
             还要限制不超过 max_model_len - 1
          2. 检查 BlockManager 是否能追加:
             - 能追加: 调用 may_append 分配新块，加入本轮调度
             - 不能追加: 从 running 队列尾部 preempt 最新加入的序列，释放其 KV Cache，再重试
          3. 扣减 token_budget

        === 阶段 2: 调度 waiting 队列 (仅当阶段 1 没有 preempt 时) ===
        遍历 waiting 队列头部的序列:
          1. 通过 get_token_layout 分析 Prefix Cache 命中情况
          2. 如果启用 chunked prefill，限制新 token 数
          3. 检查 token_budget 和 BlockManager 容量
          4. 如果够用: 调用 allocate 分配块，移到 running 队列
          5. 如果不够: 停止调度

        调用链:
          LLMEngine.step()
            -> scheduler.schedule()
            -> 返回 scheduled_seqs 给 model_runner 执行

        返回:
          scheduled_seqs: 本轮参与推理的序列列表 (running 在前, new 在后)
        """
        scheduled_seqs = []
        scheduled_running_seqs = []
        scheduled_new_reqs = []
        preempted_seqs = []
        token_budget = self.max_num_batched_tokens

        # === 阶段 1: 从 running 队列调度 ===
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            seq = self.running[req_index]
            # 计算需要新计算的 token 数量
            num_new_tokens = len(seq) - seq.num_cached_tokens
            if self.enable_chunked:
                # Chunked Prefill: 限制每轮处理的 token 数
                num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - seq.num_cached_tokens
            )
            assert num_new_tokens > 0
            # 尝试追加 KV Cache 块，不够就 preempt
            while True:
                if self.block_manager.can_append(seq, num_new_tokens):
                    seq.num_new_tokens = num_new_tokens
                    self.block_manager.may_append(seq)
                    break
                # KV Cache 不足: 从 running 队列尾部 preempt 最新的序列
                preempted_seq = self.running.pop()
                self.preempt(preempted_seq)
                preempted_seqs.append(preempted_seq)
                if len(self.running) == req_index:
                    break  # 所有后续序列都被 preempt 了
            if len(self.running) == req_index:
                break
            scheduled_running_seqs.append(seq)
            token_budget -= seq.num_new_tokens
            req_index += 1

        # === 阶段 2: 从 waiting 队列调度新请求 ===
        # 只有没有 preempt 时才调度新请求（避免恶性循环: 加入新请求 -> preempt -> 再加入）
        if not preempted_seqs:
            while self.waiting and token_budget > 0 and len(self.running) < self.max_num_seqs:
                seq = self.waiting[0]
                assert not seq.block_table
                # 分析 Prefix Cache 命中情况
                num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens = \
                    self.block_manager.get_token_layout(seq)
                if self.enable_chunked:
                    num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0
                # 检查 token_budget 和 KV Cache 容量
                if num_new_tokens > token_budget or \
                    not self.block_manager.can_allocate(num_new_computed_tokens_in_free + num_new_tokens):
                    break  # 资源不足，停止调度新请求
                seq.num_new_tokens = num_new_tokens
                # 分配 KV Cache 块（包括 Prefix Cache 命中的块 + 新块）
                self.block_manager.allocate(seq)
                assert seq.num_cached_tokens == num_new_computed_tokens_in_free + \
                    num_new_computed_tokens_in_used
                token_budget -= num_new_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_new_reqs.append(seq)

        # 合并: running 序列在前，新请求在后
        scheduled_seqs = scheduled_running_seqs + scheduled_new_reqs
        assert scheduled_seqs  # 至少调度一个序列
        return scheduled_seqs


    def preempt(self, seq: Sequence):
        """
        抢占一个序列：释放其 KV Cache，移回 waiting 队列头部。

        当 KV Cache 不足时，从 running 队列尾部抢占序列（LIFO 策略）。
        被抢占的序列会在下一轮重新调度时重新分配 KV Cache 并重新计算。

        调用链:
          Scheduler.schedule() -> self.preempt(seq)

        参数:
          seq: 被抢占的序列
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        # 加到 waiting 队列头部（优先重新调度）
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], seq_need_compute_logits) -> list[bool]:
        """
        推理后处理：将生成的 token 追加到序列，判断是否完成。

        对每个生成了 token 的序列:
          1. 追加 token 到序列
          2. 判断是否完成:
             - 遇到 EOS token（且 ignore_eos=False）
             - 达到 max_tokens 生成上限
             - 序列总长度达到 max_model_len
          3. 完成的序列: 释放 KV Cache，从 running 队列移除

        对所有未完成的序列:
          更新 num_cached_tokens（本轮计算的 token 变为已缓存），重置 num_new_tokens

        调用链:
          LLMEngine.step()
            -> scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)

        参数:
          seqs:                    本轮调度的序列列表
          token_ids:               模型生成的 token id 列表
          seq_need_compute_logits: 需要计算 logits 的序列索引
                                   (与 token_ids 一一对应)
        """
        assert len(token_ids) == len(seq_need_compute_logits)
        for seq_index, token_id in zip(seq_need_compute_logits, token_ids):
            seq = seqs[seq_index]
            # 追加新生成的 token
            seq.append_token(token_id)
            # 判断是否完成
            if (not seq.ignore_eos and token_id == self.eos) or \
                seq.num_completion_tokens == seq.max_tokens or \
                    len(seq) >= self.max_model_len:
                if len(seq) >= self.max_model_len:
                    print(f"Sequence {seq.seq_id} reached max_model_len {self.max_model_len}.")
                seq.status = SequenceStatus.FINISHED
                # 释放 KV Cache
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        # 更新未完成序列的缓存状态
        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                # 本轮的新 token 变为已缓存
                seq.num_cached_tokens = seq.num_cached_tokens + seq.num_new_tokens
                seq.num_new_tokens = 0
