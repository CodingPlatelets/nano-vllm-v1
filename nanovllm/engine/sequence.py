"""
序列状态管理模块 (Sequence State Management)

定义推理请求的状态表示，每个用户请求对应一个 Sequence 实例。
Sequence 记录了 token 数据、调度状态、KV Cache 分配信息和采样参数。

状态机:
  WAITING -> RUNNING -> FINISHED
                 ↓
              WAITING  (preempt 回退)

调用链:
  LLMEngine.add_request() -> Sequence(token_ids, sampling_params)
  Scheduler.schedule()    -> 读取/修改 Sequence 的 num_cached_tokens, num_new_tokens, block_table 等
  Scheduler.postprocess() -> seq.append_token(token_id)
  ModelRunner.prepare_model_input() -> 读取 seq 的各种属性准备模型输入
  BlockManager.*()        -> 读取/修改 seq 的 block_table, num_cached_tokens 等

使用的库函数:
  - copy.copy: 浅拷贝 token_ids 列表
  - enum.Enum, enum.auto: 定义序列状态枚举
  - itertools.count: 生成全局唯一序列 ID
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举。

    - WAITING:  等待调度（在 Scheduler.waiting 队列中）
    - RUNNING:  正在运行（在 Scheduler.running 队列中，已分配 KV Cache）
    - FINISHED: 已完成（遇到 EOS / 达到 max_tokens / 达到 max_model_len）
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    单个推理请求的状态表示。

    Token 布局 (与 BlockManager 配合):
      |<--- num_cached_tokens --->|<--- num_new_tokens --->|<--- 未来生成的 tokens --->|
      |       已缓存的 tokens       |   本轮需要计算的 tokens  |                          |
      |<------------- num_context_tokens -------------->|
      |<---------------------- num_tokens (总长度) ------------------------------>|

    属性:
      seq_id:             全局唯一序列 ID
      status:             当前状态 (WAITING/RUNNING/FINISHED)
      token_ids:          完整 token 列表 (prompt + 已生成的 completion)
      last_token:         最后一个 token (用于快速访问)
      num_tokens:         总 token 数量
      num_prompt_tokens:  prompt 部分的 token 数量
      num_cached_tokens:  已写入 KV Cache 的 token 数量 (不需要重新计算)
      num_new_tokens:     本轮调度中需要新计算的 token 数量
      block_table:        物理块号列表 (KV Cache 的 Paged 映射表)
      temperature:        采样温度
      max_tokens:         最大生成 token 数
      ignore_eos:         是否忽略 EOS token

    类变量:
      block_size: KV Cache 块大小 (默认 256 tokens/block)
      counter:    全局序列 ID 计数器 (itertools.count)
    """
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列。

        调用链:
          LLMEngine.add_request(prompt, sampling_params)
            -> Sequence(token_ids, sampling_params)

        参数:
          token_ids:       prompt 的 token id 列表
          sampling_params: 采样参数 (temperature, max_tokens, ignore_eos)
        """
        self.seq_id = next(Sequence.counter)        # 全局唯一 ID
        self.status = SequenceStatus.WAITING         # 初始状态为等待调度
        self.token_ids = copy(token_ids)             # 浅拷贝 token 列表（避免外部修改影响）
        self.last_token = token_ids[-1]              # 最后一个 token
        self.num_tokens = len(self.token_ids)        # 总 token 数量
        self.num_prompt_tokens = len(token_ids)      # prompt token 数量（固定不变）
        self.num_cached_tokens = 0                   # 已写入 KV Cache 的 token 数量
        self.num_new_tokens = 0                      # 本轮需要计算的新 token 数量
        self.block_table = []                        # KV Cache 物理块号表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回序列的总 token 数量。"""
        return self.num_tokens

    def __getitem__(self, key):
        """
        支持对 token_ids 的索引和切片访问。

        调用链:
          BlockManager.allocate() / may_append() -> seq[start:end] 获取 token 子序列
          ModelRunner.prepare_model_input() -> seq[start:end] 获取需要计算的 token
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已完成。被 LLMEngine.step() 调用以收集输出。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        已生成的 completion token 数量。

        被 Scheduler.postprocess() 调用，用于判断是否达到 max_tokens。
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_context_tokens(self):
        """
        上下文 token 总数 = 已缓存 + 本轮新计算。

        被 ModelRunner.prepare_model_input() 调用，用于确定输入范围和 slot_mapping。
        """
        return self.num_cached_tokens + self.num_new_tokens

    @property
    def prompt_token_ids(self):
        """返回 prompt 部分的 token id 列表。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """
        返回已生成的 completion token id 列表。

        被 LLMEngine.step() 调用，用于收集已完成序列的输出。
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """
        已完全缓存的块数量（不含部分填充的最后一个块）。

        被 BlockManager.may_append() 调用，确定需要更新的块范围起点。
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_current_blocks(self):
        """
        当前已分配的块数量（应等于 block_table 长度）。

        包含断言检查，确保 block_table 与 token 数量一致。
        """
        assert (self.num_cached_tokens + self.num_new_tokens + self.block_size - 1) // self.block_size == len(self.block_table)
        return len(self.block_table)

    @property
    def num_blocks(self):
        """
        序列所有 token 需要的总块数（向上取整）。

        被 BlockManager.get_token_layout() 和 allocate() 调用。
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        """
        获取第 i 个块对应的 token id 列表。

        调用链:
          BlockManager.get_token_layout() / allocate() -> seq.block(i)

        参数:
          i: 块索引，范围 [0, num_blocks)

        返回:
          第 i 个块的 token id 列表，最后一个块可能不满 block_size
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        追加一个新生成的 token。

        调用链:
          Scheduler.postprocess() -> seq.append_token(token_id)
          在模型输出 token 后被调用

        参数:
          token_id: 新生成的 token id
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        assert self.num_tokens == len(self.token_ids)

    def __getstate__(self):
        """
        序列化接口，用于通过 SharedMemory 在多进程间传递序列状态。

        调用链:
          ModelRunner.write_shm() -> pickle.dumps(seqs) -> Sequence.__getstate__()
          仅序列化必要字段，省略 seq_id/status/max_tokens/ignore_eos
          （这些在 worker rank 中不需要）

        返回: 元组 (token_ids, last_token, num_tokens, num_prompt_tokens,
                     num_cached_tokens, num_new_tokens, block_table, temperature)
        """
        return (self.token_ids, self.last_token, self.num_tokens, self.num_prompt_tokens, \
                self.num_cached_tokens, self.num_new_tokens, self.block_table, self.temperature)

    def __setstate__(self, state):
        """
        反序列化接口，用于在 worker 进程中恢复序列状态。

        调用链:
          ModelRunner.read_shm() -> pickle.loads(data) -> Sequence.__setstate__()
        """
        self.token_ids, self.last_token, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, \
            self.num_new_tokens, self.block_table, self.temperature = state
