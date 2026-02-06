"""
KV Cache 块管理器 (Block Manager)

管理 Paged KV Cache 的物理块分配、释放和 Prefix Caching。

核心概念:
  - Block: KV Cache 的基本分配单位，每个 block 存储 block_size 个 token 的 K/V
  - Paged Attention: 类似 OS 虚拟内存的分页机制，通过 block_table 映射逻辑块到物理块
  - Prefix Caching: 通过 hash 值识别相同的 token 前缀，复用已有的 KV Cache 块

Token 布局 (每个序列):
  |<---- computed (cached) ---->|<--- new_computed --->|<--- new (to compute) --->|
  |     已在 KV Cache 中的 tokens    | 命中 Prefix Cache 的 tokens |   需要新计算的 tokens     |
  |<---------- Prefix-cached tokens ---------->|<----- to be allocated ----->|

Block 状态:
  - free_block_ids:  空闲块队列（可被分配）
  - used_block_ids:  已被序列使用的块集合
  - ref_count > 0:   块正在被某个序列使用
  - ref_count == 0 且有 hash: 块内容仍有效，可被 Prefix Cache 命中复用

调用链:
  Scheduler.__init__()         -> BlockManager(num_blocks, block_size)
  Scheduler.schedule():
    - 从 waiting 队列调度:     -> get_token_layout() -> can_allocate() -> allocate()
    - 从 running 队列调度:     -> can_append() -> may_append()
    - preempt:                 -> deallocate()
  Scheduler.postprocess():
    - 完成的序列:              -> deallocate()

使用的库函数:
  - collections.deque: 空闲块队列（支持高效的两端操作）
  - xxhash.xxh64: 快速哈希函数（用于 Prefix Cache 的块内容哈希）
  - numpy.array.tobytes: 将 token id 列表转为字节串供哈希
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV Cache 物理块的元数据。

    每个 Block 对应 KV Cache 中连续的 block_size 个 token 的存储空间。
    Block 本身不存储 K/V 数据（数据在 GPU 显存中），只记录元数据。

    属性:
      block_id:  物理块 ID (0 ~ num_blocks-1)
      ref_count: 引用计数（有多少个序列正在使用此块）
                 - 0: 块空闲或仅作为 Prefix Cache 保留
                 - >0: 块正在被序列使用
      hash:      块内容的 hash 值 (用于 Prefix Caching)
                 - -1: 未计算或不完整的块
                 - 其他: xxhash64 计算的 hash
      token_ids: 块中存储的 token id 列表 (用于 Prefix Cache 验证)
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的 hash 和 token_ids（当块被完整填满时调用）。

        调用链:
          BlockManager.allocate() / may_append() -> block.update(hash, token_ids)
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块状态（分配给新序列时调用）。

        ref_count 设为 1（刚分配给一个序列），hash 和 token_ids 清空。

        调用链:
          BlockManager._allocate_block() -> block.reset()
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    KV Cache 物理块管理器。

    管理所有物理块的分配和释放，实现 Paged Attention 和 Prefix Caching。

    Blocks (or tokens) layout:

    ----------------------------------------------------------------------
    | < computed > | < new_computed > |       < new >       |
    ----------------------------------------------------------------------
    |     < Prefix-cached tokens >    |  < to be computed > |
    ----------------------------------------------------------------------
                                      | < to be allocated > |
    ----------------------------------------------------------------------
                                      |   < to be cached >  |
    ----------------------------------------------------------------------

    属性:
      block_size:        每个块存储的 token 数量 (默认 256)
      blocks:            所有物理块的列表, [Block(0), Block(1), ...]
      hash_to_block_id:  hash -> block_id 的映射 (Prefix Cache 查找表)
      free_block_ids:    空闲块 ID 队列
      used_block_ids:    已使用块 ID 集合
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器。

        调用链:
          Scheduler.__init__() -> BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        参数:
          num_blocks: 物理块总数（由 GPU 显存大小决定，ModelRunner.allocate_kv_cache 计算）
          block_size: 每个块的 token 数量
        """
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个块的 hash 值（用于 Prefix Caching）。

        hash 的计算是链式的：当前块的 hash 依赖于前一个块的 hash (prefix)，
        这确保了相同的 token 内容在不同前缀下有不同的 hash。

        调用链:
          get_token_layout() / allocate() / may_append() -> compute_hash(token_ids, prefix_hash)

        参数:
          token_ids: 当前块的 token id 列表 (必须恰好 block_size 个)
          prefix:    前一个块的 hash 值 (-1 表示第一个块)

        返回:
          64 位整数 hash 值

        使用的库函数:
          - xxhash.xxh64(): 创建 xxHash64 哈希器
          - h.update(bytes): 输入数据到哈希器
          - int.to_bytes(8, "little"): 将前缀 hash 转为 8 字节小端序
          - np.array(token_ids).tobytes(): 将 token id 列表转为字节串
          - h.intdigest(): 获取 64 位整数哈希值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前一个块的 hash 作为当前块的前缀
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个物理块：从 free 移到 used，重置块状态。

        如果该块之前有 hash 缓存（即作为 evicted 但未被覆盖的 Prefix Cache），
        先从 hash_to_block_id 中移除映射。

        调用链:
          allocate() / may_append() -> _allocate_block(block_id)

        参数:
          block_id: 要分配的物理块 ID

        返回:
          分配后的 Block 实例 (ref_count=1, hash=-1)
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 清除旧的 Prefix Cache 映射（如果有）
        if self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash, None)
        block.reset()  # ref_count = 1, hash = -1
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放一个物理块：从 used 移回 free。

        注意: 不清除 hash 信息，这样该块仍然可以作为 Prefix Cache 被复用
        （如果新序列命中了相同的 hash）。

        调用链:
          deallocate() -> _deallocate_block(block_id)

        参数:
          block_id: 要释放的物理块 ID
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, num_tokens: int) -> bool:
        """
        检查是否有足够的空闲块来容纳 num_tokens 个新 token。

        仅用于 waiting 队列中的序列（新请求），判断是否能分配所需的块。

        调用链:
          Scheduler.schedule() -> block_manager.can_allocate(num_new_computed_in_free + num_new_tokens)

        参数:
          num_tokens: 需要分配的 token 数量

        返回:
          True 如果空闲块足够
        """
        return len(self.free_block_ids) >= (num_tokens + self.block_size - 1) // self.block_size

    def get_token_layout(self, seq: Sequence):
        """
        分析序列的 token 布局，统计 Prefix Cache 命中情况。

        仅用于 waiting 队列中的序列。遍历序列的每个块，通过 hash 查找 Prefix Cache，
        统计三类 token 数量：

        1. new_computed_tokens_in_used: 命中 Prefix Cache 且块在 used 集合中
           (已被其他序列使用, 只需增加 ref_count, 不需要分配新块)
        2. new_computed_tokens_in_free: 命中 Prefix Cache 且块在 free 集合中
           (之前被释放但内容仍有效, 需要重新分配)
        3. new_tokens: 未命中缓存的 token (需要分配新块并计算)

        注意: 最后一个块即使命中也视为未命中，因为可能不完整。

        调用链:
          Scheduler.schedule() -> block_manager.get_token_layout(seq)

        参数:
          seq: 待调度的序列 (必须 block_table 为空)

        返回:
          (num_new_computed_in_used, num_new_computed_in_free, num_new_tokens)
        """
        assert not seq.block_table
        num_new_tokens = 0
        num_new_computed_tokens_in_used = 0
        num_new_computed_tokens_in_free = 0
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只有完整块才计算 hash（不完整的最后一个块 hash 为 -1）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # Cache miss 条件：hash 未找到 / token 内容不匹配 / 是最后一个块
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids or i == seq.num_blocks - 1:
                cache_miss = True
            if cache_miss:
                num_new_tokens += len(token_ids)
            else:
                if block_id in self.used_block_ids:
                    num_new_computed_tokens_in_used += len(token_ids)
                else:
                    num_new_computed_tokens_in_free += len(token_ids)
        return num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens

    def allocate(self, seq: Sequence):
        """
        为 waiting 队列中的序列分配物理块。

        分两个阶段:
          阶段 1 - 分配 Prefix Cache 命中的块:
            遍历序列的每个块，查找 Prefix Cache：
            - 命中且在 used 中: 增加 ref_count，复用块
            - 命中且在 free 中: 重新分配（从 free 移到 used）
            - 未命中: 停止（后续交给阶段 2）
            每命中一个块，seq.num_cached_tokens += block_size

          阶段 2 - 分配新块:
            为未命中缓存的 token 分配空闲块
            如果块被完整填满，计算并设置 hash（供后续 Prefix Cache 使用）

        调用链:
          Scheduler.schedule() -> block_manager.allocate(seq)

        参数:
          seq: 待分配块的序列 (block_table 必须为空)
        """
        assert not seq.block_table
        h = -1
        # === 阶段 1: 分配 Prefix Cache 命中的块 ===
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # Cache miss: hash 未找到 / 内容不匹配 / 是最后一个块
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids or i == seq.num_blocks - 1:
                break  # 停止 Prefix Cache 匹配
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                # 块已被其他序列使用，增加引用计数
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                # 块在 free 中但内容有效，重新分配
                block = self._allocate_block(block_id)
            # 更新 hash 映射（可能已存在，这里确保一致性）
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        # === 阶段 2: 为未命中的 token 分配新的空闲块 ===
        for i in range(seq.num_cached_tokens, seq.num_cached_tokens + seq.num_new_tokens, self.block_size):
            token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
            if i != seq.num_cached_tokens:
                # 不是第一个新块时更新 hash 链
                h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 从空闲队列头部取一个块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            if h != -1:
                # 完整块: 设置 hash 供后续 Prefix Cache 使用
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)


    def deallocate(self, seq: Sequence):
        """
        释放序列占用的所有物理块。

        用于：完成的序列 或 被 preempt 的序列。
        逆序遍历 block_table（先释放后面的块），减少引用计数，
        ref_count 降为 0 时释放块。

        调用链:
          Scheduler.postprocess()  -> block_manager.deallocate(seq)  # 序列完成
          Scheduler.preempt()      -> block_manager.deallocate(seq)  # 序列被抢占

        参数:
          seq: 要释放块的序列
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 重置序列的缓存状态
        seq.num_cached_tokens = 0
        seq.num_new_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        检查 running 队列中的序列是否有足够的空闲块来追加 num_new_tokens 个新 token。

        考虑最后一个已缓存块可能还有剩余空间（不需要额外分配块）。

        调用链:
          Scheduler.schedule() -> block_manager.can_append(seq, num_new_tokens)

        参数:
          seq:            running 中的序列
          num_new_tokens: 需要追加的新 token 数量

        返回:
          True 如果空闲块足够

        计算逻辑:
          last_computed_block_capacity = 最后一个已缓存块的剩余容量
          需要的新块数 = ceil((num_new_tokens - last_capacity) / block_size)
          如果 需要的新块数 <= len(free_block_ids) 则可以追加
        """
        # 最后一个块的剩余容量
        last_computed_block_capacity = self.block_size - (seq.num_cached_tokens % self.block_size)
        if last_computed_block_capacity == self.block_size:
            last_computed_block_capacity = 0  # 如果刚好整除，说明没有剩余空间
        # 需要的额外块数
        if (num_new_tokens - last_computed_block_capacity + self.block_size - 1) // self.block_size \
            <= len(self.free_block_ids):
            return True
        return False

    def may_append(self, seq: Sequence):
        """
        为 running 队列中的序列追加/更新物理块（处理新 token）。

        遍历从最后一个已缓存块开始到当前序列末尾的所有块:
          - 如果块已填满 (len(token_ids) % block_size == 0):
            - 计算 hash 并注册到 Prefix Cache
            - 如果需要新块 (block_table 不够长)，分配新块
          - 如果块未填满但需要新块:
            - 分配新块（不计算 hash，因为块还不完整）

        调用链:
          Scheduler.schedule() -> block_manager.may_append(seq)
          在每轮调度中为 running 序列更新块分配

        参数:
          seq: running 中的序列
        """
        for i in range(
            seq.num_cached_blocks * self.block_size,
            seq.num_cached_tokens + seq.num_new_tokens,
            self.block_size
        ):
            token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
            # 获取当前位置已有的块 ID (如果 block_table 够长的话)
            current_block_id = seq.block_table[i // self.block_size] \
                    if i // self.block_size < len(seq.block_table) else -1
            if current_block_id != -1:
                current_block = self.blocks[current_block_id]
                assert current_block.hash == -1  # 当前块不应已被缓存
            if len(token_ids) % self.block_size == 0:
                # 块已填满: 计算 hash 并注册到 Prefix Cache
                previous_block_id = seq.block_table[i // self.block_size - 1] if i >= self.block_size else -1
                prefix = self.blocks[previous_block_id].hash if previous_block_id != -1 else -1
                h = self.compute_hash(token_ids, prefix)
                if current_block_id == -1:
                    # 需要分配新块
                    block_id = self.free_block_ids[0]
                    current_block = self._allocate_block(block_id)
                    seq.block_table.append(block_id)
                # 设置 hash 和 token_ids，注册到 Prefix Cache
                current_block.update(h, token_ids)
                self.hash_to_block_id[h] = current_block.block_id
            elif current_block_id == -1:
                    # 块未填满但需要新块（用于存储部分 token）
                    block_id = self.free_block_ids[0]
                    self._allocate_block(block_id)
                    seq.block_table.append(block_id)
