"""
张量并行线性层模块 (Tensor Parallel Linear Layers)

实现多种支持张量并行 (Tensor Parallelism) 的线性层变体，用于在多 GPU 间分片权重。

类继承关系:
  LinearBase (抽象基类)
    ├─ ReplicatedLinear      — 全复制（每个 rank 持有完整权重）
    ├─ ColumnParallelLinear  — 按输出维度分片（每个 rank 持有 output_size/tp_size 行）
    │   ├─ MergedColumnParallelLinear — 多个列并行层合并（如 gate_proj + up_proj）
    │   └─ QKVParallelLinear          — Q/K/V 三个投影合并
    └─ RowParallelLinear     — 按输入维度分片（每个 rank 持有 input_size/tp_size 列）

张量并行策略:
  - Column Parallel: 权重按行（输出维度 dim=0）切分
    各 rank 独立计算部分输出，不需要 all_reduce
  - Row Parallel: 权重按列（输入维度 dim=1）切分
    各 rank 计算部分结果后需要 all_reduce 求和

调用链:
  Qwen3Attention:
    qkv_proj -> QKVParallelLinear (列并行, Q/K/V 合并)
    o_proj   -> RowParallelLinear (行并行, 需要 all_reduce)
  Qwen3MLP:
    gate_up_proj -> MergedColumnParallelLinear (列并行, gate + up 合并)
    down_proj    -> RowParallelLinear (行并行, 需要 all_reduce)

使用的库函数:
  - torch.nn.functional.linear(x, weight, bias): 线性变换 y = x @ weight.T + bias
  - torch.distributed.get_rank(): 获取当前进程在分布式组中的 rank
  - torch.distributed.get_world_size(): 获取分布式组中的进程总数
  - torch.distributed.all_reduce(tensor): 跨所有 rank 对 tensor 求和
  - torch.Tensor.narrow(dim, start, length): 沿指定维度截取子 tensor
  - torch.Tensor.chunk(n, dim): 沿指定维度均匀分割
  - torch.Tensor.copy_: 原地复制 tensor 数据
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """
    整除辅助函数，确保能整除。

    用于计算张量并行分片时的维度大小（如 hidden_size / tp_size）。
    如果不能整除会触发 AssertionError。
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    所有张量并行线性层的抽象基类。

    职责:
      1. 创建 weight 和可选的 bias 参数
      2. 为参数绑定 weight_loader 方法（用于 load_model 时按分片加载权重）
      3. 记录张量并行信息（tp_dim, tp_rank, tp_size）

    参数:
      input_size:  线性层输入维度（已经是分片后的大小）
      output_size: 线性层输出维度（已经是分片后的大小）
      bias:        是否使用偏置
      tp_dim:      权重分片的维度 (0=按行/输出维度, 1=按列/输入维度, None=不分片)

    权重:
      self.weight: shape [output_size, input_size]
      self.bias:   shape [output_size] 或 None
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 创建权重参数 [output_size, input_size]（注意: F.linear 的权重是转置的）
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 绑定自定义的权重加载器，供 load_model() 调用
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    全复制线性层：每个 rank 持有完整的权重副本。

    不做任何分片，所有 rank 的权重完全相同。
    当前代码中未直接使用，但作为基础设施保留。

    权重加载: 直接复制完整权重
    前向传播: 标准线性变换，无需通信
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        直接将完整权重复制到参数中（无分片）。

        调用链: load_model() -> weight_loader(param, loaded_weight)
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准线性变换: y = x @ weight.T + bias。

        使用的库函数:
          - F.linear(x, weight, bias): 计算 x @ weight.T + bias
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层：按输出维度（weight 的 dim=0）分片。

    每个 rank 只持有 output_size/tp_size 行权重，计算输出的一个子集。
    不需要 all_reduce（上游或下游负责聚合）。

    典型用途: Attention 的 QKV 投影、MLP 的 gate/up 投影
    这些投影的输出会在同一个 rank 上继续处理，直到遇到 RowParallelLinear 进行 all_reduce。

    权重:
      self.weight: shape [output_size / tp_size, input_size]
      self.tp_dim = 0: 加载权重时按 dim=0 切片

    权重加载:
      从完整权重的 dim=0 上截取 [tp_rank * shard_size, (tp_rank+1) * shard_size] 部分
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输出维度除以 tp_size，tp_dim=0 表示按输出维度分片
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        按列并行策略加载权重：从完整权重的 dim=0 截取当前 rank 的分片。

        调用链: load_model() -> weight_loader(param, loaded_weight)

        Tensor 处理:
          loaded_weight: shape [output_size, input_size] (完整权重)
          -> narrow(0, tp_rank * shard_size, shard_size): 截取当前 rank 的行
          -> copy_ 到 param.data: shape [output_size / tp_size, input_size]

        使用的库函数:
          - torch.Tensor.narrow(dim, start, length): 沿 dim 截取子 tensor（零拷贝视图）
          - torch.Tensor.copy_: 复制数据
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # output_size / tp_size
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        列并行线性变换: y = x @ weight_shard.T。

        Tensor 处理:
          输入: x, shape [num_tokens, input_size]
          输出: y, shape [num_tokens, output_size / tp_size]
          每个 rank 只计算部分输出维度

        使用的库函数:
          - F.linear(x, weight, bias): 计算 x @ weight.T + bias
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层：将多个列并行层合并为一个矩阵乘法。

    典型用途: MLP 中的 gate_proj 和 up_proj 合并为 gate_up_proj
      gate_proj: hidden_size -> intermediate_size
      up_proj:   hidden_size -> intermediate_size
      合并后:    hidden_size -> 2 * intermediate_size (一次矩阵乘法)

    权重布局 (以 gate_up 为例, tp_size=2, rank=0):
      完整权重: [gate_weights (intermediate_size rows), up_weights (intermediate_size rows)]
      rank 0:   [gate_shard_0 (intermediate_size/2), up_shard_0 (intermediate_size/2)]
      rank 1:   [gate_shard_1 (intermediate_size/2), up_shard_1 (intermediate_size/2)]

    权重加载:
      每次加载一个子模块的权重（通过 loaded_shard_id 标识是第几个子模块）
      定位到合并权重中的正确偏移位置，截取并复制
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes  # 例如 [intermediate_size, intermediate_size]
        # 总输出 = sum(output_sizes)，父类会除以 tp_size
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        按子模块分片加载权重到合并参数的对应位置。

        调用链: load_model() -> weight_loader(param, loaded_weight, shard_id)
          shard_id=0 对应 gate_proj, shard_id=1 对应 up_proj

        Tensor 处理:
          param_data: shape [sum(output_sizes) / tp_size, input_size] (合并后的完整分片)
          -> narrow(0, shard_offset, shard_size): 定位到子模块在合并权重中的位置
          loaded_weight: shape [output_sizes[shard_id], input_size] (单个子模块的完整权重)
          -> chunk(tp_size, 0)[tp_rank]: 截取当前 rank 的行
          -> copy_ 到 param_data 的对应位置

        使用的库函数:
          - torch.Tensor.narrow(dim, start, length): 定位子 tensor
          - torch.Tensor.chunk(n, dim): 均匀分割
          - torch.Tensor.copy_: 复制数据
        """
        param_data = param.data
        # 计算当前子模块在合并权重中的偏移（已除以 tp_size）
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 当前子模块分片的大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 在合并参数中定位到正确的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中截取当前 rank 的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 合并列并行线性层：将 Q、K、V 三个投影合并为一个矩阵乘法。

    典型用途: Attention 中的 q_proj, k_proj, v_proj 合并为 qkv_proj
      q_proj: hidden_size -> num_heads * head_dim
      k_proj: hidden_size -> num_kv_heads * head_dim
      v_proj: hidden_size -> num_kv_heads * head_dim
      合并后: hidden_size -> (num_heads + 2 * num_kv_heads) * head_dim

    权重布局 (每个 rank):
      [Q_shard (num_heads/tp * head_dim), K_shard (num_kv_heads/tp * head_dim), V_shard (num_kv_heads/tp * head_dim)]

    权重加载:
      通过 loaded_shard_id ("q"/"k"/"v") 标识当前加载的是哪个投影
      计算在合并权重中的偏移和大小，截取并复制
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 每个 rank 分到的 head 数量
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 总输出大小 = (Q + K + V) * head_size
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        按 Q/K/V 标识加载权重到合并参数的对应位置。

        调用链: load_model() -> weight_loader(param, loaded_weight, "q"/"k"/"v")

        Tensor 处理:
          合并参数布局 [Q部分 | K部分 | V部分]:
            Q: offset=0, size=num_heads * head_size
            K: offset=num_heads * head_size, size=num_kv_heads * head_size
            V: offset=num_heads * head_size + num_kv_heads * head_size, size=num_kv_heads * head_size
          loaded_weight: 单个投影的完整权重
          -> chunk(tp_size, 0)[tp_rank]: 截取当前 rank 的行
          -> copy_ 到合并参数的对应偏移位置

        使用的库函数:
          - torch.Tensor.narrow(dim, start, length): 定位子 tensor
          - torch.Tensor.chunk(n, dim): 均匀分割
          - torch.Tensor.copy_: 复制数据
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        # 定位到合并参数中 Q/K/V 对应的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中截取当前 rank 的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层：按输入维度（weight 的 dim=1）分片。

    每个 rank 持有 input_size/tp_size 列权重，计算部分内积后通过 all_reduce 求和得到完整输出。

    典型用途: Attention 的 o_proj、MLP 的 down_proj
    这些层的输入是列并行层的输出（已经是分片的），行并行层将其聚合。

    Column Parallel -> Row Parallel 配对:
      Column: x [N, H] -> y_shard [N, H/tp]  (每个 rank 只计算部分输出)
      Row:    y_shard [N, H/tp] -> z_partial [N, H] -> all_reduce -> z [N, H]

    权重:
      self.weight: shape [output_size, input_size / tp_size]
      self.tp_dim = 1: 加载权重时按 dim=1 切片

    前向传播:
      1. y = x @ weight_shard.T + bias (部分结果)
      2. all_reduce(y): 跨所有 rank 求和得到完整结果
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度除以 tp_size，tp_dim=1 表示按输入维度分片
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        按行并行策略加载权重：从完整权重的 dim=1 截取当前 rank 的分片。

        调用链: load_model() -> weight_loader(param, loaded_weight)

        Tensor 处理:
          loaded_weight: shape [output_size, input_size] (完整权重)
          -> narrow(1, tp_rank * shard_size, shard_size): 截取当前 rank 的列
          -> copy_ 到 param.data: shape [output_size, input_size / tp_size]

        使用的库函数:
          - torch.Tensor.narrow(dim, start, length): 沿 dim 截取子 tensor
          - torch.Tensor.copy_: 复制数据
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # input_size / tp_size
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行并行线性变换 + all_reduce 聚合。

        调用链:
          Qwen3Attention.forward() -> self.o_proj(o.flatten(1, -1))
          Qwen3MLP.forward() -> self.down_proj(x)

        Tensor 处理:
          输入: x, shape [num_tokens, input_size / tp_size] (来自列并行层的输出)
          -> F.linear(x, weight, bias): shape [num_tokens, output_size] (部分结果)
          -> all_reduce(y): 跨所有 rank 求和, 最终 shape [num_tokens, output_size]
          注意: bias 只在 rank 0 加上，避免 all_reduce 后 bias 被加多次

        使用的库函数:
          - F.linear(x, weight, bias): 线性变换
          - dist.all_reduce(y): NCCL all_reduce 求和通信
        """
        # rank 0 加 bias，其他 rank 不加（all_reduce 后 bias 只算一次）
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 跨所有 GPU 对部分结果求和
            dist.all_reduce(y)
        return y
