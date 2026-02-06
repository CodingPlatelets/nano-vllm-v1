"""
模型权重加载模块 (Model Weight Loader)

本模块负责从 safetensors 文件中加载预训练权重到模型参数中。
支持 packed modules（如 QKV 合并投影、gate_up 合并投影）的自动拆分加载，
以及张量并行（Tensor Parallelism）下的分片加载。

调用链:
  - ModelRunner.__init__() -> load_model(model, path)
    -> 遍历 safetensors 文件 -> 对每个权重调用 weight_loader() 写入对应参数

使用的库函数:
  - os.path.join: 拼接模型目录路径和文件名
  - glob.glob: 匹配模型目录下所有 .safetensors 文件
  - safetensors.safe_open: 以内存映射方式打开 safetensors 文件
  - torch.nn.Module.get_parameter: 通过 "." 分隔的名称获取模型参数
  - torch.Tensor.copy_: 将加载的权重数据复制到参数中
"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认权重加载器，直接将 loaded_weight 复制到 param.data 中。

    调用链:
      - load_model() 在参数没有自定义 weight_loader 属性时使用此函数

    Tensor 处理:
      - 输入 loaded_weight: 从 safetensors 文件中读取的完整权重 tensor
      - 输出: 直接 copy_ 到 param.data（shape 必须完全匹配）

    使用的库函数:
      - torch.Tensor.copy_: 原地复制 tensor 数据
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重的主函数，从指定路径的 safetensors 文件中加载所有权重。

    调用链:
      - 被 ModelRunner.__init__() 调用
      - 内部调用 safetensors.safe_open() 打开文件
      - 内部调用各参数的 weight_loader() 方法（如 QKVParallelLinear.weight_loader,
        ColumnParallelLinear.weight_loader 等）

    加载逻辑:
      1. 获取模型的 packed_modules_mapping（如 Qwen3ForCausalLM 定义的 QKV/gate_up 映射）
         例如: {"q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"), ...}
      2. 遍历所有 .safetensors 文件中的每个权重
      3. 检查权重名是否匹配 packed_modules_mapping 中的某个 key:
         - 如果匹配: 将原始名称中的 key 替换为合并后的名称（如 q_proj -> qkv_proj），
           获取目标参数，调用 weight_loader(param, weight, shard_id) 加载到合并参数的对应分片
         - 如果不匹配: 直接通过权重名获取参数，调用 weight_loader(param, weight) 加载

    参数:
      model: nn.Module, 待加载权重的模型实例
      path:  str, 模型权重文件所在目录（包含 .safetensors 文件）

    使用的库函数:
      - glob.glob(os.path.join(path, "*.safetensors")): 查找所有 safetensors 文件
      - safetensors.safe_open(file, "pt", "cpu"): 以 PyTorch 格式、CPU 设备打开文件
      - f.keys(): 获取 safetensors 文件中所有权重名称
      - f.get_tensor(name): 读取指定名称的 tensor
      - model.get_parameter(name): 通过 "." 分隔的名称定位模型参数
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否是需要合并加载的权重（如 q_proj, k_proj, v_proj -> qkv_proj）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        # 将原权重名中的子模块名替换为合并后的名称
                        # 例如: "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0.self_attn.qkv_proj.weight"
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # 调用带 shard_id 的 weight_loader，将权重写入合并参数的对应分片
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重，直接按名称加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
