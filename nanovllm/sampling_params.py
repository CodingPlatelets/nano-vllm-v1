"""
采样参数模块 (Sampling Parameters Module)

定义推理时的采样参数，控制文本生成的行为。

调用链:
  用户代码 -> SamplingParams(temperature=0.6, max_tokens=256)
  LLMEngine.add_request(prompt, sampling_params)
    -> Sequence(token_ids, sampling_params)
       -> 将 temperature, max_tokens, ignore_eos 存储到 Sequence 实例

使用的库函数:
  - dataclasses.dataclass: 定义参数数据类
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    文本生成采样参数。

    字段说明:
      temperature: 采样温度 (默认 1.0, 必须 > 1e-10)
                   - 越高: 分布越平坦，生成越随机/多样
                   - 越低: 越接近 greedy 采样，生成越确定
                   - 不支持 greedy (temperature=0)，需要用很小的正值代替
      max_tokens:  最大生成 token 数 (默认 64)
                   达到此数量后序列会被标记为 FINISHED
      ignore_eos:  是否忽略 EOS token (默认 False)
                   True 时即使生成了 EOS 也不停止，直到 max_tokens
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """
        参数验证：禁止 greedy 采样（temperature 太小会导致数值不稳定）。

        Gumbel-max 采样技巧在 temperature 趋近 0 时数值不稳定，
        因此要求 temperature > 1e-10。
        """
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
