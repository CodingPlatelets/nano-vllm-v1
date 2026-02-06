"""
nano-vllm 包入口 (Package Entry Point)

导出用户需要的两个核心类:
  - LLM: 推理引擎入口（LLMEngine 的别名）
  - SamplingParams: 采样参数配置

使用示例:
  from nanovllm import LLM, SamplingParams
  llm = LLM("Qwen/Qwen3-14B")
  outputs = llm.generate(["Hello!"], SamplingParams(temperature=0.6))
"""

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
