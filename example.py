"""
nano-vllm 使用示例 (Usage Example)

展示如何使用 nano-vllm 引擎进行离线批量推理。

调用链:
  main()
    -> AutoTokenizer.from_pretrained(): 加载分词器
    -> LLM(model_id, ...): 初始化推理引擎
       -> LLMEngine.__init__()
          -> Config() -> ModelRunner() -> Scheduler()
    -> SamplingParams(): 设置采样参数
    -> tokenizer.apply_chat_template(): 将 prompt 转为 chat 格式
    -> llm.generate(prompts, sampling_params): 批量推理
       -> add_request() * N -> step() * M -> decode outputs
    -> 打印结果

使用的库函数:
  - transformers.AutoTokenizer.from_pretrained(): 加载分词器
  - tokenizer.apply_chat_template(): 应用 chat 模板格式化 prompt
  - nanovllm.LLM: 推理引擎
  - nanovllm.SamplingParams: 采样参数
"""

import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """
    主函数：执行离线批量推理。

    步骤:
      1. 指定模型 ID（支持本地路径或 HuggingFace Hub ID）
      2. 加载分词器（用于 chat 模板格式化）
      3. 初始化 LLM 引擎:
         - enforce_eager=False: 启用 CUDA Graph 加速 decode
         - tensor_parallel_size=1: 单卡推理
      4. 设置采样参数:
         - temperature=0.6: 适中的随机性
         - max_tokens=256: 最多生成 256 个 token
      5. 格式化 prompt 为 Qwen3 的 chat 模板格式
      6. 调用 generate 执行批量推理
      7. 打印结果
    """
    # 模型路径或 HuggingFace 模型 ID
    model_id = "Qwen/Qwen3-14B"
    # 加载分词器（用于格式化 chat 模板）
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 初始化推理引擎
    llm = LLM(model_id, enforce_eager=False, tensor_parallel_size=1)

    # 采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # 原始 prompt 列表
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 将 prompt 转为 Qwen3 的 chat 模板格式
    # apply_chat_template 会添加系统提示和 assistant 回复开头标记
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 批量推理
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
