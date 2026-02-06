"""
LLM 用户接口模块 (LLM User Interface Module)

提供 LLM 类作为用户的主要入口点。
LLM 是 LLMEngine 的别名（简单继承），保持 API 简洁。

调用链:
  用户代码 -> LLM(model_id, ...) -> LLMEngine.__init__(model_id, ...)
  用户代码 -> llm.generate(...)  -> LLMEngine.generate(...)
"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    nano-vllm 的用户接口类。

    直接继承 LLMEngine 的所有功能，不添加额外逻辑。
    提供更简洁的类名供用户使用。

    主要方法 (继承自 LLMEngine):
      - __init__(model, **kwargs): 初始化引擎
      - generate(prompts, sampling_params): 批量生成文本
      - add_request(prompt, sampling_params): 添加单个请求
      - step(): 执行一轮推理
      - is_finished(): 检查所有请求是否完成
    """
    pass
