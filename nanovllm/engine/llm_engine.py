"""
LLM 推理引擎模块 (LLM Engine Module)

顶层引擎类，组装调度器 (Scheduler)、模型执行器 (ModelRunner) 和分词器 (Tokenizer)，
提供 generate() 接口完成从文本输入到文本输出的完整推理流程。

初始化流程:
  1. 创建 Config（加载 HF 模型配置）
  2. 多进程创建 ModelRunner（每个 GPU 一个进程）
  3. 加载 Tokenizer
  4. 创建 Scheduler

推理流程:
  generate(prompts, sampling_params)
    -> add_request(prompt, sp) * N   # tokenize + 创建 Sequence + 加入 waiting
    -> while not is_finished():
         step()                      # schedule -> run -> postprocess
    -> decode outputs               # 解码 token ids 为文本

调用链:
  用户代码 (example.py):
    -> LLM(model_id, ...) -> LLMEngine.__init__()
    -> llm.generate(prompts, sampling_params)
       -> add_request() * N -> step() * M -> return outputs

使用的库函数:
  - atexit.register: 注册退出时的清理函数
  - dataclasses.fields: 获取 Config 的字段名列表
  - time.perf_counter: 高精度计时（吞吐量统计）
  - tqdm.auto.tqdm: 进度条
  - transformers.AutoTokenizer.from_pretrained: 加载分词器
  - torch.multiprocessing: 多进程创建（spawn 模式）
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM 推理引擎，nano-vllm 的核心入口类。

    组装了推理所需的所有组件:
      - ModelRunner: 管理模型和 GPU 推理
      - Scheduler:   管理请求调度和 KV Cache
      - Tokenizer:   文本分词和解码

    支持:
      - 批量推理 (generate): 一次提交多个 prompt
      - 单步推理 (step): 手动控制推理循环
      - 多 GPU 张量并行: 自动启动 worker 进程
    """

    def __init__(self, model, **kwargs):
        """
        初始化推理引擎。

        调用链:
          用户代码 -> LLM(model_id, enforce_eager=..., tensor_parallel_size=...)
          -> LLMEngine.__init__(model, **kwargs)

        初始化步骤:
          1. 创建 Config: 解析 kwargs 中与 Config 匹配的参数，加载 HF 配置
          2. 启动 worker 进程 (rank 1 ~ tp_size-1):
             每个 worker 是一个独立进程，运行 ModelRunner
          3. 创建主进程 ModelRunner (rank 0):
             加载模型、warmup、分配 KV Cache、捕获 CUDA Graph
          4. 加载 Tokenizer
          5. 设置 EOS token id
          6. 创建 Scheduler (此时 num_kvcache_blocks 已由 ModelRunner 计算好)
          7. 注册 atexit 清理函数

        参数:
          model:    模型路径或 HuggingFace 模型 ID
          **kwargs: 可选配置参数（会自动过滤，只保留 Config 支持的字段）
                    如 enforce_eager, tensor_parallel_size, max_num_batched_tokens 等

        使用的库函数:
          - dataclasses.fields(Config): 获取 Config 的所有字段名
          - mp.get_context("spawn"): 获取 spawn 模式的多进程上下文
          - ctx.Event(): 创建进程间事件
          - ctx.Process(target, args): 创建子进程
          - AutoTokenizer.from_pretrained(): 加载分词器
          - atexit.register(): 注册退出时的清理函数
        """
        # 从 kwargs 中提取 Config 支持的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 启动 worker 进程 (rank 1 ~ tp_size-1)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 创建主进程 ModelRunner (rank 0)
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 创建调度器（此时 config.num_kvcache_blocks 已由 ModelRunner 计算好）
        self.scheduler = Scheduler(config)
        # 注册退出时的清理函数
        atexit.register(self.exit)

    def exit(self):
        """
        清理引擎资源。

        调用链:
          atexit / 用户手动调用 -> LLMEngine.exit()
          -> model_runner.call("exit"): 通知所有 rank 退出
          -> 等待 worker 进程结束

        使用的库函数:
          - process.join(): 等待子进程结束
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个推理请求。

        调用链:
          LLMEngine.generate() -> add_request(prompt, sp)
          -> tokenizer.encode(prompt) (如果是字符串)
          -> Sequence(token_ids, sampling_params)
          -> scheduler.add(seq)

        参数:
          prompt:          文本字符串或 token id 列表
          sampling_params: 采样参数

        使用的库函数:
          - self.tokenizer.encode(prompt): 文本分词
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一轮推理步骤。

        这是引擎的核心循环单元。每次调用完成:
          1. 调度: 决定哪些序列参与本轮推理
          2. 推理: 在 GPU 上执行模型前向和采样
          3. 后处理: 追加 token、判断完成

        调用链:
          LLMEngine.generate() 循环调用
          -> scheduler.schedule(): 返回本轮参与的序列
          -> model_runner.call("run", seqs): 执行推理，返回 token ids
          -> scheduler.postprocess(): 追加 token，判断完成
          -> 收集已完成序列的输出

        返回:
          (outputs, num_total_tokens) 元组
          - outputs: [(seq_id, completion_token_ids), ...] 已完成序列的输出列表
          - num_total_tokens: 已完成序列的总 token 数（用于吞吐量统计）
        """
        # 1. 调度
        seqs = self.scheduler.schedule()
        # 2. 推理 (通过 call 同步到所有 rank)
        token_ids, seq_need_compute_logits = self.model_runner.call("run", seqs)
        # 3. 后处理
        self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)
        # 收集已完成序列的输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_total_tokens = sum(len(seq) for seq in seqs if seq.is_finished)
        return outputs, num_total_tokens

    def is_finished(self):
        """
        判断所有请求是否已完成。

        调用链: generate() 循环条件 -> is_finished()
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量推理的主入口：提交多个 prompt，返回所有生成结果。

        调用链:
          用户代码 -> llm.generate(prompts, sampling_params)
          -> add_request() * N  # 添加所有请求
          -> while not is_finished(): step()  # 循环推理直到全部完成
          -> tokenizer.decode()  # 解码为文本

        流程:
          1. 将所有 prompt 添加到 scheduler 的 waiting 队列
          2. 循环调用 step()，每步:
             - 调度一批序列
             - 执行模型推理
             - 后处理（追加 token、完成判断）
             - 收集已完成的输出
          3. 按 seq_id 排序输出（保证与输入顺序一致）
          4. 解码为文本

        参数:
          prompts:         文本列表或 token id 列表的列表
          sampling_params: 采样参数（单个或与 prompts 等长的列表）
          use_tqdm:        是否显示进度条

        返回:
          输出列表: [{"text": str, "token_ids": list[int]}, ...]
          顺序与输入 prompts 一一对应

        使用的库函数:
          - tqdm(total, desc): 进度条
          - time.perf_counter(): 高精度计时
          - self.tokenizer.decode(token_ids): 解码 token id 为文本
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 如果 sampling_params 不是列表，扩展为与 prompts 等长的列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 1. 添加所有请求到 waiting 队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        num_total_tokens = 0
        t = perf_counter()
        # 2. 推理循环
        while not self.is_finished():
            output, num_step_tokens = self.step()
            num_total_tokens += num_step_tokens
            # 更新进度条（显示吞吐量）
            if use_tqdm:
                total_throughput = num_total_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "total_throughput": f"{int(total_throughput)}tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        # 3. 按 seq_id 排序（保证与输入顺序一致）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 4. 解码为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
