# Nano-vLLM-v1

A lightweight vLLM v1 implementation built from scratch. Based on the original [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), Nano-vLLM-v1 implements an efficient request scheduling strategy similar to [vLLM v1](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) and achieves chunked prefill.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, Chunked Prefill, Paged Attention, etc.
