---
title: "vLLM"
pubDate: 2023-11-15
description: "记录 vLLM 的安装、推理服务、参数调优、分布式推理和性能分析。"
author: "Westwoods"
tags: [vLLM, LLM, 部署]
draft: false
---

vLLM 应该是当下推理速度最快的大模型推理工具，其既可以像 llama.cpp 一样建立高吞吐量的 OpenAI 兼容 API 服务器，也提供 Python API，可以和🤗的代码无缝融合起来。

不过相比 llama.cpp，vLLM 的实现还不太稳定，有时候还要挑 GPU 架构。此外，vLLM 仍然处于高速迭代之中，AI 给出的信息基本上都是过时的，因此在这里记一些内容。
# 安装
vLLM 需要许多组件：
1. `vllm`：包含 vLLM、XFormer、Flash Attention、NCCL 等组件
2. `ray[default]`：用来多节点分布式推理/训练
3. `flashinfer-python`：加速推理
# 大模型推理服务器
命令基础格式：
```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123 # 可以设置密码，避免Ollama那样的安全风险
```
vllm 的参数设置非常讲究，一不小心就会出各种难以阅读报错。比如模型路径写了什么那么模型名称就是什么，一个字符都不能错。如果写了 `NousResearch/Meta-Llama-3-8B-Instruct/` 那请求中的模型名称也必须完全一样。

调用方式和所有的 OpenAI 服务器一样，但是 vLLM 支持更多的参数，具体参见 [vLLM API 列表](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#supported-apis)。

单卡推理（默认使用可见的第一张卡）：
```bash
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve Meta-Llama-3.1-8B-Instruct \
	--dtype auto \
	--host 0.0.0.0 \
	--port 11434 \
	--max-model-len 6144 \
	--gpu-memory-utilization 0.85
```

vLLM 的多卡推理、多节点推理都被打包成了**分布式推理**这一部分。而分布式推理依赖跨进程交互（IPC）。因此如果使用 docker 容器运行 vllm，务必加上 `--ipc=host` 这一参数，并且将网络设置为 `--network=host`。以我自己打包的 vllm 镜像为例：
```bash
docker run -itd \
	--name vllm \
	--network host \
	--ipc host \
	--runtime nvidia \
	--gpus all \
	-v /home/fanghejin/LLMs_Workshop:/data \
	-e NCCL_SOCKET_IFNAME=eno2 \ # 指定网络
	-e GLOO_SOCKET_IFNAME=eno2 \
	-e VLLM_HOST_IP=172.18.147.66 \
	b410-3:5000/vllm:0.1.0 \
	/bin/bash
```

具体来说， NCCL 和 Ray 是真正负责实现分布式推理的工具，前者主要使用 IPC 负责节点内通讯，后者主要使用网络来实现节点之间通讯。 

双卡推理命令：
```bash
vllm serve Qwen2.5-32B-Instruct-FP8-Dynamic \
	--host 0.0.0.0 \
	--port 12434 \
	--tensor-parallel-size 2 \
	--gpu-memory-utilization 0.95 \
	--max_num_seqs 64 \
	--max-model-len 10240 \
	--kv-cache-dtype fp8 \
	--enable-prefix-caching
```
# 加速
要想加快速度，首先确保安装了 `flashInfer`。这个包对系统、torch、CUDA 版本要求比较严，记得查阅 [Installation - FlashInfer 0.2.8 documentation](https://docs.flashinfer.ai/installation.html)。
## 参数调优
vLLM 有许多启动参数，了解一下可以调优
```bash
config: model='Qwen2.5-32B-Instruct-FP8-Dynamic', 
speculative_config=None, 
tokenizer='Qwen2.5-32B-Instruct-FP8-Dynamic', 
skip_tokenizer_init=False, 
tokenizer_mode=auto, 
```


```
revision=None, 
override_neuron_config={}, 
tokenizer_revision=None, 
trust_remote_code=False, 
dtype=torch.bfloat16, 
max_seq_len=10240, 
download_dir=None, 
load_format=auto, 
tensor_parallel_size=2, 
pipeline_parallel_size=1, 
disable_custom_all_reduce=False, 
quantization=compressed-tensors, 
enforce_eager=False, 
kv_cache_dtype=fp8,  
device_config=cuda, 
decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=Qwen2.5-32B-Instruct-FP8-Dynamic, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}, use_cached_outputs=True
```

# 分布式推理
Ray 可以实现多节点分布式推理以及训练。vLLM 和 Ray 是一起使用的，建议打包为一个镜像。

Ray 需要 `pip` 单独下载，我建议下载 `ray[default]`，这个包含完整的包，免得后面麻烦。

和 Kubernetes 一样，Ray 需要指定一个头结点，在头结点容器内：
```bash
ray start --head --port=6666 --node-ip-address "172.18.147.66" --dashboard-host "0.0.0.0"
```
6379 就会是 Ray 用来进行结点间交流的端口。在其他的工作结点容器内：
```bash
ray start --address='172.18.147.66:6666' --node-ip-address "172.18.147.77"
```
然后就可以在头结点的 8265 端口获得可视化 web 看板了：

![](/blog-assets/vLLM-1755407810312.png)

> [!warning]
> Ray 需要 Redis 支持，但是 redis 只能在一台机器上支持一个 ray 实例。因此一台机器是不能启用两个 Ray 实例的，哪怕有容器也不行。会爆出 `AssertionError: Session name session_2025-08-17_04-54-11_300193_214 does not match persisted value b'session_2025-08-16_11-20-26_903580_7408'. Perhaps there was an error connecting to Redis.` 错误。

```
export NCCL_SOCKET_IFNAME=eno2
export GLOO_SOCKET_IFNAME=eno2

export VLLM_HOST_IP=172.18.147.66
```

```bash
vllm serve /data/Qwen2.5-32B-Instruct-FP8-Dynamic \
	--data-parallel-size 2 \
	--data-parallel-backend=ray \
	--host 0.0.0.0 \
	--port 12434 \
	--tensor-parallel-size 2 \
	--gpu-memory-utilization 0.85 \
	--max-model-len 8192 \
	--kv-cache-dtype fp8
```
# 性能分析
vllm 提供一个 `metric` 接口实时提供运行信息。默认在日志里面出现：
```
INFO 11-24 03:09:29 [metrics.py:386] Avg prompt throughput: 1514.3 tokens/s, Avg generation throughput: 313.4 tokens/s, Running: 35 reqs, Swapped: 0 reqs, Pending: 5 reqs, GPU KV cache usage: 95.0%, CPU KV cache usage: 0.0%.
INFO 11-24 03:09:29 [metrics.py:402] Prefix cache hit rate: GPU: 62.26%, CPU: 0.00%
```
