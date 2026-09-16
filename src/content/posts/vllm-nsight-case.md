---
title: "vLLM 推理 Nsight 解读案例"
pubDate: 2026-09-17
description: "通过 Nsight 分析 vLLM 推理过程中的 CUDA 初始化、kernel 和通信行为。"
author: "Westwoods"
tags: [vLLM, 性能分析, AI Infra]
draft: false
---

我用 nsight 看了一下一个经典的 vllm 推理进程：
```python
from vllm import LLM, SamplingParams
import torch
import time

MODEL_PATH = "/models/Qwen2.5-32B-Instruct-FP8-Dynamic"

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=2,
    max_model_len=8*1024,
    kv_cache_dtype='fp8'
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)

# warmup：让 CUDA / vLLM 先初始化一下
print("Warmup...")
llm.generate(["hello"], sampling_params)
torch.cuda.synchronize()

time.sleep(1)

# 正式观测负载
print("Profiling workload...")
prompts = [
    "请解释 vLLM 推理中的 prefill 和 decode 阶段。"
] * 4

outputs = llm.generate(prompts, sampling_params)
torch.cuda.synchronize()

for output in outputs:
    print(output.outputs[0].text[:120])
```

具体命令：
```bash
nsys profile \
    -o /profiles/vllm_offline_b4_fork \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --trace-fork-before-exec=true \
    --wait=primary \
    --force-overwrite=true \
    python /workspace/profile_vllm_offline.py
```

# 解读
![](/blog-assets/vLLM推理Nsight解读案例-1777644150213.png)

- 贴底黑线：进程存在 / 极低 CPU 活动 / 时间线基线
- 竖起来的黑柱：CPU 使用率变高
- 灰色背景块：聚合/概览层的活动区域
- 空白：基本没有该类活动，或者该进程/线程未运行/未采样到

进一步展开：
![](/blog-assets/vLLM推理Nsight解读案例-1777644806465.png)
CUDA HW 的意思是 CUDA hardware，指的是其硬件设备上的活动，在这里也就是 GPU。这里的 `0000:41:00.0`、`0000:a1:00.0` 是 GPU 的 PCIe 地址。

我们发现只有前两个有 CUDA HW，说明这两个进程肯定参与了推理。不过它们谁是主进程还不能确定。我们继续把 CUDA HW 展开：

![](/blog-assets/vLLM推理Nsight解读案例-1777645247861.png)

这下面是很多关于 Stream 的信息。Stream 大概是 GPU 任务队列，任务从 CPU 中来，在 GPU 中计算。图里看到：

```
72.9% Default stream 726.9% Stream 25
```

这表示在这个 CUDA HW 轨道里，大部分活动发生在 `Default stream 7`，另一部分发生在 `Stream 25`。

可以粗略理解成：

```
这张 GPU 的任务主要被提交到了两个队列：一个叫 Default stream 7一个叫 Stream 25
```

另一个进程里则是：

```
58.2% Stream 2541.8% Default stream 7
```

说明两张卡/两个 rank 使用 stream 的比例不完全一样，但都主要靠这两个 stream 干活。继续展开：
![](/blog-assets/vLLM推理Nsight解读案例-1777645733331.png)
信息变多了，展现了四类操作：
1. Graph：一组打包起来的固定 kernel 操作，一次性提交，避免多次 launch kernel 的开销，vLLM 计算图就是这个环节
2. Kernel：CPU 侧的 Python / PyTorch / vLLM 最终会把很多计算任务提交给 GPU，这些在 GPU 上运行的函数就叫 **CUDA kernel**，这是具体的计算工作，这也就是核函数（比如 GEMM 等）
3. Memory：内存相关活动，比如 attention 读取 KV cache、GEMM 读取权重，这些是 kernel 内部的全局内存访问
4. NCCL：多卡通信相关活动，比如 All reduce，All gather，ReduceScatter，Broadcast，Send/Recv

接下来继续放大，来看 64s 之后那段蓝色 kernel 活动是什么：
![](/blog-assets/vLLM推理Nsight解读案例-1777646470659.png)

我们发现这段 kernel 活动发生在 stream25，是 all reduce 原语。这说明这很可能进行了一个 TP 矩阵乘法，按照 row parallel 方式。即：
$$
AB=[A_{1},A_{2}]\begin{bmatrix}
B_{1}\\B_{2}
\end{bmatrix}
$$
而不是 column parallel，即：
$$
AB=A[B_{1},B_{2}] = [AB_{1},AB_{2}]
$$
这个只需要 All gather 原语即可。

一般来说在 vLLM 里面的 TP 思路是这样的：
```
Input
  ↓
QKV projection        通常 Column Parallel
  ↓
Attention
  ↓
Output projection     通常 Row Parallel → AllReduce
  ↓
Residual
  ↓
MLP gate/up           通常 Column Parallel
  ↓
MLP down              通常 Row Parallel → AllReduce
  ↓
Residual
```

说明这很可能是个 Output projection 或者 MLP 部分。

然后我们双击这个，看下方 events view 里面这个 ncclAllReduce 究竟有什么规律：
![](/blog-assets/vLLM推理Nsight解读案例-1777647316860.png)

我们发现它每次持续 6~7ms，间隔比较有规律。看来是一笔不小的开销。我们看看它到底属于什么操作。双击它看看周围的事件：

![](/blog-assets/vLLM推理Nsight解读案例-1777647958192.png)

我们发现后面是 RMS Norm，再然后是 quant（bf16 变 fp8），然后又是 Rotary Positional Embedding。说明本层结束了，到下一层去。这应该是 Transformer 最后那个 MLP。

> **小贴士**：一般情况下，vLLM 部署的 Causal LLM 是不需要 All gather 的。注意力计算的环节是这样的：
> 1. QK：直接把  K 竖切（Column Parallel）然后分别在两个显卡上计算 $QK_{left},QK_{right}$，Softmax 在这里利用 Flash Attention 的方法也一起完成
> 2. 不急着 All gather，把 V 横切为 $[V_{up}, V_{bottom}]^T$，也分到两个显卡上
> 3. 分别计算 $QK_{left}V_{up}$ 和 $QK_{right}V_{bottom}$，接着 All reduce 即可
> 
> 你看，通过这种方法，两个次矩阵乘法我们只用了一次通信。MLP 也是相似的原理，它一样是两次矩阵乘。
