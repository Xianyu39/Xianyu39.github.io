---
title: "专家并行"
pubDate: 2026-06-08
description: "介绍 MoE 模型中的专家并行、token dispatch、All-to-All 通信和不同并行组合。"
author: "Westwoods"
tags: [MoE, 分布式计算, AI Infra]
draft: false
---

在 MoE 架构的 LLM 中，我们讲到我们可以把 FFN 层的计算拆分为了若干个独立计算的专家（expert），每个专家根据输入计算出一个向量，所有专家计算的向量累加就是原输出。现在我们引入一个打分器 router，router 基于输入的向量计算出一个专家数量维度的向量，经过 softmax 后可以看作给每个专家打分，我们只要 topk 的专家参与计算。这就节省了算力。

这个配置非常典型。比如 deepseek v3.2 就有这样的配置：
```
"n_routed_experts": 256, # 路由专家数量
"n_shared_experts": 1, # 共享专家，这个是必须带的
"num_experts_per_tok": 8, # 每次多少专家参与计算（topk）
```

其中 routed_experts 指的就是 router 可以选择的专家。router 模型是一个简单的线性投影层，其可以看作：
$$
f (\mathbf{x}W_{r})
$$
其中 $W_{r}$ 的 shape 是 `[hidden_size, n_routed_experts]`，输出一个 `n_routed_experts` 长度的向量。经过激活函数 $f$ 之后，就可以视作 router 为每个路由专家输出了一个权重了。对于 deepseek v3.2 来说，$f=\text{Sigmoid}$（权重并不需要像注意力一样保证和为 1）。

MoE 往往选择权重 `topk` 的专家参与计算，即 `num_experts_per_tok` 个专家。这样就做到了激活参数只有 `8+1` 个 expert，节省了算力（不是显存，所有专家都得待在显存）。接下来我们就来讨论：EP 是如何实现的。
# 经典 EP
## 切分
经典 EP 非常粗暴：**把专家均分到所有显卡上**。比如我们在 `8*b200` 上运行 deepseek v3.2，我们就会给每个显卡塞 `256/8=32` 个路由专家。这个思路非常优美，就像这样：
```
GPU 0: Expert 0-31
GPU 1: Expert 32-63
GPU 2: Expert 64-95
GPU 3: Expert 96-127
GPU 4: Expert 128-159
GPU 5: Expert 160-191
GPU 6: Expert 192-223
GPU 7: Expert 224-255
```

而像是有 shared expert 的模型，则需要用 data parallel 这样的方式把 shared expert 放到每个 GPU 上。
## 通信
EP 通信模型和 PP、TP 都不太一样。PP 主要是在层与层之间传 activation，TP 主要是在矩阵切分后做 all-reduce / all-gather。这两个并行方案都是适用于通用神经网络的。EP 则是专门针对 MoE 模型的 FFN 部分的。这就导致我们经常会在使用 EP 的时候配合别的并行方法，让通信复杂起来。这里我们会按照常见长度分情况讨论。
## Attention DP + EP
这是最经典的做法：Attention 部分使用**数据并行**，FFN 部分**专家并行**。在这种情况下，每个 GPU 都拥有完整的计算注意力的能力以及计算某几个专家输出的能力。在这里，EP 的核心通信是 **token dispatch**。

一开始，每个 GPU 都会获得 batched token 中的一些（可以理解为 scatter 了 batch），作为输入；在第一层的注意力计算完毕之后，每个 GPU 的 router 会通过打分+Topk 的方法为每个 token 的 hidden_state 找到去向。比如：
```
# 一个batch的token的vec (A1 A2 B1 B2 C1 C2 D1 D2) 被scatter到各个GPU
GPU0: A1 A2
GPU1: B1 B2
GPU2: C1 C2
GPU3: D1 D2

# 各个GPU的第一层的Attn + Router计算，确定token去向，这是一个dispatch过程（A2A Dispatch）
# 这里模拟了经典的expert不均衡现象
GPU0: A1 A2 B1
GPU1: B1 B2 A1
GPU2: C1 C2 D2 A2
GPU3: D1

# Expert计算
GPU0: A1 A2 B1
GPU1: B1 B2 A1
GPU2: C1 C2 D2 A2
GPU3: D1

# 归还Token给原GPU（A2A Combine）
GPU0: A1 A2
GPU1: B1 B2
GPU2: C1 C2
GPU3: D1 D2
```

> 理论上来说，token hidden_state在哪个 rank 计算完全没有关系，不归还（Combine）token 理论上可以省一步通信。但是实际上，这往往带来更多麻烦：
> 1.  KV-Cache 会需要跟着 token 走
> 2. token 被分发到多个 GPU 后，需要确认在哪个 GPU 聚合结果
> 
> 所以一般我们希望某个 token 始终在固定的 GPU 上计算注意力，我们会有归还步骤。
> 
> 此外这个 A2A combine 是一个极度耗时的操作。B300 上 EP8 的 DeepSeek V3.2 Decode 步骤的一个 A2ADispatchKernel 耗时 85us，而 A2ACombineKernel 则需要 777us。

在 EP 中，每个 routed expert 都有一个 owner GPU。router 对每个 token 计算出 top-k experts 后，系统需要把该 token 的 hidden state 发送到这些 experts 所在的 GPU 上。对于单个 token 来说，这看起来像是一次 one-to-many dispatch：一个 token 可能被发给多个 expert owner。expert 计算完成后，输出再返回 token 原本所在的 GPU，由 token owner 按 router 权重做加权 combine。

但是在真实推理中，通常不存在一个 root GPU 统一负责所有 token 的 router。更常见的做法是：每张 GPU 都持有一部分 token，并在本地计算这些 token 的 router 结果；同时，每张 GPU 也持有一部分 routed experts。因此，每张 GPU 都会把一部分本地 token 发送给其他 GPU，也会从其他 GPU 接收需要本地 experts 处理的 token。

从整个 batch / MoE group 的角度看，这就是 **all-to-all dispatch**：

```text
每张 GPU -> 按 expert owner 分桶 -> 发送给其他 GPU
每张 GPU <- 接收其他 GPU 发来的 token -> 计算本地 experts
```

计算完成后，还需要一次反向的数据交换，把 expert 输出返回 token owner，再进行本地加权合并。因此 EP routed branch 的典型流程是：

```text
local router
→ token permutation / grouping
→ all-to-all dispatch
→ local expert compute
→ all-to-all return
→ local weighted combine
```

所以 EP 的通信不应理解为 root GPU 上的 broadcast + reduce。对于单 token，它可以近似理解成 dispatch + combine；对于整个并行组，它就是 all-to-all dispatch + all-to-all return。
## Attention TP + EP
如果注意力没有使用 DP 而是使用 TP，那么考虑到 TP 的特性，通信情况就会不一样。

在 Flash Attention 实现下，我们一般使用 Megatron-LM 的 TP 实现注意力的张量并行。在这种情况下，单个 GPU 无法独立计算注意力，必须联合其他 GPU 才能获得结果。在这里 `GPUi` 只能计算出部分头的注意力 `Ai`，最终的结果 `A` 需要各个 GPU 的结果通过 Gather 操作 。这个归并通常使用 AllReduce 原语。

**AllReduce 的特性是 Reduce 完成同时，结果刚好分发到所有 GPU**。因此在这里**每个 GPU 都将持有所有的 token hidden_states**。后面的 EP 计算并不需要从其它 GPU 获得 token。**Dispatch 在这里变得非必要了**。

```
# 一个batch的token的vec都在主GPU上
GPU0: A1 A2 B1 B2 C1 C2 D1 D2
GPU1: 
GPU2:
GPU3:

# TP计算后会经过AllReduce一轮分发
GPU0: A1 A2 B1 B2 C1 C2 D1 D2
GPU1: A1 A2 B1 B2 C1 C2 D1 D2
GPU2: A1 A2 B1 B2 C1 C2 D1 D2
GPU3: A1 A2 B1 B2 C1 C2 D1 D2

# Expert计算，从本GPU挑选即可
GPU0: A1 A2 B1
GPU1: B1 B2 A1
GPU2: C1 C2 D2 A2
GPU3: D1

# 归还Token给原GPU（A2A Combine）
GPU0: A1 A2 B1 B2 C1 C2 D1 D2
GPU1: 
GPU2:
GPU3:
```

# 参数量、计算量、显存消耗
我们关注下面这些信息：
```
E = routed experts 数 （n_routed_experts）
K = 每 token 激活 experts 数（num_experts_per_tok）
S = shared experts 数 （n_shared_experts）
P_expert = 单个 expert 参数量
```

于是 FFN 部分的参数量就是 `P_expert*(E+S)`；每个 token 在这一层激活 K 个 expert 以及 shared expert 参与计算，激活参数量是 `P_expert*(K+S)`。那么一般每张卡放 ` E/EP+S ` 个 expert。
