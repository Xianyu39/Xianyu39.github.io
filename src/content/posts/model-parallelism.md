---
title: "模型并行方式"
pubDate: 2026-09-17
description: "在多 GPU 以及多结点分布式推理中，模型并行推理和训练主要有流水线并行、张量并行和数据并行。"
author: "Westwoods"
tags: [AI Infra, 分布式计算, 深度学习]
draft: false
---

在多 GPU 以及多结点分布式推理中，模型并行推理和训练主要有 3 种：
1. Pipeline Parallel（PP）：流水线并行
2. Tensor Parallel（TP）：张量并行
3. Data Parallel（DP）：数据并行

# 流水线并行
流水线并行的思路是**利用模型可以分成若干层的特点，让每个 GPU 的显存都装载一部分层**。在前面的层的 GPU 计算完毕之后，后面的 GPU 会收到前面的层计算出来的中间结果，并且接力计算。

由于后面的 GPU 必须获得前面的 GPU 计算出的结果才能开始计算，**GPU 之间是串行计算的**。这种并行方法效率不高，主要胜在方便。在训练的时候也是同样的，前向传播的时候需要等待前面的计算结果；反向传播的时候则需要等待后面的计算结果。处于等待状态的 GPU 我们称之为「气泡」（bubble）。

在经验上来说，GPU 计算的时间是大头，因此气泡出现频率还是挺高的，这也意味着同一时间有大量 GPU 都在闲着，就像接力赛那样。这同样使得 PCIe（或者 NVLINK）的占用率较低。因此流水线并行经常出现的情况就是 GPU 占用和 PCIe 占用都低。llama.cpp 就是典型的流水线并行推理。

要减少气泡，提高利用率，首要的做法就是**减小 batch size**。让每个 GPU 单轮计算的时间短才能尽可能调动整个链条。

不过因为方便，流水线并行可以很容易地拓展到数千张显卡的集群中去，还能将 CPU 也拉进来协同计算。不过这**要求模型必须是比较规则的层叠状，不能太奇怪**。如果模型内部有许多分支或是各种稀奇古怪的操作的话流水线并行就很难起作用。比如 T5 模型这样的 Encoder-Decoder 架构模型就有这种问题。

> [!note]
> **流水线并行非常适合进行大规模训练**。因为流水线并行完全不需要同步梯度，只需要将计算结果和误差在层之间传递即可。也就是说，**流水线并行模型训练时候的通信量和参数规模完全无关**。TP 和 DP 是做不到的。
# 张量并行
张量并行将一个矩阵乘法拆解到多个 GPU 上，让 GPU 能够并行工作，而不是像流水线并行那样 GPU 内部并行、GPU 之间串行。以 $A_{n\times l}B_{l\times m}$ 的计算为例：
$$
A_{n\times l}B_{l\times m} = A_{n\times l}\begin{bmatrix}
B_{l_{1}\times m}\\B_{l_{2}\times m} 
\end{bmatrix} = \begin{bmatrix}
AB_{1}\\AB_{2}
\end{bmatrix}
$$
于是，$AB_{1},AB_{2}$ 被分到两个 GPU 上同时计算；计算完毕后，结果被拼接（All-gather）起来成为最终结果。这种方式叫做**行并行**（我们按行切）。

另一种方法是把 $B$ 竖切，于是：
$$
A_{n\times l}B_{l\times m} = A_{n\times l}[B_{l\times m_{1}},B_{l\times m_{2}}]=[AB_{1},AB_{2}]
$$
计算完毕后一样 All-gather 拼出结果。这种方式叫做**列并行**。

如果 AB 都按中间维度切开，那么就可以用 AllReduce 来通信：
```
[M,K]@[K,N]
[M,K1][M,K2] @ [K1,N] = [M,K1]@[K1,N]+[M,K2][K2,N]
			   [K2,N] 
rank1: [M,K1][K1,N]
+ AllReduce
rank2: [M,K2][K2,N]
```

同理，也可以拆出 3 份、4 份进行张量并行，运行在多个 GPU 上。张量并行可以让多个 GPU 并行工作，都不闲着，效率极高。坏处是 GPU 之间必须频繁交换数据，会对 PCIe 造成较大压力，因此最好换 NVLink。

张量并行的坏处是对数据传输要求太高，PCIe 32GB/s 的带宽都扛不住，那相比之下带宽低、延迟高的网络更是扛不住。25GB/s 带宽的网络系统才能考虑张量并行多结点训练。
## Megatron-LM 的 TP
因为 Causal LLM 的 Block 通常都是三个矩阵相乘（QKV、xW1W2），所以有了一种更聪明的 TP 方法。以 QKV 为例，第一步先来**列并行**：
$$
QK=Q[K_{1},K_{2}]=[QK_{1},QK_{2}]
$$
然后并不在此时 All gather，而是完成剩下的标量计算后，继续把 V 的行也切开分到两个 GPU 上 ：
$$
[QK_{1},QK_{2}]\begin{bmatrix}
V_{1}\\V_{2}
\end{bmatrix}=QK_{1}V_{1}+QK_{2}V_{2}
$$
然后此时再用一次 all-reduce 合并结果即可。
```
Q: [b,s,n,h], K: [b,s+f,n,h], V: [b,s+f,n,h]

Step1
rank1: P1=Q1K1^T [b,n1,s,h]@[b,n1,h,s+f,] = [b,n1,s,s+f] 
rank2: P2=Q2K2^T [b,n2,s,h]@[b,n2,h,s+f,] = [b,n2,s,s+f] 

Step2
rank1: P1=softmax(P1), O1=P1V1 = [b,n1,s,s+f]@[b,n1,s+f,h] = [b,n1,s,h]
rank2: P2=softmax(P2), O2=P2V2 = [b,n2,s,s+f]@[b,n2,s+f,h] = [b,n2.s,h]

Step3
rank1: O1=O1@O_proj = [b,n1,s,h]@[n1*h,n*h] = [b,s,n*h]
rank2: O2=O2@O_proj = [b,n2,s,h]@[n2*h,n*h] = [b,s,n*h]

Step4
AllReduce O1+O2
```

# 数据并行
数据并行更简单粗暴：将一整个计算流程复制很多份分散到多个机器或者 GPU 上。于是推理问题就成了一个**负载均衡**问题。数据并行适合那种算力有相当程度冗余的豪放计算集群，多结点推理就经常用这种方式。

但是实际上数据并行也经常用于训练，其根本原理是梯度的**可加性**。

假设当前我们的模型是 $f(\mathbf{x};\theta)$，那么设我们有一个数据集 $D$，其包含一组数据点 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n$ 与对应的 ground truth 值是 $\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_n$；我们的损失函数是 $\mathcal{L}(\hat{\mathbf{y}},\mathbf{y})$。那么我们训练模型的本质其实是最小化模型在所有数据点上的期望损失：
$$
\min _{\theta} \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim D}\mathcal{L}(f(\mathbf{x};\theta),\mathbf{y}) = \min _{\theta}\frac{1}{|D|}\sum_{(\mathbf{x},\mathbf{y})\in D}\mathcal{L}(f(\mathbf{x};\theta),\mathbf{y})
$$
按照梯度下降法，要实现最小化，就需要计算 $\theta$ 的梯度。我们拿出经典的 chain rule（至于梯度怎么用那有很多办法）：
$$
\text{grad}=\nabla_{\theta}\left( \frac{1}{|D|}\sum_{(\mathbf{x},\mathbf{y})\in D}\mathcal{L}(f(\mathbf{x};\theta),\mathbf{y}) \right)=\frac{1}{|D|}\sum_{(\mathbf{x},\mathbf{y})\in D}\nabla_{f}\mathcal{L}\nabla _{\theta}f(\mathbf{x};\theta)
$$
现在，我们将 $D$ 划分为一组子数据集 $D_1, D_2, \dots, D_m$，分到 $m$ 个模型上分开训练。于是第 $i$ 个模型训练求得的梯度就是：
$$
\text{grad}_{i}= \frac{1}{|D_{i}|}\sum_{(\mathbf{x},\mathbf{y})\in D_{i}}\nabla_{f}\mathcal{L}\nabla _{\theta}f(\mathbf{x};\theta)
$$
我们试着把所有模型求出的梯度求个均值：
$$
\begin{array}l
\displaystyle\sum_{i=1}^m \frac{|D_{i}|}{|D|} \text{grad}_{i} & = & \displaystyle\frac{1}{|D|}\sum_{i=1}^m  \sum_{(\mathbf{x},\mathbf{y})\in D_{i}}\nabla_{f}\mathcal{L}\nabla _{\theta}f(\mathbf{x};\theta)\\ & = & \displaystyle\frac{1}{|D|}\sum_{(\mathbf{x},\mathbf{y})\in D}\nabla_{f}\mathcal{L}\nabla _{\theta}f(\mathbf{x};\theta) \\
 & = & \text{grad}
\end{array}
$$
可以发现，和所有数据在一个模型上训练出来的梯度在数学上是完全相等的。这告诉我们：**将一批数据分散到多台机器上的相同模型上训练是完全可行的**。这就是为什么数据并行会成为大规模分布式训练的主流方法。步骤：
1. 将数据分为若干个批次 $B_1, B_2, \dots, B_k$，每个批次按顺序训练
2. 假定当前处理第 $i$ 批数据 $B_{i}$：
	1. 假定数据并行 $m$ 个模型，那么将 $B_{i}$ 分割为 $m$ 个 mini batch，分到所有模型上训练
	2. 训练出一组梯度 $\text{grad}_1, \text{grad}_2, \dots, \text{grad}_m$
	3. 将梯度**广播**给所有数据并行模型，每个模型都获得所有梯度，同步更新，更新之后所有模型参数还是一样的
		1. 如果是 SGD（Stochastic GD），那就随机取一个 $\text{grad}_{l}$ 更新
3. 接着处理下一批数据

数据并行的方式很适合用 Kubernetes 来管理。

由于每个需要更新的参数都会计算出梯度，因此有多少参数参与训练，就有多少数字需要广播。一个 32B 的模型会需要广播 60+GB 的数据，给网络系统造成巨大的压力。如果是 NVlink （高达 600+ GB/s 带宽）或者 PCIe （128 GB/s）也许还能应对，对于宽带来说就太恐怖了。
## All Reduce
广播同步参数的过程我们就叫做 all reduce。以下面的网络拓扑为例，假设某个模型上计算得来的梯度大小是 $P$，数据并行为 $N$，那么每个模型都需要发送一次广播给所有模型， LAN1 需要承受 $P\times N$ 的通信总量。如果所有模型同时广播自己的梯度，那么交换机同一时间就要处理所有 $P\times N$ 的数据。

但是如果我们将其组成一个环线，相邻每对模型之间有一个网络，情况就会不一样：
我们可以这样做，分两个阶段：
1. Reduce Scatter
	1. 从第一个模型开始，接力向自己左手边传递自己的梯度；收到梯度的结点就将梯度**累加**到自己的梯度里，就这样走完一圈
	2. 走到最后一个模型的时候，最后一个模型就拥有完整的全部梯度了
2. All Gather
	1. 从最后一个模型开始，仍然向自己左手边传递自己的梯度，收到梯度的结点将收到的梯度**覆盖**自己的梯度

这样，每时每刻每个网络最多输送 $P$ 的数据，避免了 $P\times N$ 高峰。但是这样的做法仍然不够好，因为每时每刻只有一个网络处于忙碌状态，而且 $P$ 的数据还是太大。

为了解决这个问题，我们可以将梯度 $P$ 均分为 $N$ 份，每份大小是 $P/N$。然后将过程变为：
1. Reduce Scatter：所有模型同时发送数据，第 $i$ 个模型向下一个模型发送第 $i$ 份梯度，收到的梯度累加到对应部分，重复 $N$ 次；此时所有模型手中都有一份梯度是完整的
2. All Gather：所有模型同时发送数据，都发送完整的那份梯度；收到梯度的模型覆盖原有梯度。
