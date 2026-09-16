---
title: "Flash Attention"
pubDate: 2026-09-17
description: "来自 only-notes 的原始笔记，保留原文内容并做网页格式适配。"
author: "Westwoods"
tags: [AI Infra, Attention]
draft: false
---

在 LLM 处理输入的 prefill 阶段，我们往往直接计算缩放点积注意力 Scaled Dot-Product Attention。假设输入是 $n\times d$ 的向量序列，那么就有：
1. $S_{n\times n}=QK^T$
2. $A_{n\times n}=Softmax(S)$
3. $O_{n\times d}=AV$

其中产生的两个中间变量 $A,S$ 都是大矩阵，当用户输入一个 128k 长度的输入的时候这么大的矩阵会出现很多的困难：
1. GEMM 分块加载执行乘法以及  element_wise 操作，数据需要从显存读写 3 轮
2. 储存 `nxn` 大矩阵

因此我们用Flash Attention：
1. 不再显式储存A、S，直接输出 O，减轻存储压力
2. 通过在线计算Softmax 减少读写变得更快
3. 数学上仍然和经典算法等价
# 原版Attention
我们要按照序列这个维度（n）切块，然后轮流处理：
1. $Q_{n\times d}=Q_{1,m\times d},Q_{2,m\times d},\dots$，就这样切成最多 m 个 token 一个的小块，QKV 都这么做
2. 加载第一个 Q 块，然后轮询完所有的 KV 块，用在线 Softmax 合并，写回 HBM
3. 加载第二个 Q 块，轮询完所有 KV 块，用在线 softmax 合并，写回 HBM
4. ……

这样中间变量就被避免了，但是能这样干这都有赖于在线 Softmax。按照原本的 softmax，必须得到了 S 完全体才能计算 A。

具体来说是这样做的。

第 i 个 token 在计算其对第j个token的注意力是这样的：
$$
s_{ij}=\frac{\mathbf{q}_{i}\mathbf{k}_{j}^T}{\sqrt{ d }}
$$
然后和所有的j（在Causal LM里是 i 之前的token）都算完：
$$
\mathbf{s}_{i}=[s_{i1}, \dots, s_{ii}]
$$
经典算法中我们现在该对这行进行softmax，计算这个分母需要收集全部的 $\mathbf{s}$并且求一个和：
$$
a_{ij}=\frac{\exp(s_{ij}-m_{i})}{\sum _{k=1}^i \exp(s_{ik}-m_{i})}
$$
其中 $m_{i}=\max_{k}s_{ik}$ 是行最大值。这么做是基于一个算法创新，用于提升数值稳定性。然后计算输出：
$$
\mathbf{o}_{i}=\sum_{j=1}^ia_{ij}\mathbf{v}_{j}
$$
# 优化
这个计算可以分为两个部分，一个是分母，一个是分子，我们可以把这两个分开算：
$$
\begin{cases}
\mathbf{r}_{i}=\sum_{j=1}^i \exp(s_{ij}-m_{i})\mathbf{v}_{j} \\
\mathbf{l}_{i}=\sum _{j=1}^i\exp(s_{ij}-m_{i}) \\
\mathbf{o}_{i}=\mathbf{r}_{i}/\mathbf{l}_{i}
\end{cases}
$$
在这里，如果我们按照 $i=1,2,3,\dots$ 这样的顺序逐个计算 $\mathbf{o}_{i}$ 。以分子计算为例，分母同理：
$$
\begin{array}l 
\mathbf{r}_{1}=\exp(s_{11}-m_{1})\mathbf{v}_{1} \\
\mathbf{r}_{2}=\exp(s_{21}-m_{2})\mathbf{v}_{1}+\exp(s_{22}-m_{2})\mathbf{v}_{2} \\
\dots \\
\mathbf{r}_{i}=\exp(s_{i1}-m_{i})\mathbf{v}_{1}+\exp(s_{i2}-m_{i})\mathbf{v}_{2} +\dots+\exp(s_{ii}-m_{i})\mathbf{v}_{i}
\end{array}
$$
我们发现我们其实储存一行 $s_{ij}$ 即可，之前的 $s_{<i,j}$ 可以通通被覆盖。因此我们不需要 $O(n^{2})$ 空间存放那么多东西，只需要 $O(n)$ 即可。但是这个方案空间还是不够好，而且每次计算一个 $\mathbf{r}_{i}$ 伴随 $O(n)$ 次对显存的的读写。这还不够。

这里面最核心的问题是： **$m_{i}=\max_{j}\{ s_{ij} \}$，它必须等待这行 $s_{ij}$ 都计算完毕才能得到，$O(n)$ 存储减不下去**。这就是 Flash Attention 的优化点了。
## Flash Attention
我们先思考给它分块计算：
$$
\begin{cases}
\mathbf{r}_{i} = \sum_{j\in B_{1}}\exp(s_{ij}-m_{iB_{1}})\mathbf{v}_{j} \\
\mathbf{l}_{i}= \sum_{j\in B_{1}}\exp(s_{ij}-m_{iB_{1}})
\end{cases}
$$
这样 $m_{iB_{1}}$ 只是局部最大值，显然不行。于是 Flash Attention 用一种方法**在线更新它**。假设 $\mathbf{r}_{i} = \exp(s_{iB}-m_{iB})\mathbf{v}_{B}$ 表示从开始到 B 块的所有式子加起来的结果，$m_{iB}$ 是这段范围里的 $s$ 最大值。现在我们试图合并第 $B'$ 块的结果，于是我们现在有：
$$
\begin{array}l
\mathbf{r}_{i} = \exp(s_{iB}-m_{iB})\mathbf{v}_{B} \\
\mathbf{r}_{i}' = \exp(s_{iB'}-m_{iB'})\mathbf{v}_{B'} 
\end{array}
$$
我们现在有 $m_{iB},m_{iB'}$，那么新的 $m=\max\{ m_{iB}, m_{iB'} \}$，这个是容易得到的。我们把做一个放缩：
$$
\mathbf{r}_{i} = \exp(s_{iB}-m)\mathbf{v}_{B}\exp(m-m_{iB})
$$
于是我们发现：
$$
\mathbf{r}_{i}=\exp(s_{iB}-m)=\mathbf{r}_{i}\exp(m_{iB}-m)
$$
诶这正好是我们要的 m 更新版。同样的：
$$
l_{i}=\exp(s_{iB}-m)=l_{i}\exp(m_{iB}-m)
$$
此时我们就可以合并结果了（都是更新 m 后的结果）：
$$
\begin{array}l
\mathbf{r}_{i}^{new}=\mathbf{r}_{i}+\mathbf{r}_{i}' \\
l_{i}^{new} = l_{i}+l_{i}'
\end{array}
$$
于是我们就实现了分块+在线更新 Softmax。我们在合并过程中仅需维护这几个变量（一个旧一个新）：
$$
\begin{cases}
\mathbf{r}_{i}\times2 \\
l_{i}\times2 \\
m_{i}\times2
\end{cases}
$$
其中，他们的 shape 分别是 `[attn_batch, head_dim]` 和 `[attn_batch, 1]`。计算单个块的时候则需要 `[attn_batch, block_size, head_dim]` 的 `QKV` 矩阵。存储消耗可控了起来。

此外读写 HBM 的次数也变成了 `seq / block_size` 级别。效率大大提高了。
