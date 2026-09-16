---
title: "FlashAttention：注意力计算为什么能更快"
pubDate: 2026-09-17
description: "从注意力的显存瓶颈出发，理解 FlashAttention 如何通过分块和在线 Softmax 减少 HBM 读写。"
author: "Westwoods"
tags: [AI Infra, Attention, GPU]
draft: false
---

标准缩放点积注意力可以写成：

$$
S=QK^T,\qquad A=\operatorname{softmax}(S),\qquad O=AV
$$

序列长度为 $n$ 时，$S$ 和 $A$ 都是 $n\times n$ 矩阵。长上下文下，真正昂贵的不只是矩阵乘法，还包括这些中间结果在 GPU 高带宽显存（HBM）和片上 SRAM 之间的反复读写。

## 关键思路：不保存完整注意力矩阵

FlashAttention 不改变注意力的数学结果，而是改变计算顺序。它把 Q、K、V 按序列维度切成 block：加载一个 Q block，再逐块遍历 K、V，边计算边更新输出。这样无需把完整的 $S$ 和 $A$ 写回 HBM。

核心依赖是在线 Softmax。对一行分数，维护当前最大值 $m$、指数和 $\ell$ 以及加权输出 $o$：

$$
m_{new}=\max(m_{old},m_{block})
$$

$$
\ell_{new}=e^{m_{old}-m_{new}}\ell_{old}+e^{m_{block}-m_{new}}\ell_{block}
$$

输出也用同样的缩放因子合并。这样即使分数被分成多个 block，最后得到的结果仍然等价于对完整行做一次稳定 Softmax。

## 为什么更快

传统实现往往需要多次读写 $n\times n$ 的中间矩阵。FlashAttention 尽量让一个 block 在片上存储中完成更多操作，只把必要的结果写回 HBM，因此减少了内存流量，也减少了 kernel launch 和中间张量的分配。

这是一类典型的 IO-aware 优化：瓶颈不一定是 FLOPs，而可能是数据搬运。对长上下文尤其如此。

## 需要注意的边界

FlashAttention 并没有把注意力的理论复杂度从 $O(n^2)$ 变成线性；它主要降低了内存占用和实际运行时间。训练和 prefill 阶段收益通常很明显，而 decode 阶段的主要瓶颈还可能来自 KV-Cache 读取、batch 调度和 kernel 利用率。

因此，评估优化效果时应同时看显存峰值、prefill 吞吐、decode 延迟和不同序列长度下的表现，而不能只看单个 benchmark 数字。
