---
title: "KV-Cache：大模型自回归推理的空间换时间"
pubDate: 2026-09-17
description: "解释 KV-Cache 如何避免重复计算，以及它为什么会成为长上下文 decode 阶段的显存和带宽瓶颈。"
author: "Westwoods"
tags: [LLM, 推理优化, AI Infra]
draft: false
---

自回归模型每生成一个 token，都要把它接到已有序列后面。注意力层会读取历史 token 的 Key 和 Value。如果每一步都重新计算整段历史，计算量会快速增长。

## 不使用缓存会发生什么

假设提示词是 `a b c d e`。生成 `f` 时，模型需要计算 `e` 对 `a...e` 的注意力；生成 `g` 时，如果没有缓存，又要重新计算 `a...f` 的 K、V。历史 token 的 K、V 实际上没有变化，却被反复计算。

## 缓存什么

在每一层注意力中，将已经计算过的 K 和 V 保存下来。下一步只计算新 token 的 Q、K、V，然后让新 Q 与缓存中的全部 K 做注意力，再用对应的 V 得到输出：

$$
O_t=\operatorname{softmax}\left(\frac{Q_tK_{1:t}^{T}}{\sqrt d}\right)V_{1:t}
$$

其中 $K_{1:t}$ 和 $V_{1:t}$ 包含历史缓存以及当前 token。

KV-Cache 把重复的历史计算换成了显存空间。对标准多头注意力，缓存大小大致与层数、序列长度、batch、KV 头数和 head dimension 成正比。上下文越长、并发越高，缓存就越容易超过权重本身成为主要显存占用。

## Prefill 与 Decode

Prefill 处理用户输入，通常可以并行计算整个序列，瓶颈更接近矩阵乘法和显存容量。Decode 一次生成一个 token，计算规模变小，但每一步都要读取历史 KV，因此经常受显存带宽和访存效率限制。

这也是为什么推理系统会使用分页 KV-Cache、连续 batching、KV 量化，以及 MQA/GQA/MLA 等技术。它们分别从内存管理、调度、数据精度和缓存结构上降低成本。

## 工程上的取舍

KV-Cache 并不是越大越好。服务端需要设置最大上下文长度，并在请求结束后及时回收缓存；动态 batching 还要处理不同请求的长度差异。实际系统中，吞吐、首 token 延迟、单 token 延迟和缓存命中率需要一起观察。
