---
title: "混合专家架构"
pubDate: 2026-09-17
description: "来自 only-notes 的原始笔记，保留原文内容并做网页格式适配。"
author: "Westwoods"
tags: [LLM, 模型架构]
draft: false
---

混合专家架构（Mixture of Experts，MoE）是一种节能架构，其主要目标是减少参与计算的参数量从而实现训练和推理的降本增效。具体来说，MoE 架构会用一个门控网络充当路由器（router），只将输入路由到某一部分专家模型而非所有参数实现降本增效。

网上讲 MoE 的文章很多，但是我觉得都不触及本质。本文采纳我觉得最好的一种说法，也就是苏剑林的讲述[MoE环游记：1、从几何意义出发 - 科学空间|Scientific Spaces](https://mexue.fm/archives/10699)。
# MoE 路由
首先明确一点：相同参数规模的 MoE 模型和传统 dense 模型做比较的话，**后者**是占优势的。如果你做数据处理的时候询问 ChatGPT 是选用 Dense 模型还是相同参数的 MoE 模型的话，它的答案会是后者。原因后面会说。

大语言模型 MoE 的首要目的是为了**降本**，这意味着它的地位如同仿制药，目标是逼近原版的 Dense 模型而不是超越它。Decoder Only 架构的大语言模型中有个 FFN 层计算量很大。这个 FFN 层可以看作两个线性层中间夹着个激活函数：
$$
\text{FFN}(\mathbf{x})=f(\mathbf{x}W_{a})W_{b}
$$
将 $W_{a}$ 按列分块分为 $m$ 块（不需要均匀）$[W_{a1}, W_{a2}, \dots, W_{am}]$；$W_{b}$ 按同样的方式按行分为 $m$ 块 $[W_{b1}, W_{b2}, \dots, W_{bm}]^T$。于是 FFN 层变成了：
$$
\text{FFN}(\mathbf{x})=\sum_{i=1}^mf(\mathbf{x}W_{ai})W_{bi}=\sum_{i=1}^m\mathbf{v}_{i}
$$
于是 Dense 模型计算的结果就成了一系列向量 $\mathbf{v}_{i}$ 的和。**这每个向量就被视作一个专家计算的结果**。

我们的目标是：**从这 m 个向量中选出最合适的 k 个向量，使得其能够最大程度上逼近所有向量的和**。于是这就成了一个约束优化问题。
$$
\min _{\lambda_1, \lambda_2, \dots, \lambda_m \in \{ 0,1 \} }\begin{Vmatrix}
\displaystyle\sum_{i=1}^m\mathbf{v}_{i}\lambda_{i}-\sum_{i=1}^m\mathbf{v}_{i}
\end{Vmatrix}^{2}, \text{s.t.}\sum_{i=1}^m\lambda_{i}=k
$$
待优化的式子可以化简一下：
$$
\begin{Vmatrix}
\displaystyle\sum_{i=1}^m(\lambda_{i}-1)\mathbf{v}_{i}
\end{Vmatrix}^{2}=\left( \sum_{i=1}^m(\lambda_{i}-1)\mathbf{v}_{i} \right)\left( \sum_{i=1}^m(\lambda_{i}-1)\mathbf{v}_{i} \right)^T
$$
然后我就发现推理走到头了……解析解怕是求不出来。于是我们引入一个假设：假设我们的模型真的没有一点信息冗余，每个专家输出的向量都是各有作用、无可替代，那么这样的话， $\mathbf{v}$ 必然**两两正交**。如果正交，那么就意味着互相的点积是 0。于是原式子变为：
$$
\sum_{i=1}^m
(\lambda_{i}-1)^{2}\mathbf{v}_{i}\mathbf{v}_{i}^T
$$
为了将这个式子最小化，我们就要让模长最长的那几个向量的 $\lambda_{i}-1=0$。于是，这些向量的 $\lambda_{i}=1$。因此**选取模长最大的 k 个向量可以最大程度上逼近 m 个向量的和**。

那么问题又来了：我们的目标是要降本。向量的模长必须计算出向量之后才知道，这根本起不到降本的作用。于是我们决定：用一个小模型**预测所有向量的模长**，然后我们只计算预测模长前 k 的向量就好。如果这个模型是一个线性层+激活函数的话，它就是这样的。
$$
\boldsymbol{\rho}=h(\mathbf{x}W_{r})=[\rho_1, \rho_2, \dots, \rho_m ]
$$
最终我们只取前 $k$ 个 $\rho$ 所表示的向量进行计算，降本增效就实现了。这里的小模型就是所谓的**router**，负责决定哪些专家参与计算。

那么接下来的问题就是：如何得到这个 router？
# 负载均衡
既然有路由，那就有负载均衡问题。如果 router 总是把 token 路由到某几个专家，那么其他专家就会缺乏训练，成为累赘，这叫做 dead expert。于是我们需要想个办法，训练这个 router 负载均衡。

router 是不能拿出来单独训练的。我们一般是给主损失函数加一个辅助损失项 auxiliary loss（aux loss），然后放在一起训练。aux loss 主要是表示**模型选择专家的均衡程度**。假设 $\boldsymbol{\rho}$ 是 router 预测的值，其是一个分布；而取 topk 之后的结果是 $\mathbf{f}$（选了第 i 个专家，$f_{i}=1/k$；否则 $f_{i}=0$）。一个 batch 中所有的 token 计算之后，我们得到了 $\boldsymbol{\rho}$ 的均值是 $\mathbf{P}$；$\mathbf{f}$ 的均值是 $\mathbf{F}$，那么辅助损失 aux loss 就是：
$$
\mathcal{L}_{\text{aux}}=\mathbf{F}\cdot \mathbf{P}
$$
