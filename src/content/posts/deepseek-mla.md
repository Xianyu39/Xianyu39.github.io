---
title: "DeepSeek MLA：从 KV-Cache 压缩到低带宽注意力"
pubDate: 2026-09-17
description: "梳理 Multi-head Latent Attention 的基本动机，并说明它如何用低维 latent cache 减少 KV-Cache 的存储与读取。"
author: "Westwoods"
tags: [LLM, AI Infra, KV-Cache, Attention]
draft: false
---

在长上下文和高并发推理中，KV-Cache 不仅占用大量显存，还会在每个 decode step 被反复读取。Multi-head Latent Attention（MLA）的核心目标，是只缓存一个更紧凑的 latent 表示，在需要时恢复注意力所需的信息。

## 从完整 KV 到 latent cache

标准注意力会从输入 $X$ 直接生成：

$$K=XW_K,\qquad V=XW_V$$

MLA 先把输入压缩到低维空间：

$$C=XW_{DKV}$$

然后通过上投影矩阵得到 K、V：

$$K=CW_{UK},\qquad V=CW_{UV}$$

推理时缓存的是 $C$，而不是完整的 K、V。只要 latent 维度远小于所有 KV 头的总维度，缓存体积就可以显著下降。

## 为什么不能简单地“先恢复再计算”

如果每一步都把 $C$ 完整恢复成 K、V，再执行普通注意力，很多节省会被重新读写抵消。因此 MLA 会尝试把投影吸收到注意力计算中。例如：

$$QK^T=Q(CW_{UK})^T=(QW_{UK}^T)C^T$$

Value 路径也可以将 $C$ 与注意力权重先计算，再乘以 $W_{UV}$。这使得计算围绕 latent 表示展开，避免生成完整 KV 张量。

## RoPE 与工程细节

位置编码会影响哪些量能够被压缩。MLA 的实际设计会把需要 RoPE 的部分和不需要 RoPE 的部分拆开处理，不能把所有公式简单套成一个低秩投影。部署时还要考虑 kernel 融合、缓存布局、量化精度和不同 batch 长度下的访存模式。

MLA 的价值不只是“参数更少”，而是把 decode 阶段的主要数据流从完整 KV 转移到更小的 latent cache。最终收益要用端到端指标验证：显存占用、单 token 延迟、吞吐和长上下文稳定性缺一不可。
