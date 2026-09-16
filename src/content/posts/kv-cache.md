---
title: "KV-Cache"
pubDate: 2026-09-17
description: "来自 only-notes 的原始笔记，保留原文内容并做网页格式适配。"
author: "Westwoods"
tags: [LLM, AI Infra, 推理优化]
draft: false
---

KV-Cache 是一种重要的大语言模型加速推理技术，正是因为有了它，注意力计算才这么高效。

我们考虑一个 Decoder Only 块处理一个提示词序列 `a,b,c,d,e`。如果不做任何优化，那计算过程是这样的：
```
# 计算QKV，这一步耗时O(n)
a,b,c,d,e ->
Qa, Qb, Qc, Qd, Qe
Ka, Kb, Kc, Kd, Ke
Va, Vb, Vc, Vd, Ve

# 计算注意力，假装我们有Flash Attention，忽略softmax，这里耗时O(1+...+n)=O(n^2)
Aa=(Qa*Ka^T)*Va
Ab=(Qb*Ka^T)*Va+(Qb*Kb^T)*Vb
Ac=(Qc*Ka^T)*Va+(Qc*Kb^T)*Vb+(Qc*Kc^T)*Vc
...
-> Aa,Ab,Ac,Ad,Ae

# 计算残差、Norm、FFN、残差、Norm，步骤不少，但是合起来是O(n)
Aa,Ab,Ac,Ad,Ae ->
a',b',c',d',e'

# 得到新token
f <- Linear & Norm (e')
```
反正最后，如果真一点都不优化，第一个 token 的 kv 计算 n 次，处理完序列、计算第一个 token 需要 $O(n^{2})$ 时间，计算整个序列需要 $O(n^{3})$ 时间。忍不了，根本忍不了。

为此，我们决定采纳动态规划的思想，空间换时间。我们把：
```
Ka, Kb, Kc, Kd, Ke
Va, Vb, Vc, Vd, Ve
```
都存起来，那么计算注意力环节就只剩下了：
```
f -> Qf, Kf, Vf
Af=(Qf*Ka^T)*Va+(Qf*Kb^T)*Vb+(Qf*Kc^T)*Vc+...+(Qf*kf^T)*Vf
```
只有 $O(n)$ 次注意力计算，效率提高了。而由于矩阵 GPU 加速的特性，这一个 $O(n)$ 的运算因为并行的存在，随着可能随着时间增长并不明显。

而且：
1. FFN 和残差结构是逐个 token 计算的，不需要之前的 token
2. 得益于 Layer Norm 可以「各自算各自」的特点，也不需要之前的 token
3. 最后 Linear 层计算 logit 仍然不需要之前的 token，只拿计算出来的一个 token 向量

所以只需要存储 KV，prefill 之后 Q 可以直接丢弃。就可以实现接近 $O(1)$ 的 token 预测。这就是 KV-缓存的意义。
# 计算
kv cache 的大小是可以预估的。我们一般按照 model card 的信息就能估算。我们重点关注：
1. head_length：如果只给 hidden_size，记得 Transformer 的的 hidden_vec 是 concat 的所有 heads，所以 hidden_size / Q 的数量即可得到 head_length
2. num_heads：有多少 KV 头，如果是经典 MHA，那么 Q 有多少它们也对应有多少；但是如今的 GQA 和 MQA 会让多个 Q 对应一组 KV；比如 Qwen 2.5 32B 就写着 40 / 8，意思是 Q 有 40 个，KV 各 8 个；如果是 MQA 那么 KV 就只有各一个了
3. batch_size
4. seq_len：主要考虑实际上有多长
5. layers
6. 数据格式 byte，比如我们量化了 fp8 kv-cache 那就是 8bit/1byte 了

最后：`num_heads*head_length*layers*seq_len*batch_size*byte`，注意 num_heads 要 kv 分开算。举个例子：
> 我们使用Qwen2.5 32B，这个模型实际参数量是32.5B，经过8bit量化之后占据显存32.5GB；属于GQA架构，40/8，64层，hiddensize 是 5120。我使用的配置是fp8 kv cache，双4090 PCIe，TP方法，vllm

1. head_len 不知道，按照 Transformer 结构是 5120/40=128
2. GQA 架构，8 个 kv 对，这里 16 个 heads
3. 那么一个 token 就对应 `16*128*64*1byte = 2*1k*64byte = 128KB`
4. 这时候再去考虑 batch_size 和 seq_len 的影响就好多了
5. 此外注意，TP 是把 KVcache 也分割了，毕竟是矩阵乘法，所以不会额外增加显存消耗
# 收费机制解析
按照上述说法，收费机制会变得难以理解：为什么输入需要消耗 $O(n^{2})$ 时间，但是会比输出便宜？

其实这是因为 [[Flash Attention]] 的存在，大矩阵乘法优化使得这一过程可以简化为两次矩阵相乘，这是一个 Compute Bound 的问题，GPU 擅长这个。而输出则严格遵守串行，不能优化，还必须不断把 KV-cache 搬进片上，这是个 IO Bound 任务。

此外，收费项中的缓存不是 KV 缓存，而是**上下文缓存**。即某些输入 prompt 的 KV 会以**前缀树**的形式存起来，避免重复计算。

| 收费项               | 价钱   |
| ----------------- | ---- |
| 百万tokens输入（缓存命中）  | 0.5元 |
| 百万tokens输入（缓存未命中） | 4元   |
| 百万tokens输出        | 12元  |
