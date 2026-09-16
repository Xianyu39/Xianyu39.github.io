---
title: "DeepSeek MLA"
pubDate: 2026-09-17
description: "来自 only-notes 的原始笔记，保留原文内容并做网页格式适配。"
author: "Westwoods"
tags: [LLM, AI Infra, KV-Cache]
draft: false
---

MLA 是继承 Attention GQA 之后的又一个 KV-Cache 优化技术。众所周知，KV-Cache 不仅占显存，还必须在每层、每个 decode 步都全量读取，这造成了很严重的 decode 瓶颈。
# Non-RoPE (nope)
MLA 的基本思路是 LoRA。原本的流程是：
```
输入向量 X
Q = XW_q
K = XW_k
V = XW_v

Softmax(QK^T/sqrt(d))V
```

这里我们先把 X 投影为一个更小的向量 `c`，然后用这个向量重建 `KV` 。
```
C = XW_{DKV} # c的size远小于x，我们只把C存在显存中，我们用两个矩阵重建KV
K = CW_{UK}
V = CW_{UV}
```

只是这样的话，KV 展开之后还是需要占据巨大的空间，而且制造更多读写。因此我们直接融合 Attention 操作：
```
P = QK^T = Q(CW_{UK})^T = (QW_{UK}^T) C^T
P = Softmax(P)
O = PV = (PC) W_{UV}
```

我们来按照维度进行推理。假设 decode 的时候 X 的形状是 `[B, 1, N, D]`，那么 QKV 就都得是 `[B, S, N, D]`。此外有 `W_q, W_k, W_v = [N, D, D]` ，`C = [B, S, R]` 那么我们来推理全过程的读写量：
1. 读 Q 和 KV：`BND + 2*BSND`
2. 写 O：`BND`

优化后：
1. 读 Q、C、`W_{UK}`、`W_{UV}`：`BND + BSR + 2NDR` （假设 Flash Attention 避免了 C 被读两次）
2. 写 O：`BND`

乘项变成了加项，Memory 大大优化了。
# With-RoPE
MLA 的核心是利用低秩压缩 KV 成 c 然后避免 c 展开成 KV 实现的，即 `(QW_{UK}^T) C^T` 。但是如果我们需要给 KV 加上 RoPE，那就意味着需要计算 `RoPE(K)`，又得展开 K，属于是完蛋了。

在这里，我们并没有数学等价的方法来把 RoPE 加回去，只能想办法加上一部分位置编码信息。于是 `head_dim` 被延长了一小部分用来承载位置信息。这样的话 rope 和 nope 就成了两个并行的路线：
```python
bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = self.q_norm(self.wq_a(x)) #降秩后做归一化 Input Tensor Shape:[b, s, 7168] -> Output Tensor Shape:[b, s，1536] 
        q = self.wq_b(qr) #升秩 Input Tensor Shape:[b, s，1536] -> Output Tensor Shape:[b, s，128*(128+64)]
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim) #Hidden Dim拆成Head Num，Head Dim两个维度 Input Tensor Shape:[b, s，128*(128+64)] -> Output Tensor Shape:[b, s，128, (128+64)]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)# 最后一维拆成128与64两个Tensor， Input Tensor Shape:[b, s，128*(128+64)] -> Output Tensor Shape:[b, s，128, 128], [b, s，128, 64]
        q_pe = apply_rotary_emb(q_pe, freqs_cis) #针对上述的第二个Tensor添加位置信息 Input Tensor Shape:[b, s，128, 64] -> Output Tensor Shape:[b, s，128, 64]
        kv = self.wkv_a(x) #KV的降秩矩阵乘 Input Tensor Shape:[b, s，7168] -> Output Tensor Shape:[b, s，　(512+64)]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) #最后一维拆成512与64两个Tensor，Input Tensor Shape:[b, s，(512+64)] -> Output Tensor Shape:[b, s，512], [b, s，64]
        kv = self.kv_norm(kv) #上述第一个Tensor做归一化 Input Tensor Shape:[b, s，512] -> Output Tensor Shape:[b, s，512]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) #上述第二个Tensor添加位置信息，Input Tensor Shape:[b, s，64] -> Output Tensor Shape:[b, s，64]
        self.kv_cache[:bsz, start_pos:end_pos] = kv   # 存KVCache
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2) # 存KVCache
        if mask is not None:    #MHA prefill: Prefill阶段采取的是Naive方案推理，Attention的计算逻辑是MHA
            q = torch.cat([q_nope, q_pe], dim=-1) #合并得到最终的Q Input Tensor Shape:[b, s，128, 128], [b, s，128, 64] -> Output Tensor Shape:[b, s，128, 128+64]
            kv = self.wkv_b(kv) #降秩的KV再做升维 Input Tensor Shape:[b, s，512] -> Output Tensor Shape:[b, s， 128*(128+128)]
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim) #Hidden Dim拆成Head Num，Head Dim两个维度, Input Tensor Shape:[b, s， 128*(128+128)] -> Output Tensor Shape:[b, s，128, (128+128)]
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1) #最后一维拆成128与128两个Tensor，Input Tensor Shape:[b, s，128, (128+128)] ->  Output Tensor Shape:[b, s，128, 128], [b, s，128, 128]
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1) #合并得到最终的K Input Tensor Shape:[b, s，128, 128], [b, s，64] -> Output Tensor Shape:[b, s，128, (128+64)]
            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale #Q*K/sqrt(192),  Input Tensor Shape:[b, s，128, 128+64] * [b, s，128, (128+64)] -> Output Tensor Shape:[b, s，128, s]

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask) # 调用indexer计算得到与当前Token相关性最高的2048个Token
            index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0) # 根据选出的2048个Token Index生成Mask，在Attention SoftMax计算时选中2048个Token参与计算
            index_mask += mask
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32) #Attention中的SoftMax计算， Input Tensor Shape:[b, s，128, s] -> Output Tensor Shape:[b, s，128, s]
            x = torch.einsum("bsht,bthd->bshd", scores.type_as(x), v) #再乘以V Input Tensor Shape:[b, s，128, s] -> Output Tensor Shape:[b, s，128, 128]
        else:                   # MQA decode: Decode阶段采用Absorb方案，将wkv_b矩阵乘拆成两份，分别串在了wq_b后面，与，wo前面。这样操作后，Attention模块就是MQA。
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank) # [128, 128+128, 512]
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim]) #wkv_b矩阵乘拆分后第一个矩阵乘，Input Tensor Shape:[b, s，128, 128] * [128, 128+128, 512] -> Output Tensor Shape:[b, s，128, 512]
            scores = (torch.einsum("bshc,btc->bsht", q_nope.float(), self.kv_cache[:bsz, :end_pos].float()) +
                      torch.einsum("bshr,btr->bsht", q_pe.float(), self.pe_cache[:bsz, :end_pos].float())) * self.softmax_scale
            # torch.einsum("bshc,btc->bsht", q_nope.float(), self.kv_cache[:bsz, :end_pos].float()) : Input Tensor Shape:[b, s，128, 512] * [b，s, 512] -> Output Tensor Shape:[b, s，128, s]
            # torch.einsum("bshr,btr->bsht", q_pe.float(), self.pe_cache[:bsz, :end_pos].float()) : Input Tensor Shape:[b, s，128, 64] * [b，s, 64] -> Output Tensor Shape:[b, s，128, s]
            # Q*K/sqrt(192) 

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask) # 调用indexer计算得到与当前Token相关性最高的2048个Token
            index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0) # 根据选出的2048个Token Index生成Mask，在Attention SoftMax计算时选中2048个Token参与计算
            index_mask += mask
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32)  # [b, s，128, s] -> [b, s，128, s]
            x = torch.einsum("bsht,btc->bshc", scores.type_as(x), self.kv_cache[:bsz, :end_pos])  #Attention的乘以V，Input Tensor Shape:[b, s，128, s] * [b，s, 512] -> Output Tensor Shape:[b, s，128, 512]
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:]) # [b, s，128, 512] * [128, 128, 512] -> [b, s，128, 128]
        x = self.wo(x.flatten(2)) #wkv_b矩阵乘拆分后第二个矩阵乘， Input Tensor Shape:[b, s，128*128] * [128*128, 7168] -> Output Tensor Shape:[b, s， 7168]
        return x
```

# TensorRT-LLM 如何实现 MLA
trtllm 的 pytorch backend 中的 MLA 类直接实现了带 DSA 的 MLA。它主要接受一个 `hidden_states` 和一个 `latent_cache_gen`。其中不包含 DSA 的路径大致按照如下方式计算：
1. 投影： `q,compressed_kv,k_pe = kv_a_proj_with_mqa(hidden_states).split([self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim])`
	1. 这部分计算输入的 q、latent 和 k_pe；它其实把这几个操作融合为了一个 linear 层：
		1. `q = hidden_states @ W_q_a` （这个是低秩表示 `qr`，同时包含 rope 和 nope 部分）
		2. `kv = hidden_states @ W_kv_a`（降维到 `kv_lora_rank+kv_rope_head_dim`）
		3. `compressed_kv, kv_pe = kv.split([kv_lora_rank+kv_rope_head_dim])`（分出 nope 和 rope ）
2. 统一并行加 norm：
	1. `q = RMSNorm(q)`
	2. `compressed_kv = RMSNorm(compressed_kv)`（按照算法， `k_pe` 不走 norm）
3. 统一并行做 `q` 升维投影和拼接 `compressed_kv` 和 `k_pe`：
	1. `q = q @ W_q_b`
	2. `latent_cache = concat([compressed_kv, k_pe])`
4. 按照 prefill 和 decode 的情况分开处理：
	1. 如果有 context token，需要做 prefill：
		1. 把 ctx token 都切出来
		2. 若前面的算子并没有融合 rope，那就为 `k_pe` 和 `q` 的 pe 部分做一下 rope
		3. 如果当前是 flashinfer 后端、MHA，则会做不同的处理：
			1. 默认的处理：
				1. 不做吸收，会再升维展开 `k_nope,v = (compressed_kv@W_kv_b).split()`；但是好在这不是 kv-cache，所以不吸收不会有太大影响，而且可以复用已有的算子
				2. 复用经典 mha 算子，计算 qkv 返回注意力（这条路不涉及 DSA）
			2. flashinfer 且enable_context_mla_with_cached_kv 且num_ctx_cached_tokens>0 （prefill 但是有 kvcache 可用），**这是唯一走吸收算法的路线**
				1. nope 路径执行经典吸收算法 `q_nope@W_k_b^T`，后面的部分视作一个正常 Attention。这里按照不同精度走两个 `bmm` 算子（一个头一次 mm，所以是 bmm）：
					1. `bf16`：`_bmm_bf16_out`
					2. `float8_e4m3fn`：`fp8_block_scaling_bmm_out`
				2. 然后给 pe 部分加 RoPE，此时的 `fused_q = [q_nope, q_pe]` 就能直接走经典 attention 算子按照计算 QKV 的方式。只不过这里的 QKV 含义不同
				3. 最后补上 `WUV`，同样是按头用 bmm 计算
			3. 如果要走 TrtllmAttention 的实现，就会针对 chunked_prefill 是否打开、是否是 Blackwell 以及之后的架构（SM 版本>=100）、是否有 kvcache 额外优化
				1. warmup
				2. chunked_prefill 且 blackwell：forward_context_with_chunked_prefill，不做吸收，分块读取计算，叠加 softmax，按照算法来看是直接展开了
				3. chunked_prefill 或有 kvcache：forward_context_with_cached_kv，一样不做吸收，会直接把压缩的 kvcache 展开走标准 Flash Attention
				4. 都不满足则 fallback 到默认
	2. 如果有 generation token，需要做 generation；和 prefill 不同这里它们都搞 absorption。除了 generation 需要标出更新后的各个序列在 token 和 kvcache 矩阵上的范围，其它和 prefill absorption 采取了类似的策略。

如果被指定走 DSA 路径，那么所有操作被打包为两个：
1. `forward_dsa_proj`：主要是批量把 q 和 latent 都基于 hidden 投影出来，并且使用 indexer 把 weights 和动态量化版的数据算出来
2. `forward_dsa_attn`：使用 Indexer 的函数接受上述数据，获得 indices 
	1. `sm_version>=100`： 会走吸收路线，针对 `bf16` 或 `fp8`（fp8 需要处理动态量化） 用对应的 BMM 算子，indices 连着 QKV 一起塞给 attention_backend 包办
	2. 否则，需要自己显式地 append kv、indices、调用 flashmla 来完成计算
