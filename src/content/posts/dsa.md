---
title: "DSA：稀疏注意力与推理优化"
pubDate: 2026-09-17
description: "讨论 DSA、Lightning Indexer、细粒度 token 选择及其在推理系统中的实现思路."
author: "Westwoods"
tags: [LLM, Attention, AI Infra]
draft: false
---

DSA 是 DeepSeek-V3.2-Exp 引入的一种机制，其改变了经典注意力机制中每个 token 需要和之前的所有 token 计算注意力的的形式，转而选取历史的 topk 个 token 参与注意力计算，进一步减少计算量和 IO。

论文把 DSA 原型分成两个部分：
1. **Lightning Indexer**
2. **Fine-grained Token Selection**

大致流程：
```
query hidden h_t
   ↓
lightning indexer 给所有历史 token 打分 I_{t,s}
   ↓
top-k selector 选出 k 个历史 KV entry
   ↓
主 attention 只在这 k 个 KV entry 上算
```
# Lightning Indexer
这是一个打分器，基于当前 token 的 hidden 给所有历史 token 打分。打分逻辑：
$$
I_{t,s} = \sum_{j=1}^H w_{t,j}\text{ReLU}(\mathbf{q}_{t,j}\cdot \mathbf{k}_{s})
$$
解释（以 DeepSeek v3.2 为例），假设当前 token 的 hidden 是 $\mathbf{x}_{t}$，长度 7168：
1. $I_{t,s}$ 是当前 token t 对历史 token s 的评分
2. indexer 一样是多头的，`index_n_heads: 64`，也就是这里的 $H$，每个头的大小是 `index_head_dim: 128`；但是这里不是所有头拼接，而是相加
3. $w_{t,j}$ 是由可学习矩阵 $W_{w}$ 动态生成的标量，每个头一个，一般直接矩阵乘计算出来： $\mathbf{w}_{t}=\mathbf{x}_{t}W_{w}$，`W_{w} [7168, 64]`
4. $\mathbf{q}_{t,j}=\text{RMSNorm}(\mathbf{x}_{t}W_{D})W_{q,j}$，这两个矩阵形状分别是 `[7168, 1536], [1536, 128]`，变换成一个 head 的长度
	1. 这里有个细节：$W_{D}$ 是一个矩阵，$W_{q,j}$ 是一个头一个，一种更常见的写法是分两步：
		1. $\mathbf{r}_{t}=\text{RMSNorm}(\mathbf{x}_{t}W_{D})\in \mathbb{R}^{1\times1536}$；$\mathbf{r}_{t}$ 其实就是 DeepSeek MLA 的低秩表示，可以直接复用
		2. $Q_{t}=\mathbf{r}_{t}W_{q}^{(64,1536,128)}\in \mathbb{R}^{64\times 128}$。其中，$\mathbf{q}_{t,j}$ 就是它的第 j 行，这里可以用 BMM 实现（64 个矩阵乘），也可以把形状变成 `[1536, 64*128]` 直接 GEMM，然后再 reshape 回来
5. $\mathbf{k}_{s}=\text{LayerNorm}(\mathbf{x}_{s}W_{k})$，形状也是 `[7168, 128]`，它是为所有 head 共享的

评分可以被描述为 q 的多头表示分别和历史 token 的 key 表示的点积的加权和；其中点积为负的头会被去掉。

这个过程也可以简写成：
$$
I_{t,s} = \mathbf{w}_{t}\text{ReLU}(Q_{t}\mathbf{k}_{s}^T)
$$
如果是批量计算，则把 K 聚集起来， $K=[\mathbf{k}_{1},\dots,\mathbf{k}_{n}]^T$，然后用 mask M 遮住后面的部分；$\mathbf{w}_{t}$ 则广播一个维度：
$$
I_{t} = \mathbf{w}_{t}\text{ReLU}(Q_{t}K^T)
$$
![](/blog-assets/DSA-1784029739071.png)
# Fine-grained Token Selection
数学上来说，这就是个简单的 `Topk`。但是我们要注意在 prefill 中， $t$ 后的 token 是不能参与排序的，需要用 mask 遮起来（源码这样写）：
$$
S_t
=
\operatorname{TopK}_{s \le t}
\left(
I_{t,s},
k
\right)
$$
其中 $S_t$ 是当前 token $t$ 选出的历史 token 下标集合，$k$ 是稀疏 attention 保留的 token 数。以 DeepSeek-V3.2 为例，`index_topk = 2048`。

如果写成 mask 形式，可以表示为：
$$
M_{t,s}
=
\begin{cases}
0, & s \in S_t \\
-\infty, & s \notin S_t
\end{cases}
$$
主 attention 会在原始 DeepSeek MLA attention scores 上加这个 mask：

$$
\mathrm{score}^{sparse}_{t,s}
=
\mathrm{score}^{MLA}_{t,s}
+
M_{t,s}
$$

于是 softmax 后，只有被 top-k 选中的 token 会参与 attention：

$$
\alpha_{t,s}
=
\operatorname{Softmax}_{s\in S_t}
\left(
\mathrm{score}^{MLA}_{t,s}
\right)
$$
最后：
$$
u_t
=
\operatorname{Attn}
\left(
h_t,
\{c_s \mid s \in S_t\}
\right)
$$
这里的 $c_s$ 是 MLA 中缓存的 latent KV entry，而不是传统 MHA 中完整的 key/value。
# k 缓存
我们可以注意到上文的 $\mathbf{k}_{s}$ 是只和历史 token 有关的，其一样可以复用。因此 indexer 会有一个 k cache。配合上 dynamic 量化的话，这个 k cache 会另外包含一个 per-token scale factor。合起来是两个：
- `k_cache`: indexer key，维度是 index_head_dim = 128  
- `k_scale_cache`: indexer key 的 FP8 scale
# 源码
这是 DS 提供的源码（未优化）。
```python
class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim: int = args.dim # 7168
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size   # 64 / TP
        self.head_dim: int = args.index_head_dim # 128
        self.rope_head_dim: int = args.qk_rope_head_dim # 64
        self.index_topk: int = args.index_topk  # 2048
        self.q_lora_rank: int = args.q_lora_rank # 1536
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim) # [b, s, 1536] -> [b, s, 64*128]
        self.wk = Linear(self.dim, self.head_dim) # [b, s, 7168] -> [b, s, 128]
        self.k_norm = LayerNorm(self.head_dim) # [b, s, 128] -> [b, s, 128]
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.get_default_dtype()) # [b, s, 7168] -> [b, s, 64]
        self.softmax_scale = self.head_dim ** -0.5 # sqrt(128)
        self.scale_fmt = args.scale_fmt

        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn), persistent=False) # [b, s, 128]
        self.register_buffer("k_scale_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32), persistent=False)


    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()  # [b, s, 7168] 
        end_pos = start_pos + seqlen 
        q = self.wq_b(qr) #针对输入的qr做矩阵乘， Input Tensor Shape:[b, s, 1536] * [1536, 64*128] -> Output Tensor Shape:[b, s, 64*128] 
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim) #将Hidden维度拆成Head Num, Head Dim两个维度，Input Tensor Shape:[b, s, 64*128] -> Output Tensor Shape:[b, s, 64, 128] 
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1) #Head Dim维度拆成两个64, Input Tensor Shape:[b, s, 64, 128] -> Output Tensor Shape:[b, s, 64, 64], [b, s, 64, 64]  
        q_pe = apply_rotary_emb(q_pe, freqs_cis) #针对上述拆分后的第一个Tensor添加位置信息， Input Tensor Shape:[b, s, 64, 64] -> Output Tensor Shape:[b, s, 64, 64]  
        q = torch.cat([q_pe, q_nope], dim=-1) #合并得到最终的q, Input Tensor Shape:[b, s, 64, 64], [b, s, 64, 64] -> Output Tensor Shape:[b, s, 64, 128]
        k = self.wk(x) #针对输入的x做矩阵乘,该输出只有一个head， Input Tensor Shape:[b, s, 7168] * [7168, 128] -> Output Tensor Shape:[b, s, 128]
        k = self.k_norm(k) #针对k做LayerNorm， Input Tensor Shape:[b, s, 128] -> Output Tensor Shape:[b, s, 128] 
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1) #Head Dim维度拆成两个64, Input Tensor Shape:[b, s, 128] -> Output Tensor Shape:[b, s, 64], [b, s, 64]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2) #针对上述拆分后的第一个Tensor添加位置信息， Input Tensor Shape:[b, s, 64] -> Output Tensor Shape:[b, s, 64]
        k = torch.cat([k_pe, k_nope], dim=-1) # [b, s, 64], [b, s, 64] -> [b, s, 128]
        q = rotate_activation(q) # 对每个Token的128维数据做Hadamard 变换 -> [b, s, 64, 128]
        k = rotate_activation(k) # 对每个Token的128维数据做Hadamard 变换 -> [b, s, 128]
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt) # 以perGroup=128量化：第一个输出是被量化的q:[b, s, 64, 128]， 第二个输出是q被量化的Scale:[b, s, 64]
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt) # 以perGroup=128量化：第一个输出是被量化的k:[b, s, 128]， 第二个输出是k被量化的Scale:[b, s]
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(x) * self.n_heads ** -0.5 #针对输入的x做矩阵乘， [b, s, 7168] -> [b, s, 64] / sqrt(128)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale #上述Weight与Q的量化Scale乘在一起，省一次点乘，并做扩维操作，Input Tensor Shape:[b, s, 64] -> Output Tensor Shape:[b, s, 64, 1]
        #总的逻辑就是：Relu(q_fp8 X weights) * weights 再延Head维度求和，再点K的Scale。 Input Tensor Shape : [b, s, 64, 128]， [b, s, 64，1], [b, s, 128], [b, s]
        # 
        index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(), self.k_scale_cache[:bsz, :end_pos].contiguous())
        # Output Tensor Shape: [b, s, s] Token之间的相关性的分数
        if mask is not None:
            index_score += mask
            
        # deepseek_v3_topk_kernel
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        topk_indices_ = topk_indices.clone() # [b, s, s] -> [b, s, 2048] 选出分数最高的2048个索引
        dist.broadcast(topk_indices_, src=0)
        assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices
```

# TensorRT-LLM 如何实现 DSA
trtllm 中的 MLA.forward_impl_with_dsa 使用了 DSA 计算注意力。
1. `pre_indexer_proj`：
	1. `k`、`weights` 都是 hidden_states 投影出来的，其中 `wk = [hidden_size, head_dim]` 、`ww = [hidden_size, num_heads]`；一合计可以直接合并成 `w_fused_wk_wp_weight = [hidden_size, num_heads+head_dim]`，用一次 GEMM 完成
	2. 然后分别生成 `q`、`k` 的 nope 和 pe 路径：`q_pe, q_nope = qr @ w_q_b`；`k_pe, k_nope = norm(k)`
2. `sparse_attn_indexer`
	1. 更新 indexer K cache
	2. 如果有 prefill token，而且不是可以直接走dense attn的短序列：
		1. 打开了chunked prefill：
			 1. 考虑 Attention TP：xxxx
			 2. Chunked Prefill，这里会进行一个分块，每个分块完成 `_call_mqa_logits` 式的 QKV 计算（多个 Q，1 个 K，中间走 `ReLU`，v 是 weights）；也可以选择自定义算子
			 3. 获得 logits 之后选取 topk
		2. 跳过了 index：直接使用预定义的 topk
	3. 如果有 decode token，情况会比较复杂，因为需要同时处理 DeepSeek MTP 的问题
		1. 如果有 MTP，就意味着每个 decode request 的 generation token 会多于 1；TRTLLM 会把摊平的 generation token 恢复成 `[num_generations,next_n,head_num,head_dim]` 的形式，需要确保一批 decode request 的 generation token 一样多。如果不一样就会报错要求 padding
		2. 使用 mqa 完成 decode 计算，获得 logits
		3. topk（GVR）

特点：
1. 主要的算子是 GEMM、多种 mqa_logits、RoPE、和 TopK；DeepSeek MLA 的 absorb 路线有 BMM
2. 只有 fp4 和 fp8 路线，没有 bf16
3. prefill 会考虑 TP，按 token 分 Q，k 预先计算好存在于所有 rank；因为不需要 softmax，最后直接 allgather 起来即可
4. K cache 也用 Paged Attention 类似的机制管理，避免搬出完整的 k
5. Decode 充分支持 DeepSeek MTP，但是算子 fp8_paged_mqa_logits 在 sm100 上只支持 MTP=1/2/4；在 sm90 只支持 1/2；不属于此列的 MTP 会直接被摊平成数个 decode token 造成性能下降

> [!note] DSA 的 Topk 问题
> 经典的 K th Element 问题使用堆排序和快速排序这样的算法，消耗时间是 $O(n\log n)$。但是 GPU 的本领是多线程，我们可以用**阈值搜索**实现（Guess Verify Refine，GVR）。
> 1. 根据随便一个元素猜测一个阈值 `t`
> 2. 统计出大于 `t` 的元素数量
> 3. 如果元素数量多于 `k` 则调大阈值 `t`
> 4. 如果元素数量小于 `k` 则调小阈值 `t`
> 
> 这个方法利用了多线程可以同时检查多个元素的方法实现加速。在这里，`t` 的初始值就会非常影响性能。NV 的源码提到，因为 logits 分布有规律，对 decode 来说 prefill 的最后一个 token 留下的 topk 索引列表很适合用于估算一个大概率很接近答案的 `t`。因此 warmup topk 算子可以大大提高性能。
