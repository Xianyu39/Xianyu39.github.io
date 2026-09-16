---
title: "卷积神经网络优化"
pubDate: 2026-08-19
description: "整理 im2col、GEMM 和卷积实现中的计算优化方法。"
author: "Westwoods"
tags: [深度学习, 模型优化]
draft: false
---

CNN 是炼丹学的宠儿，虽然在 Transformer 统治天下之后没有那么火爆，但是在多模态编解码、Linear Attention 上有不少应用。所以这边我们来看一下卷积神经网络一般被如何优化。
# im2col+GEMM
卷积本质上可以看作对输入 `x` 的各个局部做了 projection，而且这些个局部可能是重叠的。以 conv1d 为例，对 `x=[S, H]` 这个序列做一个 `kernel=3,stride=2,padding=0,output=cout (k,sd,p,cout)` 的卷积是这样的：
```
x0 x1 x2 x3 x4 x5 x6 x7
|------|
   y0  |-----|
         y1  |-----|
                y2 |---...
```
假设 y 的数量是 n，按照模拟可知我们应该按照 stride 计数，同时保证卷积核不越界。模拟可知 `n*sd+2<=S`，即 `n <= (S-2)/sd`。

这里面每个 y 根本就是范围内所有元素混合 projection 出来的。如果把每次处理的范围都列出来，`x` 的形状可以变为 `x_unfold = [n, k*H]`（包含大量重复元素），那么最终卷积结果可以直接 GEMM 出来：`y=x_unfold@W`，其中 `W=[k*H, cout]`。

这个思路很完美：若干次移动变成一次 GEMM。问题在于千万不要真的把 `x_unfold` 那些重叠的数据也真正存下来。因此我们要向虚拟内存一样建立一个**寻址机制**。
```python
def address(i,j):
	return i*sd+j/H, j%H
```

这就是最通用的 Implicit GEMM 方法。
