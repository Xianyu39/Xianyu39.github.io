---
title: "ML Infra 推理服务抽象模型"
pubDate: 2025-12-23
description: "从管理平面、控制平面和数据平面理解机器学习推理服务的抽象结构。"
author: "Westwoods"
tags: [ML Infra, 推理服务, 系统设计]
draft: false
---

最近觉得被合在主服务里面的 Embedding 和 Rerank 服务很不好用，TEI 又太不稳定，于是准备自己写一个服务。GPT 向我建议了一个架构方案，我觉得很有价值。
# 三层结构
GPT 宣扬一种三层结构：
1. Service：负责想办法高效地处理大批请求，将它们打包为一个一个高效处理的**任务**，在这里实现缓存队列、continuous batching、LRO 等内容；它关心的是：
	- QPS / 并发
	- 排队 / 限流
	- 批处理策略
	- SLA / timeout
	- Retry / Fallback
2. Task：负责想办法高效处理 Service 层传递过来的**任务**，这一层需要处理输入和输出后处理，调用 Runtime 完成任务；
	1. tokenizer 使用方式
	2. 输入格式（单句 / pair / 多字段）
	3. pooling / score 逻辑
	4. 输出 shape 的解释
	5. batch 内部结构
3. Runtime：硬件执行层，负责高效处理底层问题，比如选择硬件，导入模型，想办法实施高效的计算；这一层不负责理解输入，也不负责解释输出。

其实这种结构在食养通项目中已经存在。`EmbeddingBackend` 类实现了 `embed_sentence` 等方法，然后提供了 `inference` 接口待实现。这个接口只有一个 `inputs` 参数，返回也只是返回 logits；我们只继承这个类，然后实现 `inference` 底层内容即可。输入输出已经在基类内完成。这其实就是在试图复用 task 部分，并且要求你完成 runtime 部分。

```
--- Stress Test Results (默认版本)---
Total requests: 100
Successful requests: 21
Failed requests: 79
Success rate: 21.00%
Total time: 192.47 seconds
Average time per request: 1.92 seconds
FTesting with 100 large texts, total chars: 73769
Large payload test successful: torch.Size ([100, 512]) in 24.14s

--- Stress Test Results (10线程)---
Total requests: 100
Successful requests: 100
Failed requests: 0
Success rate: 100.00%
Total time: 16.68 seconds
Average time per request: 0.17 seconds
.Testing with 100 large texts, total chars: 77007
Large payload test successful: torch.Size ([100, 512]) in 1.51s

--- Stress Test Results (onnx，10线程)---
Total requests: 100
Successful requests: 100
Failed requests: 0
Success rate: 100.00%
Total time: 13.83 seconds
Average time per request: 0.14 seconds
.Testing with 100 large texts, total chars: 73602
Large payload test successful: torch.Size([100, 512]) in 3.56s

--- Stress Test Results (onnx参数调优，task、10线程runtime6线程)---
Total requests: 100
Successful requests: 100
Failed requests: 0
Success rate: 100.00%
Total time: 11.60 seconds
Average time per request: 0.12 seconds
.Testing with 100 large texts, total chars: 74802
Large payload test successful: torch.Size([100, 512]) in 2.26s
```

队列逻辑：
1. 设定 k 个槽位，意味着最多同时处理 k 个请求
2. 请求到来时，如果有槽位，就直接处理
3. 如果没有，就加入队列
4. 当有槽位释放，就从队列里取请求，一次性取若干个请求并作一个 batch 处理
5. 处理完毕之后，batch 拆解为各个请求的返回值，全部返回
