---
title: "Hugging Face 模型部署：从下载到推理"
pubDate: 2026-09-17
description: "整理 Hugging Face 模型下载、显存不足处理和推理服务选择中的几个实用原则。"
author: "Westwoods"
tags: [Hugging Face, LLM, 部署]
draft: false
---

Hugging Face Transformers 覆盖了模型下载、加载、微调和推理的完整流程。实际部署时，最容易出问题的地方通常不是调用 API，而是模型权限、权重下载、显存规划和服务化方式。

## 安全登录与下载

需要权限的模型应先在 Hugging Face 网页完成授权，再使用 CLI 登录。token 不要写进代码或提交到 Git：

```bash
hf auth login
```

下载模型时，优先使用支持断点续传和缓存的方式：

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen2.5-7b
```

在 Python 中也可以使用 `snapshot_download`，并通过 `local_dir` 或缓存目录管理文件。生产环境应固定模型版本或 commit，避免同一个服务在不同时间拉到不同权重。

## 显存不够怎么办

显存预算首先取决于参数量和权重精度。除了降低精度，还要把 KV-Cache、激活、临时 workspace 和 batch 预留出来。常见手段包括：

- 使用 FP16 或 BF16 权重；
- 使用 8-bit 或 4-bit 量化；
- 通过 `device_map` 或 Accelerate 将模型切分到多张 GPU；
- 限制上下文长度和 batch size；
- 对不常用的参数启用 CPU offload，但要接受更高延迟。

例如，使用 Transformers 加载时应确保输入和模型位于正确设备上；模型被切分到多卡时，不要简单地把输入固定到一张与 embedding 不匹配的 GPU。

## 从 Transformers 到推理服务

Transformers 适合验证模型和开发原型。高并发服务通常会进一步使用 vLLM、SGLang 等推理引擎，它们会提供连续 batching、PagedAttention、KV-Cache 管理和更高效的调度。

选择部署方案时，不要只比较单次生成速度，还要测首 token 延迟、单 token 延迟、吞吐、显存峰值、并发稳定性和服务成本。模型能跑起来只是第一步，能在目标负载下稳定运行才是部署完成。
