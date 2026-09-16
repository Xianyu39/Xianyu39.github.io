---
title: "使用hugging face部署大模型"
pubDate: 2026-09-17
description: "来自 only-notes 的原始笔记，保留原文内容并做网页格式适配。"
author: "Westwoods"
tags: [Hugging Face, LLM, 部署]
draft: false
---

用hugging face部署大模型来实现微调和训练是一个很好的选择。本文记录一些使用经验。
# 部署相关问题
Hugging Face（以下简称 hf）这套框架很完整，从下载模型、训练、微调到发布，什么都有。

现在 hf 下载模型是需要权限的，所以务必先去网站上注册账号，获取一个验证 token 过来。这个 token 不要写在代码里，在命令行用 `huggingface-cli` 命令 `login` 输入 token 就可以表明自己的身份的同时避免 token 泄露。除了这个，也可以把 token 写在环境变量里面。

虽然著名的 `from_pretrained` 方法会自动下载模型，但是由于 hf 在国外，下载模型也难免被墙的命运，所以最好还是手动下载。手动下载方法包括：
1. 去网页手动下载（免登录）
	1. 可以使用 IDM、Aria 2 多线程下载，很快[^1]
2. `git clone`，但是下载断了就要重新下载，巨慢，还会下载多余的历史版本
3. `huggingface-cli download`，好用，支持断点续传，但是断了需要手动恢复
	1. 下载 `hf_transfer` 可以让 `huggingface-cli` 支持多线程下载[^1]，需要设置环境变量 `HF_HUB_ENABLE_HF_TRANSFER=1` 来开启；**这个方法只允许从官方网站下载内容**
4. `snapshot_download`，这是最好的，还可以选择性下载，但是设置比较复杂
5. 镜像网站 `https://hf-mirror.com`，设置环境变量 `HF_ENDPOINT` 为镜像站就好了
6. HFD：有大佬提供的下载脚本 [CLI-Tool for download Huggingface models and datasets with aria2/wget: hfd](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f#file-hfd-sh)，脚本不依赖 huggingface-cli 以及 python，可以使用 aria 2 以及 wget 等下载 huggingface 模型，而且允许加速下载镜像网站的模型；但是这个脚本只能用于 Linux/MacOS
# 推理问题
测试了一下，qwen 2.5-8 b-instruct 在 hf 里面需要约莫 30 G 显存才能实现 GPU 加速推理，单张显卡是不够的。但是很走运的是 hf 贴心地弄了一个 `accelerate` 库用于自动把模型分配到不同的 GPU 上实现推理。如果显存还是不够，可以考虑：
1. 降低参数精度：参数占据了绝大多数内存，考虑换成 16 位甚至 8 位浮点数，hf 有 `bitsandbytes` 供你实现这个目的
2. 把部分参数存在硬盘上交给 CPU 处理（设置 `offload_folder` 和 `offload_state_dict` 两个参数）
3. 减小输入长度

输入不会自动分配到 GPU 上面，要记得使用 `input.to(model.device)`，或者手动把输入放到模型第一层（输入层所在的 GPU 上面）
## vLLM
原生 HuggingFace Transformer 效率非常低下，而 vLLM 则可以无缝嵌入到 HF 代码中并且将效率拉到最高。

[^1]: https://zhuanlan.zhihu.com/p/663712983?s_r=0
