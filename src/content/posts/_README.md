# 文章发布说明

在当前目录新建 `.md` 或 `.mdx` 文件，然后提交并推送到 `main`，GitHub Pages 会自动构建发布。

```md
---
title: "文章标题"
pubDate: 2026-09-16
description: "文章摘要，会显示在文章列表。"
author: "Westwoods"
tags: [LLM, AI Infra]
image: "banner.png"
draft: false
---

正文从这里开始。
```

文件名会成为文章地址的一部分，例如 `vllm-serving.md` 对应 `/blog/vllm-serving/`。

`draft: true` 的文章只在本地开发环境显示，不会进入线上构建。
