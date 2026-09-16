# 方贺进的个人网站

基于 Astro Minimal Portfolio 定制的静态个人网站，聚焦 AI Infra、LLM 训练与推理系统。

## 本地开发

```bash
npm install
npm run dev
```

## 发布到 GitHub Pages

1. 在 GitHub 创建公开仓库 `Xianyu39.github.io`。
2. 将本地仓库推送到 `main` 分支。
3. 在仓库 `Settings → Pages` 中将 Source 设为 `GitHub Actions`。

后续每次推送到 `main` 都会自动构建和发布。

## 内容维护

- 个人资料、经历和项目：`src/config.ts`
- 文章：`src/content/posts/`
- 全局样式：`src/styles/global.css`

### 发布文章

在 `src/content/posts/` 新建一个 `.md` 或 `.mdx` 文件，写入以下 frontmatter：

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
```

文件名会成为文章地址的一部分，例如 `vllm-serving.md` 对应 `/blog/vllm-serving/`。设置 `draft: true` 的文章只在本地开发环境显示，不会发布到线上。推送到 `main` 后，GitHub Actions 会自动构建并更新网站。

原始模板由 Tim Witzdam 发布，详见 `LICENSE`。
