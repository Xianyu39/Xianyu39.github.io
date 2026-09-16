# Westwoods 的个人网站

基于 Astro Minimal Portfolio 定制的静态个人网站，聚焦 AI Infra、LLM 训练与推理系统。

## 给 AI 的内容维护说明

这是一份静态 Astro 网站。AI 协助维护内容时，应优先保留用户提供的原始事实、措辞和附件，不凭空补充经历、指标、日期、链接或引用。无法确认的内容先标为“待确认”，不要直接发布。

### 目录约定

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 文章正文 | `src/content/posts/<slug>.md` 或 `.mdx` | 文件名就是文章 URL 的 slug |
| 文章图片 | `public/images/posts/<slug>/` | 封面和正文图片放在对应文章目录 |
| 可下载附件 | `public/attachments/<slug>/` | PDF、表格、演示文稿、数据文件等 |
| 个人资料与站点文案 | `src/config.ts` | 首页、经历、项目和 SEO 文案的来源 |
| 页面样式 | `src/pages/`、`src/styles/` | 修改前先确认是否会影响其他页面 |

### 处理附件

1. 先列出附件并确认文件类型、文件名和用途；不要覆盖、删除或重命名用户原始文件。
2. 需要随网站发布的附件复制到 `public/attachments/<slug>/`，保持清晰、稳定的文件名，例如 `architecture.pdf`、`benchmark-results.csv`。
3. 文章中的下载链接使用站点路径，例如：

   ```md
   [下载实验报告](/attachments/vllm-serving/benchmark-results.pdf)
   ```

4. 图片放到 `public/images/posts/<slug>/`，正文中使用 `/images/posts/<slug>/filename.webp`；图片必须有描述性的 `alt` 文本。
5. 不把临时截图、原始附件或无关文件提交到仓库。附件若包含密码、Token、身份证件、银行卡信息或其他隐私，先停止并要求用户处理。

### 整理文本并生成文章

1. 将用户提供的文本、笔记或附件内容作为唯一事实来源，先提炼标题、摘要、日期、标签和正文结构。
2. 默认先创建 `draft: true` 的文章，让用户检查；只有用户明确要求发布时才改为 `draft: false`。
3. 文章 slug 使用小写 kebab-case，只使用英文、数字和连字符，例如 `vllm-serving-benchmark.md`。
4. 新文章的最小格式如下：

   ```md
   ---
   title: "文章标题"
   pubDate: 2026-09-16
   description: "文章摘要，会显示在文章列表。"
   author: "Westwoods"
   tags: [LLM, AI Infra]
   image: "/images/posts/vllm-serving/cover.webp"
   draft: true
   ---

   正文从这里开始。
   ```

5. `title`、`pubDate`、`description` 是必填项；`author` 默认是 `Westwoods`，`tags` 默认是空数组，`image` 和 `updatedDate` 可选。
6. 保留原文中的代码、数据和引用关系；需要改写时只改善结构和表达，不改变技术结论。引用外部资料时补充来源链接。

### 发布流程

只有用户明确说“发布”“推送”或等价指令时，AI 才执行最后的 Git 操作。

1. 检查文章 frontmatter、图片路径、附件链接和 slug 是否正确。
2. 本地运行构建检查：

   ```bash
   npm run build
   ```

3. 查看变更范围，确认没有把临时文件、密钥或无关修改带入提交：

   ```bash
   git status --short
   git diff --check
   ```

4. 用户确认后提交并推送到 `main`：

   ```bash
   git add -A
   git commit -m "发布文章：文章标题"
   git push origin main
   ```

5. GitHub Actions 会自动构建并发布到 GitHub Pages。发布后检查 `/blog/`、文章详情页、图片和附件下载链接；若构建失败，先修复并重新检查，不要反复盲目推送。

文章地址格式为 `https://xianyu39.github.io/blog/<slug>/`。`draft: true` 的文章只在本地开发环境显示，不会进入线上构建。

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
