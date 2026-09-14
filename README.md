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

原始模板由 Tim Witzdam 发布，详见 `LICENSE`。
