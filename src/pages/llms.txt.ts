import { getCollection } from "astro:content";

export async function GET({ site }: { site: URL }) {
  const posts = await getCollection("posts", ({ data }) => !data.draft);
  posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());

  const siteUrl = site.href.replace(/\/$/, "");
  const lines = [
    "# Westwoods",
    "",
    "> Westwoods 的个人技术博客，主题集中在 AI Infra、LLM Systems、大模型训练与推理、GPU 性能优化和 LLMOps。",
    "",
    "## Site map",
    `- [首页](${siteUrl}/): 个人简介与 AI 全栈能力概览。`,
    `- [经历](${siteUrl}/about/): 研究方向、工程经历与技术栈。`,
    `- [项目](${siteUrl}/projects/): 大模型推理、性能建模与 LLMOps 项目。`,
    `- [文章](${siteUrl}/blog/): LLM 系统、性能分析与工程实践笔记。`,
    "",
    "## Articles",
  ];

  for (const post of posts) {
    lines.push(`- [${post.data.title}](${siteUrl}/blog/${post.id}/): ${post.data.description}`);
  }

  return new Response(`${lines.join("\n")}\n`, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
