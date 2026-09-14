import type {
  NavBarLink,
  SocialLink,
  Identity,
  AboutPageContent,
  ProjectPageContent,
  BlogPageContent,
  HomePageContent,
} from "./types/config";

export const identity: Identity = { name: "方贺进", logo: "/logo.svg", email: "fanghejin@qq.com" };

export const navBarLinks: NavBarLink[] = [
  {
    title: "首页",
    url: "/",
  },
  {
    title: "经历",
    url: "/about",
  },
  {
    title: "项目",
    url: "/projects",
  },
  {
    title: "文章",
    url: "/blog",
  },
];

export const socialLinks: SocialLink[] = [
  {
    title: "GitHub",
    url: "https://github.com/Xianyu39",
    icon: "mdi:github",
    external: true,
  },
  {
    title: "Email",
    url: "mailto:fanghejin@qq.com",
    icon: "mdi:email",
  },
];

// Home (/)
export const homePageContent: HomePageContent = {
  seo: {
    title: "方贺进 · AI Infra Engineer",
    description: "专注大模型训练、推理优化与 LLM 系统工程。",
    image: identity.logo,
  },
  role: "AI Infra Engineer · LLM Systems",
  description:
    "我关注大模型训练与推理系统：从 GPU 性能建模、算子评估，到推理引擎适配和 LLMOps 工程化。现在于江南大学攻读软件工程硕士。",
  socialLinks: socialLinks,
  links: [
    {
      title: "查看项目",
      url: "/projects",
    },
    {
      title: "了解经历",
      url: "/about",
    },
  ],
};

// About (/about)
export const aboutPageContent: AboutPageContent = {
  seo: {
    title: "经历 | 方贺进",
    description: "方贺进的研究方向、工程经历与技术栈。",
    image: identity.logo,
  },
  subtitle: "训练、推理与系统工程",
  about: {
    description: `
我是一名关注 **LLM Systems 与 AI Infra** 的工程师，目前在江南大学攻读软件工程硕士，研究方向为大模型推理增强与可信化。

我的工作横跨训练框架、推理引擎、GPU 性能分析与 LLMOps。相比单点模型调优，我更享受把复杂实验变成稳定、可观测、可复用的系统。`,
    image_l: {
      url: "/demo-1.jpg",
      alt: "Left Picture",
    },
    image_r: {
      url: "/demo-1.jpg",
      alt: "Right Picture",
    },
  },
  work: {
    description: `围绕大模型训练与推理，我积累了从底层性能分析到平台工程的完整实践。`,
    items: [
      {
        title: "AI Infra 工程实习生",
        company: {
          name: "快手",
          image: "/logo.svg",
          url: "/about",
        },
        date: "2026.09 — 至今",
      },
      {
        title: "AI Software Engineer Graduate Intern",
        company: {
          name: "Intel",
          image: "/logo.svg",
          url: "/about",
        },
        date: "2026.06 — 2026.09",
      },
    ],
  },
  connect: {
    description: `如果你也在做大模型系统、GPU 性能优化或 AI Infra，欢迎交流。`,
    links: socialLinks,
  },
};

// Projects (/projects)
export const projectsPageContent: ProjectPageContent = {
  seo: {
    title: "项目 | 方贺进",
    description: "大模型推理、性能建模与 LLMOps 项目。",
    image: identity.logo,
  },
  subtitle: "把模型能力落到可靠、高效的系统里",
  projects: [
    {
      title: "vLLM-Omni · SparkTTS",
      description: "将 SparkTTS 拆分为三个可调度 Stage，设计重叠分块流式解码；decoder stage latency 降低 **26.91%**。",
      year: "2026",
      url: "https://github.com/Xianyu39",
      tags: ["vLLM-Omni", "Streaming", "H200"],
    },
    {
      title: "auto-bench",
      description: "面向 B200/B300 的 TensorRT-LLM 多 workload 批量测试链路，用于性能模型校准与硬件平台对比。",
      year: "2026",
      url: "https://github.com/Xianyu39/auto-bench",
      tags: ["TensorRT-LLM", "Benchmark", "GPU"],
    },
    {
      title: "Squeeze LLMOps",
      description: "面向 30+ 研究人员的共享推理底座，统一模型服务、智能路由与批量实验，将完整实验从 200+ 小时缩短至约 1 小时。",
      year: "2024 — 2026",
      url: "https://github.com/Xianyu39/SqueezeLM",
      tags: ["vLLM", "LLMOps", "Observability"],
    },
  ],
};

// Blog (/blog)
export const blogPageContent: BlogPageContent = {
  seo: {
    title: "文章 | 方贺进",
    description: "关于 LLM 系统、性能分析与工程实践的笔记。",
    image: identity.logo,
  },
  subtitle: "记录 LLM 系统、性能分析与工程实践",
};
