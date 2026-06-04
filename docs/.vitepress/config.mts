import { defineConfig } from "vitepress";

const repo = "https://github.com/YANG-Zijie/dnarna";

const enSidebar = [
  {
    text: "Overview",
    items: [
      { text: "Home", link: "/en/" },
      { text: "Paper", link: "/en/paper" },
    ],
  },
  {
    text: "Data",
    items: [
      { text: "Sequence Input", link: "/en/data/seq" },
      { text: "DNA-RNA Pair Data", link: "/en/data/pair" },
    ],
  },
  {
    text: "Models",
    items: [{ text: "Pair Prediction", link: "/en/models/pair" }],
  },
];

const zhSidebar = [
  {
    text: "概览",
    items: [
      { text: "首页", link: "/zh/" },
      { text: "论文", link: "/zh/paper" },
    ],
  },
  {
    text: "数据",
    items: [
      { text: "序列输入", link: "/zh/data/seq" },
      { text: "DNA-RNA 对数据", link: "/zh/data/pair" },
    ],
  },
  {
    text: "模型",
    items: [{ text: "配对预测", link: "/zh/models/pair" }],
  },
];

export default defineConfig({
  title: "DnaRna",
  description: "DNA-RNA interaction prediction model and workflow",
  base: process.env.VITEPRESS_BASE ?? "/dnarna/",
  cleanUrls: true,
  lastUpdated: true,
  themeConfig: {
    socialLinks: [{ icon: "github", link: repo }],
    search: {
      provider: "local",
    },
  },
  locales: {
    en: {
      label: "English",
      lang: "en-US",
      title: "DnaRna",
      description: "DNA-RNA interaction prediction model and workflow",
      themeConfig: {
        nav: [
          { text: "Paper", link: "/en/paper" },
          { text: "Data", link: "/en/data/seq" },
          { text: "Model", link: "/en/models/pair" },
        ],
        sidebar: enSidebar,
      },
    },
    zh: {
      label: "中文",
      lang: "zh-CN",
      title: "DnaRna",
      description: "DNA-RNA 潜在互作预测模型与流程",
      themeConfig: {
        nav: [
          { text: "论文", link: "/zh/paper" },
          { text: "数据", link: "/zh/data/seq" },
          { text: "模型", link: "/zh/models/pair" },
        ],
        sidebar: zhSidebar,
      },
    },
  },
});
