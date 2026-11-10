import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type DefaultTheme } from 'vitepress'

const srcDir = fileURLToPath(new URL('..', import.meta.url))

// 侧边栏显示的章节名（与文件名中的标题对应，仅用于展示）
const parts: { text: string; chapters: [number, string][] }[] = [
  {
    text: '一、仿真基础',
    chapters: [
      [1, 'Isaac Sim 环境配置与安装'],
      [2, 'Isaac Sim 基本使用'],
      [3, 'Isaac Sim 综合实践'],
    ],
  },
  {
    text: '二、运动控制',
    chapters: [
      [4, '移动底盘运动学与控制'],
      [5, '机械臂运动学与关节控制'],
      [6, '综合实践：移动与抓取'],
    ],
  },
  {
    text: '三、环境感知',
    chapters: [
      [7, '视觉感知算法'],
      [8, '雷达点云建图'],
      [9, '综合实践：视觉感知控制'],
    ],
  },
  {
    text: '四、规划与导航',
    chapters: [
      [10, '移动底盘规划与 Nav2'],
      [11, '机械臂规划与 MoveIt 2'],
      [12, '综合实践：全场景自主导航操作'],
    ],
  },
  {
    text: '五、具身智能大模型',
    chapters: [
      [13, 'VLM 接入与 Prompt 工程'],
      [14, '视觉语言动作模型部署'],
      [15, '经验驱动的 VLA 强化学习微调'],
      [16, '潜在动作世界模型'],
    ],
  },
]

// 每个 chapterN 目录下只有一篇正文，这里把它映射成 /chapterN/，得到简洁的网址
const rewrites: Record<string, string> = {}
for (const part of parts) {
  for (const [n] of part.chapters) {
    const dir = `chapter${n}`
    const file = fs.readdirSync(path.join(srcDir, dir)).find((f) => f.endsWith('.md'))
    if (file) rewrites[`${dir}/${file}`] = `${dir}/index.md`
  }
}

const sidebar: DefaultTheme.SidebarItem[] = parts.map((part) => ({
  text: part.text,
  collapsed: false,
  items: part.chapters.map(([n, title]) => ({ text: `${n}. ${title}`, link: `/chapter${n}/` })),
}))

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hello Robot',
  description: '从仿真到具身智能大模型的机器人实战教程',
  base: '/hello-robot/',
  cleanUrls: true,
  lastUpdated: true,
  rewrites,
  head: [['link', { rel: 'icon', href: '/hello-robot/datawhale-logo.png' }]],

  markdown: {
    math: true,
    image: { lazyLoading: true },
  },

  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '首页', link: '/' },
      { text: '开始学习', link: '/chapter1/' },
    ],
    sidebar,
    socialLinks: [{ icon: 'github', link: 'https://github.com/datawhalechina/hello-robot' }],
    editLink: {
      pattern: 'https://github.com/datawhalechina/hello-robot/edit/main/docs_new/:path',
      text: '在 GitHub 上编辑此页',
    },
    footer: {
      message: 'Datawhale 开源教程',
      copyright: `Copyright © ${new Date().getFullYear()} Datawhale`,
    },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '没有找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一章', next: '下一章' },
    lastUpdated: { text: '最后更新于' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
  },
})
