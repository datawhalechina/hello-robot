import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { useRoute } from 'vitepress'
import imageViewer from 'vitepress-plugin-image-viewer'
import 'viewerjs/dist/viewer.min.css'
import './custom.css'

export default {
  extends: DefaultTheme,
  setup() {
    // 点击正文中的图片可放大查看
    imageViewer(useRoute())
  },
} satisfies Theme
