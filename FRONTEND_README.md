# 强化学习实验平台 - 前端开发规范与实施计划

## 1. 项目概述

本项目旨在为强化学习智能体自适应能力评估框架开发一个现代化、交互式的前端用户界面。用户将能够通过该界面直观地配置实验、启动和监控训练过程，并对生成的多种可视化结果进行动态浏览和分析，从而完全摆脱对命令行的依赖。

## 2. 技术栈

- **构建工具**: [Vite](https://vitejs.dev/)
- **核心框架**: [Vue 3](https://vuejs.org/)
- **开发语言**: [TypeScript](https://www.typescriptlang.org/)
- **编码风格**: Composition API
- **路由管理**: [Vue Router](https://router.vuejs.org/) (History 模式)
- **状态管理**: [Pinia](https://pinia.vuejs.org/)
- **UI 组件库**: [Element Plus](https://element-plus.org/)
- **HTTP客户端**: [Axios](https://axios-http.com/)
- **图表库**: [Apache ECharts](https://echarts.apache.org/) (通过 `vue-echarts` 集成)
- **代码规范**: ESLint + Prettier

## 3. 关键前提：后端 API

前端项目的运行依赖于一个稳定、高效的 **RESTful API** 服务器。后端需要将 `main_experiment.py` 的核心功能（如实验配置、运行、结果查询等）封装成 API 接口。推荐使用 **FastAPI** 进行开发。

### 核心 API 端点设计

| 方法 | 路径 | 描述 |
| :--- | :--- | :--- |
| `GET` | `/api/config/options` | 获取实验的可选配置项（如算法、分布列表）。 |
| `POST` | `/api/experiments/run` | **异步**启动一个新实验，请求体包含完整配置。 |
| `GET` | `/api/experiments` | 获取所有历史实验的列表及其状态。 |
| `GET` | `/api/experiments/{run_id}` | 获取单个实验的详细信息（配置、状态、时间等）。 |
| `GET` | `/api/experiments/{run_id}/results/raw` | 获取原始结果数据（CSV/JSON），用于前端动态绘图。 |
| `GET` | `/api/experiments/{run_id}/results/plots` | 获取已生成的静态图表文件URL列表。 |
| `GET` | `/plots/{run_id}/{image_name}` | 静态文件服务，用于访问生成的图片。 |

## 4. 前端项目结构

```
/frontend
├── public/
├── src/
│   ├── assets/               # 静态资源 (CSS, images)
│   ├── components/           # 全局通用组件
│   │   ├── charts/           # ECharts 封装组件
│   │   ├── experiment/       # 实验配置相关组件
│   │   └── layout/           # 布局组件 (Header, Sidebar)
│   ├── router/               # 路由配置 (index.ts)
│   ├── services/             # API 请求封装 (api.ts)
│   ├── stores/               # Pinia 状态管理 (experiments.ts, options.ts)
│   ├── types/                # TypeScript 类型定义
│   ├── utils/                # 工具函数
│   ├── views/                # 页面级视图
│   │   ├── ExperimentSetup.vue # 实验配置页
│   │   └── ResultsDashboard.vue# 结果仪表盘
│   ├── App.vue               # 根组件
│   └── main.ts               # 入口文件
├── .eslintrc.js
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 5. 核心功能模块

### 5.1. 实验配置页 (`/setup`)
- **功能**: 提供一个可视化的表单，用于配置新的实验。
- **UI组件**:
  - 算法选择器（下拉菜单）。
  - 初始/目标环境分布配置器（动态表单，根据所选分布显示不同参数输入框）。
  - 超参数优化（HPO）设置（开关及参数输入）。
- **交互**:
  - 点击“启动实验”后，将配置发送至 `POST /api/experiments/run`。
  - 请求期间显示加载状态，成功后提示用户并导航至结果仪表盘。

### 5.2. 结果仪表盘 (`/dashboard`)
- **功能**: 展示所有历史实验的列表和单个实验的详细结果。
- **UI组件**:
  - 左侧为可搜索、可排序的历史实验列表。
  - 右侧为结果展示区，使用标签页（Tabs）对图表进行分类。
- **交互**:
  - 点击左侧列表项，右侧内容根据路由参数 `:run_id` 动态更新。
  - **图表展示（两种方案）**:
    1.  **快速实现**: 直接加载并显示后端生成的静态图片。
    2.  **高级交互**: 获取原始数据，使用 ECharts 在前端进行动态、可交互的渲染。

## 6. 开发环境与工作流程

1.  **环境准备**:
    - 安装 [Node.js](https://nodejs.org/) (LTS版本)
    - 安装 [pnpm](https://pnpm.io/) (推荐)

2.  **项目初始化** (在项目根目录下):
    ```bash
    # 进入前端目录 (如果已创建)
    # cd frontend

    # 安装依赖
    pnpm install
    ```

3.  **启动开发服务器**:
    ```bash
    pnpm run dev
    ```
    应用将在本地启动，并支持热模块重载（HMR）。

4.  **构建生产版本**:
    ```bash
    pnpm run build
    ```
    构建后的文件将输出到 `frontend/dist` 目录。

5.  **代码规范检查与修复**:
    ```bash
    # 检查
    pnpm run lint

    # 自动修复
    pnpm run lint:fix
    ```

## 7. 贡献指南

- **分支管理**:
  - `main`: 主分支，保持稳定。
  - `develop`: 开发分支，所有新功能分支从此切出。
  - `feature/xxx`: 新功能开发分支。
- **Commit 规范**: 请遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范，便于生成清晰的变更日志。
- **代码风格**: 遵循项目配置的 ESLint 和 Prettier 规则，提交前请确保通过 `lint` 检查。 