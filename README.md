# Palette Explorer

一款交互式 Gradio 应用，可从任意图片中提取主色调，通过带复制按钮的排版卡片展示结果，并在 RGB 三维空间中可视化 k-means 聚类分布。

## 功能特点
- 使用可配置数量的 k-means 聚类提取主色调，并支持随机种子复现结果。
- 生成平滑的渐变带与排版预览卡片，提供一键复制 HEX 颜色码。
- 在交互式 Plotly 三维散点图中查看采样像素与聚类中心的 RGB 分布。
- 以结构化 JSON 展示调色板数据，便于二次处理。
- 提供命令行模式，无需启动界面即可快速导出调色信息。

## 快速上手

### 1. 本地（Conda 环境）
```
conda activate pytorch
python palette_app.py --ui
```
Gradio 会在终端输出访问地址（默认 `http://127.0.0.1:7860`）。

### 2. 命令行调色板提取
```
python palette_app.py -n 8 path/to/image.jpg
```
终端将输出颜色的 HEX、RGB 值及占比信息。

### 3. Docker
手动构建（GitHub Action 在 `main` 分支上会自动构建）：
```
docker build -t palette-app .
docker run -p 7860:7860 palette-app
```
浏览器访问 `http://localhost:7860` 查看应用。

## 项目结构
```
.
├─ palette_app.py       # Gradio / CLI 主入口
├─ requirements.txt     # 运行时依赖
├─ Dockerfile           # 部署用容器定义
├─ .dockerignore        # Docker 构建忽略列表
└─ .github/workflows/
   └─ deploy.yml        # 构建并推送镜像到 GHCR 的 GitHub Action
```

## GitHub Actions 部署
工作流 `.github/workflows/deploy.yml` 会在推送到 `main`（或手动触发）时执行：
1. 安装依赖并运行烟雾测试（`python -m compileall palette_app.py`）。
2. 构建 Docker 镜像。
3. 推送镜像到 `ghcr.io/<owner>/palette-app:latest`（需启用 GitHub Packages 权限）。

在其他环境拉取已发布镜像：
```
docker login ghcr.io
docker pull ghcr.io/<owner>/palette-app:latest
```

## 开发提示
- 若不使用 Conda，可执行 `pip install -r requirements.txt` 安装依赖。
- 三维散点依赖 Plotly，若在无图形界面的服务器运行，可考虑导出静态文件或使用具备浏览器的环境。
- 若需本地化或自定义文案，可修改 `palette_app.py` 中 `_typography_html` 的短句与复制按钮行为。

## 许可证
请在此填写项目适用的许可证（如 MIT、Apache-2.0 等）。
