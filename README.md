# Palette Explorer

Interactive Gradio app that extracts dominant colors from any image, presents them through typography previews with quick copy buttons, and visualizes the k-means clusters inside an RGB 3D space.

## Features
- Extract configurable number of colors via k-means clustering (with reproducible seeds).
- Display palette as a smooth gradient and as typography preview cards with one-click HEX copy.
- Inspect sampled pixels and centroids in an interactive 3D Plotly scatter (RGB cube).
- View palette data as structured JSON for downstream use.
- Command-line mode for quick palette dumps without launching the UI.

## Quick Start

### 1. Local (Conda environment)
    conda activate pytorch
    python palette_app.py --ui
The UI is available at the URL printed by Gradio (defaults to http://127.0.0.1:7860).

### 2. Command Line Palette Extraction
    python palette_app.py -n 8 path/to/image.jpg
Outputs HEX, RGB, and percentage shares in the terminal.

### 3. Docker
Build manually (Action does this automatically on main):
    docker build -t palette-app .
    docker run -p 7860:7860 palette-app
Access the app at http://localhost:7860.

## Project Structure
    .
    |- palette_app.py       # Main Gradio/CLI entrypoint
    |- requirements.txt     # Runtime dependencies
    |- Dockerfile           # Container definition for deployment
    |- .dockerignore        # Docker build context ignore list
    \- .github/workflows/
       \- deploy.yml        # GitHub Action to build & push image to GHCR

## GitHub Actions Deployment
The workflow .github/workflows/deploy.yml runs on pushes to main (or manually). Steps:
1. Install dependencies and run a smoke check (python -m compileall palette_app.py).
2. Build the Docker image.
3. Push the image to ghcr.io/<owner>/palette-app:latest (requires GitHub Packages access).

To pull the published image elsewhere:
    docker login ghcr.io
    docker pull ghcr.io/<owner>/palette-app:latest

## Development Tips
- Install dependencies locally with pip install -r requirements.txt if you are not using Conda.
- The 3D scatter relies on Plotly; when running in headless servers, ensure a browser-capable environment or export using Plotly tools.
- Modify the typography phrases or copy-button behavior in _typography_html within palette_app.py if you want localized content.

## License
Specify your preferred license here (MIT, Apache-2.0, etc.).
