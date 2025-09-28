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

The UI is available at the URL printed by Gradio (defaults to ).

### 2. Command Line Palette Extraction

Outputs HEX, RGB, and percentage shares in the terminal.

### 3. Docker
Build manually (Action does this automatically on ):

Access the app at .

## Project Structure


## GitHub Actions Deployment
The workflow  runs on pushes to  (or manually). Steps:
1. Install dependencies and run a smoke check ().
2. Build the Docker image.
3. Push the image to  (requires GitHub Packages access).

To pull the published image elsewhere:


## Development Tips
- Install dependencies locally with  if you are not using Conda.
- The 3D scatter relies on Plotly; when running in headless servers, ensure a browser-capable environment or export using Plotly tools.
- Modify the typography phrases or copy-button behavior in  within  if you want localized content.

## License
Specify your preferred license here (MIT, Apache-2.0, etc.).
