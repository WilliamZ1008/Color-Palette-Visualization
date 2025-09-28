"""Interactive color palette extractor and demo.

This script loads an image, extracts a palette with a configurable
number of colors, and showcases the palette inside a Gradio interface
using multiple visual styles (gradient bar and card layout).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - handled when running the UI
    raise ImportError(
        "Gradio is required to run the interactive demo."
    ) from exc

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - plotting is optional
    raise ImportError(
        "Plotly is required for the 3D scatter visualization. Install it via `pip install plotly`."
    ) from exc


@dataclass
class PaletteColor:
    """Container for palette metadata."""

    rgb: Tuple[int, int, int]
    percentage: float

    @property
    def hex(self) -> str:
        return "#" + "".join(f"{channel:02X}" for channel in self.rgb)


@dataclass
class PaletteResult:
    """Aggregated palette data and clustering artifacts."""

    colors: List[PaletteColor]
    samples: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray


def _prepare_pixels(image: Image.Image, max_sample: int = 5000) -> np.ndarray:
    """Convert an image into a 2D array of pixels and optionally subsample."""

    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")

    pixels = np.array(image)
    if pixels.ndim == 3 and pixels.shape[2] == 4:  # strip alpha channel
        pixels = pixels[:, :, :3]

    flat_pixels = pixels.reshape(-1, 3).astype(np.float32)

    if len(flat_pixels) > max_sample:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(flat_pixels), size=max_sample, replace=False)
        flat_pixels = flat_pixels[indices]

    return flat_pixels


def _kmeans(
        pixels: np.ndarray,
        num_colors: int,
        *,
        max_iter: int = 25,
        tol: float = 1e-2,
        seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple k-means clustering implementation for RGB pixels."""

    if len(pixels) == 0:
        raise ValueError("No pixels available for clustering")

    num_colors = max(1, min(num_colors, len(pixels)))

    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(len(pixels), size=num_colors, replace=False)
    centroids = pixels[initial_indices]

    for _ in range(max_iter):
        distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.vstack(
            [pixels[labels == idx].mean(axis=0) if np.any(labels == idx) else centroids[idx]
             for idx in range(num_colors)]
        )

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    counts = np.array([(labels == idx).sum() for idx in range(num_colors)], dtype=np.int32)
    order = np.argsort(counts)[::-1]
    remap = np.empty_like(order)
    remap[order] = np.arange(num_colors)
    remapped_labels = remap[labels]
    return centroids[order], counts[order], remapped_labels


def extract_palette(
        image_source: Image.Image | np.ndarray | str,
        num_colors: int,
        seed: int | None = None,
) -> PaletteResult:
    """Extract a color palette with num_colors entries from the image."""

    if isinstance(image_source, Image.Image):
        image = image_source
    elif isinstance(image_source, np.ndarray):
        image = Image.fromarray(image_source.astype(np.uint8))
    elif isinstance(image_source, str):
        image = Image.open(image_source)
    else:
        raise TypeError("Unsupported image source type")

    pixels = _prepare_pixels(image)
    centroids, counts, labels = _kmeans(pixels, num_colors, seed=seed)

    total = counts.sum()
    palette = []
    for centroid, count in zip(centroids, counts):
        rounded = np.clip(np.round(centroid), 0, 255).astype(np.uint8)
        palette.append(PaletteColor(tuple(int(channel) for channel in rounded), count / total))

    return PaletteResult(colors=palette, samples=pixels, labels=labels, centroids=centroids)


def _gradient_html(palette: Sequence[PaletteColor]) -> str:
    """Create a CSS gradient bar to display the palette smoothly."""

    if not palette:
        return "<div>暂无调色数据</div>"

    stops = []
    total = len(palette) - 1 or 1
    for idx, color in enumerate(palette):
        percent = (idx / total) * 100
        stops.append(f"{color.hex} {percent:.2f}%")

    gradient = ", ".join(stops)
    return f"<div style=\"height: 48px; border-radius: 8px; border: 1px solid #d1d5db; background: linear-gradient(90deg, {gradient});\"></div>"


def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """Calculate the WCAG relative luminance for an RGB tuple."""

    def _channel_linear(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.03928 else ((srgb + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _channel_linear(r) + 0.7152 * _channel_linear(g) + 0.0722 * _channel_linear(b)


def _typography_html(palette: Sequence[PaletteColor]) -> str:
    """Show typography samples on colored backgrounds."""

    if not palette:
        return "<div>暂无调色数据</div>"

    previews = []
    phrases = [
        "用色彩描绘心境",
        "颜色是灵魂的触觉",
        "色彩即语言"
    ]
    for idx, color in enumerate(palette[:6]):
        luminance = _relative_luminance(color.rgb)
        text_hex = "#101321" if luminance > 0.55 else "#F5F7FF"
        secondary_hex = "#2E3143" if luminance > 0.55 else "#D8E2FF"
        phrase = phrases[idx % len(phrases)]
        button = (
            f"<button type='button' style=\"float:right;margin-bottom:0.6rem;padding:0.3rem 0.6rem;"
            "border:1px solid rgba(17,24,39,0.2);border-radius:6px;background-color:rgba(255,255,255,0.8);"
            "font-size:0.75rem;cursor:pointer;\""
            f" onclick=\"navigator.clipboard.writeText('{color.hex}');"
            "this.textContent='已复制';setTimeout(()=>this.textContent='复制 HEX',1200);\">复制 HEX</button>"
        )
        previews.append(
            "<div style=\"border:1px solid #d1d5db;border-radius:8px;padding:0.8rem;"
            f"background:{color.hex};color:{text_hex};margin-bottom:0.6rem;position:relative;overflow:hidden;\">"
            f"  {button}"
            f"  <div style=\"font-weight:600;\">{phrase}</div>"
            f"  <div style=\"font-size:0.85rem;color:{secondary_hex};margin-top:0.2rem;\">{color.hex} · {color.rgb}</div>"
            "  <div style=\"margin-top:0.6rem;font-size:0.85rem;\">"
            "    在当前背景色上预览文字对比度。"
            "  </div>"
            "</div>"
        )

    return "".join(previews)


def _scatter_figure(result: PaletteResult) -> go.Figure:
    """Build a 3D scatter figure of sampled pixels in RGB space."""

    fig = go.Figure()

    samples = result.samples
    labels = result.labels
    for idx, color in enumerate(result.colors):
        cluster_points = samples[labels == idx]
        if cluster_points.size == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode="markers",
                marker=dict(size=3, color=color.hex, opacity=0.35),
                name=f"Cluster {idx + 1}",
                hovertemplate="R:%{x:.0f}<br>G:%{y:.0f}<br>B:%{z:.0f}<extra>{color.hex}</extra>",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=result.centroids[:, 0],
            y=result.centroids[:, 1],
            z=result.centroids[:, 2],
            mode="markers",
            marker=dict(
                size=9,
                color=[color.hex for color in result.colors],
                symbol="diamond",
                line=dict(width=1.5, color="#111111"),
            ),
            name="Centroids",
            hovertemplate="R:%{x:.0f}<br>G:%{y:.0f}<br>B:%{z:.0f}<extra>Centroid</extra>",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Red", range=[0, 255]),
            yaxis=dict(title="Green", range=[0, 255]),
            zaxis=dict(title="Blue", range=[0, 255]),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", x=0.0, y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def analyze_image(
    image: Image.Image | np.ndarray,
    num_colors: int,
    seed: int,
) -> Tuple[str, str, go.Figure | None, List[dict]]:
    """Processing function used by the Gradio interface."""

    if image is None:
        empty = "<div>请上传图片</div>"
        return empty, empty, None, []

    result = extract_palette(image, num_colors=num_colors, seed=seed)

    json_payload = [
        {
            "hex": color.hex,
            "rgb": color.rgb,
            "percentage": round(color.percentage, 4),
        }
        for color in result.colors
    ]

    return (
        _gradient_html(result.colors),
        _typography_html(result.colors),
        _scatter_figure(result),
        json_payload,
    )


CUSTOM_CSS = None


def build_demo(default_num_colors: int = 5, default_seed: int = 42) -> "gr.Blocks":
    """Create the Gradio Blocks interface."""

    default_num_colors = int(max(2, min(12, default_num_colors)))
    default_seed = int(max(0, min(10_000, default_seed)))

    with gr.Blocks(css=CUSTOM_CSS, title="Palette Explorer") as demo:
        gr.Markdown(
            """
            # 🎨 Palette Explorer

            上传图片，提取主要色调，并以多种方式查看调色板数据。
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=6, min_width=420):
                input_image = gr.Image(label="上传图片", type="pil", height=420)
                with gr.Row():
                    num_colors = gr.Slider(
                        minimum=2,
                        maximum=12,
                        value=default_num_colors,
                        step=1,
                        label="调色板颜色数量",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=10_000,
                        value=default_seed,
                        step=1,
                        label="随机种子",
                    )

                run_btn = gr.Button("生成调色板")
                gr.Markdown("_提示：尝试调整颜色数量和随机种子，以探索不同的聚类结果。_")

            with gr.Column(scale=6, min_width=420):
                gradient_view = gr.HTML(label="渐变光带")
                typography_view = gr.HTML(label="排版预览")
                scatter_view = gr.Plot(label="RGB 三维散点")
                data_view = gr.JSON(label="调色板数据")

        run_btn.click(
            fn=analyze_image,
            inputs=[input_image, num_colors, seed],
            outputs=[
                gradient_view,
                typography_view,
                scatter_view,
                data_view,
            ],
        )

    return demo


def launch_demo(
    default_num_colors: int = 5,
    default_seed: int = 42,
    *,
    share: bool = False,
    server_name: str | None = None,
    server_port: int | None = None,
) -> None:
    """Launch the Gradio demo with optional configuration overrides."""

    demo = build_demo(default_num_colors=default_num_colors, default_seed=default_seed)
    demo.launch(share=share, server_name=server_name, server_port=server_port)



def run_cli(image: str, num_colors: int, seed: int) -> None:
    """Command-line execution path printing palette information."""

    result = extract_palette(image, num_colors=num_colors, seed=seed)

    print("Hex	RGB	Percentage")
    for color in result.colors:
        print(f"{color.hex}	{color.rgb}	{color.percentage * 100:.2f}%")



def main() -> None:
    """Entry point that supports both CLI and UI usage."""

    parser = argparse.ArgumentParser(description="Extract a color palette or launch the Gradio UI")
    parser.add_argument("image", nargs="?", help="Path to the input image for CLI mode")
    parser.add_argument("-n", "--num-colors", type=int, default=5, help="Number of colors in the palette or default for UI")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for k-means initialisation or default for UI")
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio interface instead of the CLI output")
    parser.add_argument("--share", action="store_true", help="Share the Gradio demo publicly")
    parser.add_argument("--server-name", type=str, default=None, help="Hostname for Gradio server")
    parser.add_argument("--server-port", type=int, default=None, help="Port for Gradio server")

    args = parser.parse_args()

    if args.ui or args.image is None:
        launch_demo(
            default_num_colors=args.num_colors,
            default_seed=args.seed,
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
        )
    else:
        run_cli(args.image, num_colors=args.num_colors, seed=args.seed)


if __name__ == "__main__":
    main()
