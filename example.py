import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pi4 geometry and motion inference")
    parser.add_argument("input", type=Path, help="image, image directory, or video")
    parser.add_argument(
        "--weights",
        default="yyfz233/Pi4",
        help="local weight file/directory or Hugging Face model ID (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--query-frame", type=int, help="tracking query index after frame sampling"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="input frame sampling stride (default: 1)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="optional limit on sampled input frames (default: all frames)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps"),
        default="auto",
        help="inference device (auto selects CUDA, then Apple Silicon MPS)",
    )
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="export browser-viewer data with a 40 MB point-cloud budget",
    )
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Choose a new output directory: {arguments.output}")

    import torch

    from pi4 import Pi4
    from pi4.io import load_images
    from pi4.visualization import save_visualization

    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps"
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        parser.error(
            "Pi4 requires CUDA or Apple Silicon MPS"
            if arguments.device == "auto"
            else f"Requested device {device!r} is unavailable"
        )

    frames, metadata = load_images(
        arguments.input,
        stride=arguments.stride,
        max_frames=arguments.max_frames,
        max_size=arguments.max_size,
    )
    print(
        f"Processing {len(frames)} frames at {frames.shape[2]}x{frames.shape[1]} "
        f"on {device} in one forward (frame cap: {arguments.max_frames or 'all'}, stride: {arguments.stride}).",
        flush=True,
    )
    images = (
        torch.from_numpy(frames.copy())
        .permute(0, 3, 1, 2)
        .float()
        .div_(255)
        .unsqueeze(0)
        .to(device)
    )
    model = Pi4.from_pretrained(arguments.weights).to(device)
    prediction = model(images, query_frame=arguments.query_frame)
    scene = save_visualization(
        arguments.output, frames, prediction, metadata, viewer=arguments.viewer
    )
    if arguments.viewer:
        print(f"Viewer: python viewer/serve.py {arguments.output}")
    print(f"Pi4: {len(frames)} images, {scene['preview']['points']} preview points")
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
