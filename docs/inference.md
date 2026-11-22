# Inference Guide

[← Project overview](../README.md)

Run all commands in this guide from the repository root.

Pi4 reconstructs scene geometry and dense motion from RGB images and videos
without supplied camera calibration.
This repository contains one fixed inference model, not a configurable training framework.

## Install

### Apple Silicon

Use an ARM64 Python 3.13 environment. Pi4 uses PyTorch MPS for inference,
including sparse geometry refinement, and loads the same Safetensors checkpoint
as the CUDA build. Tracking buffers stay on MPS when their combined size is
at most 256 MiB. Larger buffers are staged in CPU memory. Video decoding and
viewer export use the host CPU.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[example]'
python -c "import torch; assert torch.backends.mps.is_available()"
python example.py examples/figure_skating.mp4 --device mps \
  --weights yyfz233/Pi4 --max-size 224 --max-frames 2 --query-frame 0 \
  --output outputs/mac-skating --viewer
python viewer/serve.py outputs/mac-skating
```

The example downloads the checkpoint on first use. To run offline, pass a
local file with `--weights /path/to/model.safetensors`. Start with a short
sequence, then raise `--max-frames` and `--max-size` as your unified memory
allows. The CUDA memory and timing tables below were measured on an A800.
For the Python API, use `.to("mps")` for both the model and the input tensor.

### NVIDIA CUDA

The recommended CUDA setup is Linux
x86-64, Python 3.11, an NVIDIA GPU with BF16 support, and a CUDA 12.6-compatible
NVIDIA driver. Install Pi4 inside an isolated environment rather than your
system Python or an unrelated project's environment. Git is required to install
the pinned FlexGEMM dependency, and Triton's runtime compilation needs a C
compiler such as GCC.

### Create an Isolated Environment

Use Python 3.11 for the recommended environment:

```bash
conda create -n pi4 python=3.11 -y
conda activate pi4
```

Keep the `pi4` environment activated for all subsequent installation and
inference commands.

### Install PyTorch and Pi4

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-cu126.txt
```

`requirements-cu126.txt` selects PyTorch 2.12.1's CUDA 12.6 build, Triton 3.7.1,
the tested versions of the model and image/video dependencies, and an editable
installation of `.[example]`. The example extra installs NumPy, Pillow, and
OpenCV for image/video loading and export. The CUDA runtime comes
with the PyTorch dependencies; a separate CUDA toolkit is not needed for the
standard installation.

For a different CUDA build, install **PyTorch 2.12.1** using the
[official previous-version commands](https://pytorch.org/get-started/previous-versions/#v2121),
then run `python -m pip install -e '.[example]'`. Keep Triton at **3.7.1**;
do not combine these kernels with an older PyTorch/Triton pair. Other CUDA
builds need their own forward-pass validation on the target hardware.

### Check the Installation

```bash
python -m pip check
python -c "import torch, triton; from pi4 import Pi4; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Triton:', triton.__version__); assert torch.cuda.is_available(), 'CUDA is not available'"
```

These checks verify dependency metadata, imports, and CUDA availability. Run
the quickstart below to check weight loading and a complete forward pass.
Do not bypass dependency conflicts with `--no-deps` or force-install mismatched
PyTorch and Triton versions.

The model dependencies are PyTorch, FlexGEMM/Triton, Safetensors, and Hugging Face
Hub. Image/video dependencies belong to the example extra and are not imported
by the model. FlexGEMM is pinned to revision
`b2fadb29d41846c7981ade6801ffc689fae119cf`.

This environment was validated from an isolated Python 3.11.15 installation on
an A800 80 GB: dependency checks, strict checkpoint loading from `yyfz233/Pi4`,
geometry-only inference, dense tracking with query frames 0 and 1, and viewer
exports. The sequence-length measurements below use this environment.

## Python

```python
import torch
from pi4 import Pi4

model = Pi4.from_pretrained("yyfz233/Pi4").cuda().eval()
images = torch.rand(1, 3, 3, 256, 384, device="cuda")

geometry = model(images)
prediction = model(images, query_frame=0)
```

Use your own RGB images in `[0, 1]`, shaped `[B, T, 3, H, W]`. Height and width
must be positive multiples of 16. Normalization and precision handling are
internal. Tracking requires at least two images. A scalar query index applies to
the whole batch; an integer tensor shaped `[B]` selects one query per sequence.
Images are jointly processed, not silently split into temporal windows.
Independent spatial operators use small frame batches internally. Tracking
features and recurrent state are staged in CPU RAM on CUDA and for larger MPS
sequences. Temporal attention sees the complete sequence.

`from_pretrained` accepts a local Safetensors file, a local directory containing
`model.safetensors`, or a Hugging Face repository ID with optional `revision=`.
Loading is always strict. No other pretrained checkpoints or configuration files
are needed. The default is [yyfz233/Pi4](https://huggingface.co/yyfz233/Pi4),
which is downloaded and cached automatically, including when calling
`Pi4.from_pretrained()` without arguments.

If the automatic download is slow, [download model.safetensors manually](https://huggingface.co/yyfz233/Pi4/resolve/main/model.safetensors)
and load it with `Pi4.from_pretrained("/path/to/model.safetensors")`, or pass
`--weights /path/to/model.safetensors` to `example.py`. A local directory
containing `model.safetensors` also works.

## Package Layout

```text
pi4/
├── __init__.py          # Public entry point: from pi4 import Pi4
├── models/
│   ├── __init__.py
│   ├── pi4.py           # Model assembly, weight loading, and joint inference
│   ├── encoder.py       # Shared image encoder
│   ├── decoder.py       # Geometry and motion token decoders
│   ├── layers.py        # Shared attention and convolution building blocks
│   ├── refiner.py       # Sparse depth refinement
│   └── heads/
│       ├── __init__.py
│       ├── point.py     # Pointmap and geometry-confidence heads
│       ├── camera.py    # Camera-pose head
│       ├── metric.py    # Metric-scale head
│       ├── tracking.py  # Dense tracking branch
│       └── motion.py    # Moving-object mask branch and its image encoder
├── geometry.py          # Geometry utilities for result export
├── postprocess.py       # Non-mutating display-confidence postprocessing
├── io.py                # Image/video loading and array/file serialization
└── visualization.py     # Preview/viewer export and display-mask policies
```

Model components live under `pi4.models`; input/output helpers stay outside the
model package. The supported entry point remains `from pi4 import Pi4`.
This organization does not change the model's tensor outputs or Safetensors
state-dict keys, so existing `model.safetensors` checkpoints load unchanged.

`pi4.visualization.save_visualization(...)` is the preview/viewer export entry
point. Display thresholds, sampling budgets, and motion-mask margins are defined
at the top of that module. Native masks, expanded background-hiding masks, and
eroded trajectory-selection masks are separate helpers; none modifies the model
predictions. `pi4.io` handles loading and serialization, and `viewer/` handles
browser rendering and interaction. The existing `pi4.io.save_results(...)`
entry point remains a compatibility wrapper with the same arguments and outputs.

## Outputs

| Key | Shape | Meaning |
| --- | --- | --- |
| `local_points` | `[B,T,H,W,3]` | Refined XYZ in each camera's coordinates, in the model's native scale |
| `camera_poses` | `[B,T,4,4]` | Camera-to-world poses in the same native scale |
| `metric_scale` | `[B]` | Positive sequence-level conversion to predicted metric units |
| `confidence` | `[B,T,H,W]` | Geometry confidence probability |
| `tracks_2d` | `[B,T,H,W,2]` | Dense target-image XY for each query-view pixel; integer-centered pixels |
| `tracks_3d` | `[B,T,H,W,3]` | Correspondences lifted onto reconstructed surfaces in metric world coordinates |
| `visibility` | `[B,T,H,W]` | Tracking visibility probability |
| `tracking_confidence` | `[B,T,H,W]` | Tracking accuracy probability, distinct from geometry confidence |
| `dynamic_mask` | `[B,T,H,W]` | Query-independent moving-object probability with sequence-level motion semantics |
| `query_frame` | `[B]` | Query frame indices |

The last six keys are returned only when `query_frame` is supplied. For a local
point `P`, camera pose `(R,t)`, and scale `m`, metric world coordinates are
`m * (R @ P + t)`. Camera translations must be scaled together with points.
World tracks sample the target-view local pointmap and apply that same transform;
they are not a separate learned 3D trajectory prediction. Out-of-image world
tracks are NaN. Visibility is returned separately, not used to invent positions
through occlusions. Moving-object masks do not gate geometry or tracking.

`dynamic_mask` describes sequence-level object motion: an object that moves
within the sequence remains positive in every frame where it appears, including
during pauses. It does not indicate whether a pixel is moving at that instant.
The mask prediction is independent of the query-view selection; this API returns
it together with the tracking outputs when `query_frame` is supplied.

## Quickstart

All three RGB input videos are included in `examples/`. The commands below load
`yyfz233/Pi4` automatically to generate geometry, motion masks, and tracks:

### Figure Skating

```bash
python example.py examples/figure_skating.mp4 --output outputs/quickstart --query-frame 0 --stride 1 --max-frames 150 --viewer
python viewer/serve.py outputs/quickstart --port 8080
```

### Indoor Karting

```bash
python example.py examples/kart.mp4 --output outputs/kart --query-frame 0 --stride 2 --viewer
python viewer/serve.py outputs/kart --port 8081
```

### Cycling

```bash
python example.py examples/cycling.mp4 --output outputs/cycling --query-frame 0 --stride 2 --viewer
python viewer/serve.py outputs/cycling --port 8082
```

The CLI defaults to **stride 1 with no frame cap**, at a maximum image
side of 512 pixels. Figure Skating is a long 60-second clip, so its quickstart
sets the optional `--max-frames 150` limit to reduce runtime and GPU memory use.
Kart and Cycling use stride 2 without a frame cap to cover their complete clips.
The bundled MP4s are already stored at their
inference resolution, with all source frames retained. Each command processes
its sampled input in one joint forward:

| Demo | Source video | Sampled input | Coverage |
| --- | --- | --- | --- |
| Figure Skating (`figure_skating.mp4`) | 750 frames, 12.5 fps | Stride 1, capped at 150 frames at 512×288 | First approximately 12 seconds of the 60-second clip |
| Kart (`kart.mp4`) | 300 frames, approximately 15 fps | Stride 2, 150 frames at 512×288 | Entire approximately 20-second clip |
| Cycling (`cycling.mp4`) | 356 frames, 30 fps | Stride 2, 178 frames at 288×512 | Entire approximately 11.87-second clip |

Sampling preserves source timestamps; the Figure Skating quickstart is **not** a subsampled
version of the complete 750-frame video. `--max-frames` caps the number of frames
**after** sampling, not the number of source frames read. Omit it to use all
sampled frames. The startup message
states the loaded frame count, frame cap, and stride. These settings apply only
to this example; `Pi4.forward()` processes every frame supplied by the caller.
The videos are inference inputs, not precomputed website results. See
`examples/README.md` for clip details.

For smaller GPUs, set `--max-frames` to limit memory use. Measurements with tracking enabled,
512×288 Figure Skating input, and **stride 1** on an A800 80 GB, Python 3.11.15,
PyTorch 2.12.1 + CUDA 12.6, and Triton 3.7.1:

| Input frames | Example arguments | Peak GPU memory | Forward time |
| --- | --- | --- | --- |
| 150 | `--stride 1 --max-frames 150` | 12.03 GiB | 48.7 seconds |
| 75 | `--stride 1 --max-frames 75` | 7.26 GiB | 14.1 seconds |
| 50 | `--stride 1 --max-frames 50` | 5.22 GiB | 8.6 seconds |
| 25 | `--stride 1 --max-frames 25` | 4.13 GiB | 4.1 seconds |

Peak process GPU memory is sampled every 0.25 seconds with `nvidia-smi`.
Forward times exclude weight loading and result export, and follow a 25-frame
warm-up; first use can take longer while Triton compiles and FlexGEMM tunes
kernels. All configurations process the entire sampled sequence in one forward.

These are measurements, not guarantees for other hardware or inputs. The
150-frame default has not been validated on an actual 8 GB GPU. There is no
automatic OOM retry, resolution reduction, or hidden temporal windowing.

## Reproduce the Project-Page Demo

**750 frames · 60 seconds · one forward · joint geometry and dense tracking.**

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python example.py examples/figure_skating.mp4 --output outputs/figure_skating_full --query-frame 0 --stride 1 --viewer
python viewer/serve.py outputs/figure_skating_full --port 8080
```

This reproduces the full Figure Skating demo from the paper and project page:
all 750 frames at 512×288, with stride 1, no frame cap, and no temporal windows. With
the allocator setting in the command, it measured **33.99 GiB peak process GPU
memory** (27.61 GiB PyTorch-allocated) and **223.7 seconds per forward** on the
A800 in the environment above, after a 25-frame warm-up. Budget at least a
40 GB-class GPU; 48 GB or more leaves additional headroom. These are
measurements, not minimum-hardware guarantees.

`PYTORCH_ALLOC_CONF=expandable_segments:True` reduces reserved-memory overhead
for long sequences; it does not change frame sampling or model computation.
Without it, the same 750-frame forward measured **50.24 GiB peak process GPU
memory**. Use the setting for this example rather than relying on the default
allocator to fit a 40–48 GB GPU. Long sequences also use CPU RAM for tracking
state. Viewing an already-exported demo needs only the browser and the
lightweight CPU server, not CUDA or model weights.

## Example Outputs

The example is just image loading, `model(images, query_frame=...)`, and
`save_visualization(...)`. Display/export policies live in
`pi4/visualization.py`; they do not change the model's returned tensors. Omit
`--query-frame` for geometry only. With tracking,
the query index refers to the sequence **after** input frame sampling.
`--device auto` is the default and selects CUDA first, then MPS.
`--max-frames` is optional and defaults to no cap; `--stride` defaults to 1.
Images are RGB uint8 `[T,H,W,3]` before conversion to the model's floating-point
`[1,T,3,H,W]` input.

```python
from pi4.visualization import save_visualization

scene = save_visualization("output", frames, prediction, metadata, viewer=True)
```

These are sampled visualization artifacts, not a dense raw-prediction archive.
Use the tensors returned by `model(...)` for evaluation or raw-result storage.

```text
output/
├── pc_preview.ply     # Binary little-endian XYZ float32 + RGB uint8
├── scene.json         # Metric camera-to-world poses, source/timing, array layouts
├── tracks.bin         # Only with tracking; one shared set of <=1,024 query tracks
├── tracks.ply         # Only with tracking, without --viewer; colored trajectory points
├── frames/*.jpg       # Only with --viewer; model-input RGB frames
└── points/*.bin       # Only with --viewer; eight-frame point chunks
```

The quick PLY retains finite points with geometry confidence **>= 0.05** and
removes depth-normal edges on the original per-frame pointmaps. An edge requires
both a >3% relative depth jump and a >5-degree local normal discontinuity in a
3x3 neighborhood. The remaining world XYZ and matching RGB are flattened in
frame/raster order and selected with the same `linspace` indices. The entire
file, including its header, is at most **10,000,000 bytes**. These fixed preview
settings are recorded in `scene.json`; they are not architecture options.

Without `--viewer`, tracking also writes `tracks.ply`: a static colored point
cloud of trajectory samples, in the same metric world coordinates as the scene.
It uses up to 1,024 queries sampled with `linspace` inside the eroded query-frame
motion mask. Each observation must have finite XY/XYZ, **visibility >= 0.5**,
**tracking confidence >= 0.5**, and a target-image position inside the eroded
target-frame motion mask. Masks use the model's native dynamic probability
>= 0.5, a fixed 5x5 erosion (two pixels), and nearest-pixel sampling. Empty masks
produce an empty PLY, never background tracks. No auxiliary mask model is used.

Colors follow the viewer's continuous 2D-query-position palette, not image RGB
or time: each query keeps one color across all frames. The PLY contains points,
not edges connecting observations through occlusions. Its vertices are stored
in frame/query order; `scene.json["tracks"]["preview"]` records the filters,
query count, and palette. This preview can use different queries from the raw
`tracks.bin`, whose arrays, scores, and validity rule are unchanged. With
`--viewer`, the interactive tracks replace this extra static preview.

All exported XYZ and camera translations are in the same predicted metric world
coordinates. `scene.json` stores image size as `[width,height]`, original source
frame indices and video timestamps when available. No dense pointmaps, full or
per-frame PLYs, confidence images, or checkpoint copies are written.

### Reading Tracks

`scene.json["tracks"]` describes `tracks.bin`; it is `null` for geometry only.
Every binary array has an explicit little-endian dtype, shape, byte offset, and
byte length, with four-byte-aligned offsets. There is no pickle or hidden schema.

```python
import json
import numpy as np
from pathlib import Path

output = Path("outputs/tracking")
scene = json.loads((output / "scene.json").read_text())
description = scene["tracks"]
buffer = (output / description["file"]).read_bytes()
tracks = {
    name: np.ndarray(field["shape"], dtype=field["dtype"], buffer=buffer, offset=field["offset"])
    for name, field in description["arrays"].items()
}
world_xyz = tracks["world_xyz"]  # [T,Q,3], predicted metric world coordinates
```

The same query order is used in every frame: `query_uv [Q,2]` is integer-centered
query-image XY, `query_rgb [Q,3]` is uint8 RGB, `pixel_xy [T,Q,2]` is target-image
XY, and `world_xyz [T,Q,3]` is metric world XYZ. `visibility` and `confidence`
are raw float32 tracking probabilities `[T,Q]`; neither is geometry confidence.
`valid [T,Q]` is uint8, requiring finite XY/XYZ and visibility >= 0.5. Viewer
exports additionally apply the motion-mask rule below. Invalid
observations retain their original values and break displayed trails; they are
not interpolated. Query candidates with finite query-frame XYZ are selected by
`linspace`, once per export; viewer candidates must also lie inside the eroded
query-frame motion mask. Consequently normal and viewer exports can select
different queries. Their coordinates and probabilities are never recomputed.

## Lightweight Viewer

Export with `--viewer`, then start the local server:

```bash
python example.py examples/figure_skating.mp4 \
  --output outputs/demo --query-frame 0 --max-frames 150 --viewer
python viewer/serve.py outputs/demo --port 8080
```

Open the printed address in your browser. For remote machines, forward port 8080.
Viewing an existing export needs no GPU or extra Python packages.

- **Navigate:** drag to rotate, right-drag to pan, scroll to zoom; double-click
  to set the rotation center.
- **Playback:** play, pause, scrub the RGB filmstrip, choose 0.5×/1×/2× speed,
  or follow the predicted camera. Clicking or dragging the filmstrip and clicking
  the RGB inset smoothly enter the selected camera. Scrubbing pauses playback
  while dragging and resumes it only if it was already playing. Manual scene
  navigation cancels follow; reduced-motion preferences disable the transition.
- **Display:** adjust point size, opacity, confidence filters, motion visibility,
  camera frustums, and track trails. Fit, reset, and fullscreen controls sit at
  the top right; advanced geometry/trajectory filters are collapsible.
- **Motion visibility:** hiding dynamic regions uses each frame's predicted
  dynamic probability >= 0.5, dilated outward by **5 pixels** with an 11x11
  square kernel at the model's resized image resolution. `Current motion +
  background` hides this expanded region in noncurrent frames; `Static
  background only` hides it in every frame. `Current motion only` still selects
  the current frame's native, undilated mask. This display-only dilation does
  not change model predictions or the separate 5x5 erosion used for tracks.
  Older exports without dilation metadata retain their original masks;
  re-export to apply the margin.
  Point-bundle `motion` values are uint8 bitfields: bit 0 is the native mask,
  and bit 1 is the dilated mask. `scene.json` records this encoding and the
  dilation radius; legacy binary masks remain supported.
- **Point budget:** the slider defaults to up to **2,000,000** points, using
  deterministic `linspace` indices over loaded geometry. Lower limits reduce
  the actual indexed GPU draw count; confidence/motion filters can reduce the
  visible count further. The existing 40 MB export cap provides approximately
  two million preview points across all frames. Small scenes have fewer points.
  This is display-only and never changes model predictions or export files.
