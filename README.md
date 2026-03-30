# 🤖 Open-Vocabulary 3D Scene Understanding
### OAK-D Pro Wide + Jetson Orin Nano — Perception & Spatial Reasoning System

> A hybrid pipeline combining **open-vocabulary object detection** (no fixed class labels) with **metric depth estimation** to produce spatially-grounded, semantically-rich scene representations — the perceptual foundation of autonomous navigation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What Problem This Solves](#what-problem-this-solves)
3. [Hardware Requirements](#hardware-requirements)
4. [Architecture Overview](#architecture-overview)
5. [Technical Deep Dive](#technical-deep-dive)
   - [Stage 1 — Stereo Depth & Point Cloud](#stage-1--stereo-depth--point-cloud)
   - [Stage 2 — Open-Vocabulary Detection](#stage-2--open-vocabulary-detection)
   - [Stage 3 — 3D Localization & Fusion](#stage-3--3d-localization--fusion)
   - [Stage 4 — VLM Scene Reasoning](#stage-4--vlm-scene-reasoning)
   - [Stage 5 — Scene Graph Output](#stage-5--scene-graph-output)
6. [Full Pipeline Diagram](#full-pipeline-diagram)
7. [Software Stack](#software-stack)
8. [Algorithms Reference](#algorithms-reference)
9. [Roadmap](#roadmap)
10. [Resources & Papers](#resources--papers)
11. [Known Constraints & Tradeoffs](#known-constraints--tradeoffs)

---

## Project Overview

This project builds a **static perceptual system** for robotics research. Given a live RGB-D stream from an OAK-D Pro Wide camera, the system:

- Detects **any object described in natural language** (no fixed class list)
- Estimates the **metric 3D position** of each detected object in the camera frame
- Generates a **relational scene graph**: "the cup is 0.8m ahead, 15° left; the chair is 2.1m ahead"
- Uses a **Vision-Language Model (VLM)** to produce open-ended semantic descriptions of the scene as a whole

The key scientific novelty here is **fusing open-vocabulary detection (semantic) with stereo depth (geometric)**. Classical pipelines use fixed-class detectors (YOLO) that can only find what they were trained on. This pipeline removes that constraint entirely.

---

## What Problem This Solves

Classical object detection operates in a **closed-vocabulary** paradigm: you train on a fixed set of N classes and the model can only detect those. This is fundamentally limiting for robots that must navigate arbitrary real-world environments.

**Open-vocabulary detection** uses a joint vision-language embedding space (typically trained via contrastive learning, e.g. CLIP) where the distance between an image region and a text embedding determines whether that region matches the query. This means:

- You describe what you're looking for in natural language at inference time
- The model generalizes to objects it has never seen as a discrete class
- No retraining needed when the environment or task changes

Fusing this with metric depth (from stereo vision) produces not just "what is here" but "what is here, and exactly where in 3D space."

---

## Hardware Requirements

| Component | Spec | Role |
|---|---|---|
| **OAK-D Pro Wide** | IMX378 RGB + OV9282 stereo mono (wide FOV), MyriadX VPU, IR dot projector | Stereo depth, RGB stream, on-device inference |
| **Jetson Orin Nano 8GB** | 6-core ARM Cortex-A78AE, 1024-core Ampere GPU, 8GB unified memory | VLM inference, point cloud processing, scene graph |
| **USB 3.1 cable + powered hub** | Required for stable power delivery | Prevents xusb-tegra disconnect bug |
| **NVMe SSD (≥256GB)** | Via M.2 slot | Model storage (VLM weights ~6-8GB), dataset logging |
| **Active cooling** | Fan + heatsink | Sustained GPU workloads on Jetson |

> ⚠️ **Known issue**: The Jetson Orin's `xusb-tegra` USB driver has a stability bug that can cause OAK-D disconnections after several hours. **Mitigation**: use an externally powered USB hub between the OAK-D and the Jetson. This decouples power delivery from the host controller.

---

## Architecture Overview

The system is divided into two computational planes:

```
┌─────────────────────────────────────────────────┐
│               OAK-D Pro Wide (MyriadX VPU)      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ RGB 4K   │  │ Stereo   │  │ Depth Engine  │  │
│  │ Camera   │  │ Mono ×2  │  │ (SGM/SGBM)    │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       │              │               │            │
│       └──────────────┴───────────────┘            │
│                        │                          │
│              DepthAI Pipeline (on-device)         │
│         [Lightweight YOLO / MobileNet optional]   │
└────────────────────────┬────────────────────────┘
                         │ USB 3.1 (synchronized RGB+Depth frames)
                         ▼
┌─────────────────────────────────────────────────┐
│              Jetson Orin Nano (Host)             │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │           ROS2 Humble Node Graph         │   │
│  │                                          │   │
│  │  /rgb/image_raw ──────────────────────┐  │   │
│  │  /stereo/depth ────────────────────┐  │  │   │
│  │  /stereo/pointcloud ────────────┐  │  │  │   │
│  │                                 │  │  │  │   │
│  │  ┌─────────────────┐  ┌──────┐  │  │  │  │   │
│  │  │ OWL-ViT / GD    │◄─┘  └──┘  │  │  │  │   │
│  │  │ (open-vocab det)│            │  │  │  │   │
│  │  └────────┬────────┘            │  │  │  │   │
│  │           │ 2D bboxes           │  │  │  │   │
│  │  ┌────────▼────────┐            │  │  │  │   │
│  │  │ 3D Localization │◄───────────┘  │  │  │   │
│  │  │ (depth backproj)│               │  │  │   │
│  │  └────────┬────────┘               │  │  │   │
│  │           │ 3D object poses        │  │  │   │
│  │  ┌────────▼────────┐               │  │  │   │
│  │  │   Scene Graph   │               │  │  │   │
│  │  │   Builder       │               │  │  │   │
│  │  └────────┬────────┘               │  │  │   │
│  │           │ structured graph       │  │  │   │
│  │  ┌────────▼────────┐               │  │  │   │
│  │  │  VLM Reasoner   │◄──────────────┘  │  │   │
│  │  │ (Qwen2.5-VL-3B) │  RGB frame       │  │   │
│  │  └────────┬────────┘                  │  │   │
│  │           │ natural language output   │  │   │
│  │  ┌────────▼────────────────────────┐  │  │   │
│  │  │        Output Interface          │  │  │   │
│  │  │  JSON scene graph | ROS topics   │  │  │   │
│  │  └──────────────────────────────────┘  │  │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## Technical Deep Dive

### Stage 1 — Stereo Depth & Point Cloud

**Goal**: Obtain a calibrated, metric depth map and a 3D point cloud from the OAK-D stereo pair.

#### How Stereo Depth Works

The OAK-D Pro Wide has two monochrome cameras separated by a known **baseline** `b`. For any pixel in the left image that matches a pixel in the right image, the horizontal displacement is called **disparity** `d`. Depth `Z` is then:

```
Z = (f × b) / d
```

Where `f` is the focal length in pixels. This is the **stereo triangulation** equation — the geometric inverse of projection. The OAK's MyriadX VPU runs **Semi-Global Block Matching (SGBM)** on-device to compute this disparity map at up to 30fps without touching the Jetson's CPU.

The **Pro** variant adds an **IR dot projector** (active stereo). This matters because passive stereo fails on textureless surfaces (white walls, floors). The IR pattern artificially adds texture in the IR band visible to the mono cameras, making depth robust in exactly those cases.

#### Point Cloud Generation

With a depth map `Z(u,v)` and the camera intrinsic matrix `K`:

```
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = Z(u,v)
```

This **back-projection** converts every pixel into a 3D point. The result is a dense point cloud in the **camera coordinate frame** (origin at the left camera, Z forward, X right, Y down).

#### Tools
- `depthai` Python SDK — pipeline configuration and frame retrieval
- `depthai-ros` — publishes `/stereo/depth`, `/stereo/pointcloud`, `/rgb/image_raw` as ROS2 topics
- `open3d` — point cloud visualization, filtering, voxel downsampling

#### Key Config Parameters
```python
# depthai pipeline setup
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB
stereo.setSubpixel(True)          # subpixel disparity for higher accuracy
stereo.setExtendedDisparity(False) # enables close-range detection (<20cm)
stereo.setLeftRightCheck(True)    # filters occlusion artifacts
```

---

### Stage 2 — Open-Vocabulary Detection

**Goal**: Detect objects described by arbitrary text queries in the RGB frame.

#### The Core Idea: Vision-Language Embedding Space

Models like **OWL-ViT** (Google) and **Grounding DINO** (IDEA Research) learn a joint embedding space where images and text live in the same high-dimensional vector space. Contrastive pretraining (CLIP-style) teaches the model to minimize the distance between matching image-text pairs and maximize it for non-matching ones.

At inference:
1. You provide text queries: `["red mug", "wooden chair", "laptop"]`
2. The model extracts visual features from image regions (via Vision Transformer patches)
3. It computes the **cosine similarity** between each region embedding and each text embedding
4. Regions exceeding a similarity threshold are returned as detections

This is fundamentally different from YOLO, where output classes are hard-coded weights. Here the "class" is a dynamic text embedding computed at runtime.

#### Model Comparison

| Model | Architecture | Speed (Jetson Orin Nano) | Accuracy | Notes |
|---|---|---|---|---|
| **OWL-ViT v2** | ViT-B/16 backbone + CLIP | ~3-5 fps | Good | Google, simpler to deploy |
| **Grounding DINO** | Swin-T + BERT fusion | ~1-3 fps | Better | Better for complex queries |
| **YOLO-World** | YOLOv8 + CLIP text encoder | ~8-15 fps | Good | Best speed/accuracy tradeoff |
| **Grounding DINO 1.5 Edge** | Distilled | ~5-8 fps | Good | Optimized for edge |

**Recommended starting point**: **YOLO-World** for real-time (~10fps) or **Grounding DINO** for accuracy-first experiments.

#### TensorRT Optimization (Critical for Jetson)

Raw PyTorch models are not efficient on Jetson. TensorRT compiles models into optimized execution engines:

```bash
# Export YOLO-World to TensorRT
yolo export model=yolov8s-world.pt format=engine device=0 half=True
```

TensorRT applies: layer fusion, INT8/FP16 quantization, kernel auto-tuning. This typically yields 3-5× speedup on Jetson's Ampere GPU.

---

### Stage 3 — 3D Localization & Fusion

**Goal**: Given 2D bounding boxes from Stage 2 and the depth map from Stage 1, compute the metric 3D position of each detected object.

#### Back-projection with ROI Depth

For each detected bounding box `(x1, y1, x2, y2)`:

1. Extract the **depth ROI**: the sub-region of the depth map corresponding to the bounding box
2. Compute a **robust depth estimate** for that region. Do NOT use the mean — depth maps have noise and occlusion artifacts at object edges. Use the **median of the central 50% of pixels** (crop the border pixels of the ROI to avoid background bleed):

```python
def get_robust_depth(depth_map, bbox, crop_ratio=0.5):
    x1, y1, x2, y2 = bbox
    # Crop to central region to avoid edge occlusion artifacts
    dx = int((x2 - x1) * (1 - crop_ratio) / 2)
    dy = int((y2 - y1) * (1 - crop_ratio) / 2)
    roi = depth_map[y1+dy:y2-dy, x1+dx:x2-dx]
    valid = roi[roi > 0]  # filter invalid (0) depth values
    return np.median(valid) if len(valid) > 0 else None
```

3. Back-project the bounding box center `(cx, cy)` and the estimated depth `Z` into 3D:

```python
def backproject_to_3d(cx, cy, Z, K):
    # K = [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]
    X = (cx - K[0,2]) * Z / K[0,0]
    Y = (cy - K[1,2]) * Z / K[1,1]
    return np.array([X, Y, Z])  # meters, camera frame
```

4. Express position as **polar coordinates** for human-readable output:

```python
distance = np.linalg.norm([X, Z])           # horizontal distance
angle_deg = np.degrees(np.arctan2(X, Z))    # positive = right
```

#### Point Cloud Segmentation (Advanced)

For better accuracy, instead of using the bounding box center, you can project the bounding box into the point cloud and fit a **plane or centroid** to the returned 3D points. This is more robust for large or irregular objects.

---

### Stage 4 — VLM Scene Reasoning

**Goal**: Use a Vision-Language Model to provide holistic scene understanding beyond what structured detection can offer.

#### Role in the Pipeline

Structured detection (Stage 2-3) gives precise 3D positions but limited semantic understanding. VLMs complement this by:
- Describing spatial relationships the detection model can't express ("the chair is partially occluded by the table")
- Answering open-ended questions about the scene ("is this a safe path for a robot?")
- Generating natural language summaries of the scene graph
- Handling ambiguous visual context (clutter, unusual configurations)

#### Recommended Model: Qwen2.5-VL-3B

The Jetson Orin Nano 8GB is suitable for VLMs and LLMs up to nearly 4B parameters, such as Qwen2.5-VL-3B, VILA 1.5-3B, or Gemma-3/4B.

**Qwen2.5-VL-3B** is a strong choice because:
- 3B parameters fits in 8GB unified memory alongside the detection pipeline (with careful scheduling)
- Strong spatial reasoning capabilities
- Accepts image + text prompts natively
- Good quantization support (AWQ 4-bit reduces memory to ~2GB)

#### Memory Management Strategy

Running detection + VLM simultaneously in 8GB requires careful memory scheduling. Two viable approaches:

**Approach A — Sequential (simpler)**
```
Every frame:  run detection + 3D localization
Every N frames (e.g., N=30): run VLM on keyframe
```
This avoids simultaneous GPU memory pressure. N=30 at 10fps = VLM query every 3 seconds.

**Approach B — Threaded with memory pool**
```
Thread 1: detection pipeline (holds ~2GB GPU memory)
Thread 2: VLM (loads on demand, releases after inference)
Synchronization via frame queue
```
More complex but enables lower VLM latency.

#### VLM Prompt Engineering for Robotics

Structure your prompts to get structured, parseable outputs:

```python
SYSTEM_PROMPT = """
You are a spatial reasoning assistant for a robot. 
You receive an image and a list of detected objects with their 3D positions.
Respond in JSON with the following structure:
{
  "scene_summary": "brief description",
  "traversability": "free|cluttered|blocked",
  "notable_relations": ["object A is to the left of object B", ...],
  "recommended_attention": "object the robot should prioritize"
}
"""

def build_vlm_prompt(scene_graph):
    objects_str = "\n".join([
        f"- {obj['label']}: {obj['distance']:.2f}m away, {obj['angle']:.1f}° {'right' if obj['angle'] > 0 else 'left'}"
        for obj in scene_graph['objects']
    ])
    return f"Detected objects:\n{objects_str}\n\nDescribe this scene for robot navigation."
```

#### Deployment via NanoLLM or vLLM

```bash
# Option 1: NanoLLM (NVIDIA Jetson optimized)
pip install nano-llm
python -m nano_llm.chat --model Qwen/Qwen2.5-VL-3B-Instruct --api mlc

# Option 2: vLLM (more flexible)
pip install vllm
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --dtype half --max-model-len 4096
```

---

### Stage 5 — Scene Graph Output

**Goal**: Produce a structured, machine-readable representation of the scene that downstream navigation components can consume.

#### Scene Graph Data Structure

```python
@dataclass
class DetectedObject:
    label: str              # "red chair"
    confidence: float       # 0.0 - 1.0
    bbox_2d: Tuple[int,int,int,int]  # (x1,y1,x2,y2) in pixels
    position_3d: np.ndarray # [X, Y, Z] in meters, camera frame
    distance: float         # meters
    angle_deg: float        # degrees from optical axis (+ = right)
    depth_confidence: float # ratio of valid depth pixels in ROI

@dataclass
class SceneGraph:
    timestamp: float
    objects: List[DetectedObject]
    relations: List[Tuple[str, str, str]]  # (subject, relation, object)
    vlm_summary: Optional[str]
    traversability: Optional[str]

# Example output:
{
  "timestamp": 1711800000.123,
  "objects": [
    {"label": "wooden chair", "distance": 1.2, "angle_deg": -15.3, "position_3d": [-0.32, 0.1, 1.15]},
    {"label": "laptop", "distance": 0.8, "angle_deg": 5.1, "position_3d": [0.07, -0.2, 0.79]}
  ],
  "relations": [
    ["wooden chair", "is_left_of", "laptop"],
    ["laptop", "is_closer_than", "wooden chair"]
  ],
  "vlm_summary": "A desk workspace with a chair slightly to the left...",
  "traversability": "cluttered"
}
```

#### Relation Extraction (Rule-Based + VLM)

Spatial relations can be computed geometrically from 3D positions:

```python
def compute_relations(objects: List[DetectedObject]) -> List[Tuple]:
    relations = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j:
                continue
            # Lateral relation
            dx = a.position_3d[0] - b.position_3d[0]
            if abs(dx) > 0.15:  # threshold in meters
                rel = "is_left_of" if dx < 0 else "is_right_of"
                relations.append((a.label, rel, b.label))
            # Depth relation
            dz = a.distance - b.distance
            if abs(dz) > 0.2:
                rel = "is_closer_than" if dz < 0 else "is_farther_than"
                relations.append((a.label, rel, b.label))
    return relations
```

---

## Full Pipeline Diagram

```
OAK-D Pro Wide
│
├── [MyriadX VPU] ──► Stereo SGM ──► Disparity Map ──► Depth Map (metric, mm)
│                                                              │
└── [ISP] ──────────────────────────────► RGB Frame ──────────┤
                                                               │
                                          USB 3.1             │
                                               │               │
                                    ┌──────────▼───────────────▼──────────┐
                                    │         Jetson Orin Nano             │
                                    │                                      │
                                    │  ┌───────────────────────────────┐  │
                                    │  │ depthai-ros bridge            │  │
                                    │  │ /rgb/image_raw                │  │
                                    │  │ /stereo/depth                 │  │
                                    │  │ /stereo/pointcloud            │  │
                                    │  └──────────┬────────────────────┘  │
                                    │             │                        │
                                    │  ┌──────────▼────────────────────┐  │
                                    │  │ Open-Vocab Detector Node      │  │
                                    │  │ Model: YOLO-World / GDino     │  │
                                    │  │ Input: RGB + text_queries[]   │  │
                                    │  │ Output: List[BBox2D, label,   │  │
                                    │  │         confidence]           │  │
                                    │  └──────────┬────────────────────┘  │
                                    │             │                        │
                                    │  ┌──────────▼────────────────────┐  │
                                    │  │ 3D Localization Node          │  │
                                    │  │ - ROI depth extraction        │  │
                                    │  │ - Median depth estimation     │  │
                                    │  │ - Back-projection with K      │  │
                                    │  │ Output: List[Object3D]        │  │
                                    │  └──────────┬────────────────────┘  │
                                    │             │                        │
                                    │  ┌──────────▼────────────────────┐  │
                                    │  │ Scene Graph Builder Node      │  │
                                    │  │ - Geometric relation mining   │  │
                                    │  │ - Graph assembly              │  │
                                    │  │ Output: SceneGraph            │  │
                                    │  └──────────┬────────────────────┘  │
                                    │             │                        │
                                    │  ┌──────────▼────────────────────┐  │
                                    │  │ VLM Reasoner Node (async)     │  │
                                    │  │ Model: Qwen2.5-VL-3B (AWQ)   │  │
                                    │  │ Input: keyframe RGB +         │  │
                                    │  │        scene graph JSON       │  │
                                    │  │ Output: NL summary +          │  │
                                    │  │         traversability        │  │
                                    │  └──────────┬────────────────────┘  │
                                    │             │                        │
                                    │  ┌──────────▼────────────────────┐  │
                                    │  │ Output Publisher              │  │
                                    │  │ /scene_graph (JSON)           │  │
                                    │  │ /scene_description (String)   │  │
                                    │  │ /detected_objects (MarkerArray│  │
                                    │  │  for RViz visualization)      │  │
                                    │  └───────────────────────────────┘  │
                                    └──────────────────────────────────────┘
```

---

## Software Stack

### Core Dependencies

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **OS** | JetPack 6.x (Ubuntu 22.04) | 6.1+ | Jetson base OS with CUDA 12.x |
| **Middleware** | ROS2 Humble | LTS | Node graph, topic pub/sub, visualization |
| **Camera SDK** | DepthAI Python | ≥2.24 | OAK-D pipeline configuration and streaming |
| **Camera ROS** | depthai-ros | Humble branch | ROS2 bridge for OAK-D |
| **Detection** | YOLO-World or Grounding DINO | Latest | Open-vocabulary detection |
| **ML Runtime** | TensorRT 8.6+ | Bundled in JetPack | Model optimization for Jetson GPU |
| **VLM** | Qwen2.5-VL-3B-Instruct | Latest | Scene reasoning |
| **VLM Runtime** | NanoLLM or vLLM | Latest | Jetson-optimized LLM serving |
| **Point Cloud** | Open3D | 0.18+ | 3D visualization and processing |
| **Computer Vision** | OpenCV | 4.x | Image processing utilities |
| **Numerics** | NumPy | 1.26+ | Array operations |
| **Visualization** | RViz2 | Bundled with ROS2 | 3D scene visualization |

### Installation Order

```bash
# 1. Flash JetPack 6.x via SDK Manager on host machine
# 2. On Jetson:

# ROS2 Humble
sudo apt install ros-humble-desktop ros-humble-depthai-ros

# DepthAI
pip install depthai

# Grounding DINO (if chosen)
pip install groundingdino-py

# YOLO-World (if chosen)
pip install ultralytics

# Open3D (ARM64 build)
pip install open3d

# NanoLLM for VLM
pip install nano-llm

# Download VLM model
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

---

## Algorithms Reference

### Stereo Vision
- **SGBM (Semi-Global Block Matching)** — Hirschmüller (2008). Runs on MyriadX VPU.
- **Active stereo** with IR structured light — extends range to textureless surfaces.

### Open-Vocabulary Detection
- **CLIP** — Radford et al. (2021). Contrastive vision-language pretraining. Foundation.
- **OWL-ViT** — Minderer et al. (2022). Open-vocabulary detection via ViT + CLIP.
- **Grounding DINO** — Liu et al. (2023). Combines DINO (object detection) with BERT text encoder via cross-modal attention.
- **YOLO-World** — Cheng et al. (2024). Integrates CLIP text embeddings into YOLOv8 neck, achieving real-time open-vocab detection.

### 3D Localization
- **Stereo triangulation** — classical multi-view geometry (Hartley & Zisserman).
- **Back-projection** — inverse of perspective projection using camera intrinsics.
- **ROI depth median** — robust statistics to handle outliers at object boundaries.

### Scene Graphs
- **IMP (Image scene graph generation)** — Yang et al. (2018).
- **3D Scene Graphs** — Armeni et al. (2019). Hierarchical 3D scene representation.
- **CONCEPTGRAPHS** — Gu et al. (2024). Open-vocabulary 3D scene graphs — directly relevant to this project.

### VLM Deployment
- **AWQ (Activation-aware Weight Quantization)** — Lin et al. (2023). 4-bit quantization with <1% accuracy loss.
- **MLC-LLM** — NVIDIA NanoLLM's backend for Jetson-optimized inference.

---

## Roadmap

### Phase 0 — Environment Setup (Week 1-2)

- [ ] Flash JetPack 6.x on Jetson Orin Nano
- [ ] Verify OAK-D Pro Wide detection via `python -c "import depthai; print(depthai.Device())"`
- [ ] Install depthai-ros and verify ROS2 topic publication
- [ ] Visualize RGB + depth in RViz2
- [ ] Benchmark USB stability (run 2+ hours, log disconnects) → configure powered hub if needed
- [ ] Download and run YOLO-World basic example on a sample image

**Milestone**: stable RGB + depth stream visible in RViz2, YOLO-World detects objects in test images.

---

### Phase 1 — Stereo Depth Baseline (Week 3-4)

- [ ] Configure DepthAI pipeline: RGB aligned to depth, subpixel enabled, LR check enabled
- [ ] Publish synchronized `(rgb, depth)` frame pairs as ROS2 topics
- [ ] Generate and visualize a 3D point cloud from depth frames using Open3D
- [ ] Measure depth accuracy with a ruler at known distances (0.5m, 1m, 2m, 3m)
- [ ] Document systematic error profile (expected: ±2-5% at 1m for OAK-D stereo)
- [ ] Apply voxel downsampling and statistical outlier removal to point cloud

**Milestone**: accurate metric depth with <5% error at 1m; clean point cloud visualized.

---

### Phase 2 — Open-Vocabulary Detection (Week 5-7)

- [ ] Deploy YOLO-World on Jetson (PyTorch baseline first, then TensorRT export)
- [ ] Benchmark FPS before/after TensorRT optimization
- [ ] Implement text query interface: define queries via YAML config file
- [ ] Visualize detections overlaid on RGB stream in real time
- [ ] Tune confidence threshold to minimize false positives in your test environment
- [ ] (Optional) Evaluate Grounding DINO on same test set for accuracy comparison

**Milestone**: open-vocabulary detector running at ≥8 FPS on Jetson with TensorRT.

---

### Phase 3 — 3D Localization Fusion (Week 8-10)

- [ ] Implement `get_robust_depth()` and `backproject_to_3d()` functions
- [ ] For each detection, compute 3D position and express as (distance, angle)
- [ ] Validate: place known objects at measured distances, compare system output vs. ground truth
- [ ] Implement geometric relation extraction (`is_left_of`, `is_closer_than`, etc.)
- [ ] Build `SceneGraph` data structure and JSON serializer
- [ ] Publish scene graph as ROS2 topic (`/scene_graph`)
- [ ] Visualize 3D object positions as RViz2 Markers with text labels

**Milestone**: scene graph JSON with correct metric distances for 3+ objects simultaneously.

---

### Phase 4 — VLM Integration (Week 11-13)

- [ ] Deploy Qwen2.5-VL-3B-Instruct-AWQ via NanoLLM on Jetson
- [ ] Measure memory footprint with and without detection pipeline running
- [ ] Implement async VLM node (keyframe-based, every N frames)
- [ ] Design and iterate on system prompt for navigation-relevant outputs
- [ ] Fuse VLM output (traversability, NL summary) into SceneGraph
- [ ] Benchmark end-to-end latency: from frame capture to VLM response

**Milestone**: VLM produces structured JSON scene descriptions from live camera feed.

---

### Phase 5 — Integration & Evaluation (Week 14-16)

- [ ] Full pipeline integration test: all stages running simultaneously
- [ ] Define and implement evaluation metrics:
  - Detection mAP on a small hand-labeled test set
  - Depth RMSE vs. ground truth measurements
  - Relation accuracy (precision/recall on a curated scene set)
  - VLM response coherence (manual evaluation rubric)
- [ ] Profile total system latency and CPU/GPU utilization
- [ ] Build a simple CLI or web dashboard to display live scene graph output
- [ ] Write documentation and record demo video

**Milestone**: complete working system with documented performance metrics.

---

### Phase 6 — Extensions (Open-ended)

- [ ] **SLAM integration**: add Spectacular AI or rtabmap for ego-motion estimation. Now objects have positions in a world frame, not just the camera frame.
- [ ] **Temporal tracking**: assign persistent IDs to objects across frames using IoU-based tracking (ByteTrack or BotSort). Detect when objects move.
- [ ] **Occupancy grid**: project depth map into a 2D bird's-eye-view occupancy grid. Run A* on it. Simulate navigation planning without a robot.
- [ ] **VLA prototype**: use the scene graph + VLM to generate navigation instructions ("move 1.5m forward, avoid the chair on the left") that could command a future mobile base.
- [ ] **Dataset collection**: log synchronized (rgb, depth, scene_graph, vlm_summary) tuples. Build a small dataset of your environment for fine-tuning or evaluation.

---

## Resources & Papers

### Foundational Papers

| Paper | Year | Why It Matters |
|---|---|---|
| [CLIP — Radford et al.](https://arxiv.org/abs/2103.00020) | 2021 | Foundation of open-vocab vision-language alignment |
| [OWL-ViT — Minderer et al.](https://arxiv.org/abs/2205.06230) | 2022 | First open-vocab detector using CLIP |
| [Grounding DINO — Liu et al.](https://arxiv.org/abs/2303.05499) | 2023 | Best open-vocab detection accuracy |
| [YOLO-World — Cheng et al.](https://arxiv.org/abs/2401.17270) | 2024 | Real-time open-vocab detection |
| [ConceptGraphs — Gu et al.](https://arxiv.org/abs/2309.16650) | 2024 | Open-vocab 3D scene graphs (closest to this project) |
| [3D Scene Graphs — Armeni et al.](https://arxiv.org/abs/1904.11482) | 2019 | Scene graph formalism for 3D spaces |
| [AWQ — Lin et al.](https://arxiv.org/abs/2306.00978) | 2023 | 4-bit VLM quantization method used for deployment |

### Documentation

| Resource | URL |
|---|---|
| Luxonis DepthAI Docs | https://docs.luxonis.com |
| depthai-ros GitHub | https://github.com/luxonis/depthai-ros |
| YOLO-World GitHub | https://github.com/AILab-CVC/YOLO-World |
| Grounding DINO GitHub | https://github.com/IDEA-Research/GroundingDINO |
| Jetson AI Lab (VLM tutorials) | https://www.jetson-ai-lab.com |
| NanoLLM | https://github.com/dusty-nv/NanoLLM |
| Qwen2.5-VL HuggingFace | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |
| Open3D | http://www.open3d.org/docs |
| ConceptGraphs Project | https://concept-graphs.github.io |

### Books

- **Multiple View Geometry in Computer Vision** — Hartley & Zisserman (2004). The definitive reference for stereo geometry and camera models.
- **Probabilistic Robotics** — Thrun, Burgard, Fox (2005). Occupancy grids, SLAM, sensor models.
- **Computer Vision: Algorithms and Applications** — Szeliski (2022, free online). Depth, detection, 3D reconstruction.

---

## Known Constraints & Tradeoffs

### Memory Pressure (8GB Unified)

The Jetson Orin Nano's 8GB is shared between CPU and GPU. Running YOLO-World (~1.5GB) + Qwen2.5-VL-3B AWQ (~2.5GB) + ROS2 + OS overhead leaves little headroom. Solutions:
- Use asynchronous VLM inference (not every frame)
- Reduce YOLO-World input resolution (e.g., 640→320 pixels)
- Use SmolVLM-500M instead of Qwen if memory is too tight

### Depth Accuracy Limits

OAK-D stereo depth is most accurate between 0.5m and 5m. Outside this range:
- **< 0.5m**: disparity values overflow; use `setExtendedDisparity(True)` to partially mitigate
- **> 5m**: disparity values become very small (1-2 pixels); quantization error dominates. For long-range, the IR dot projector has no effect (IR power drops with distance²)

### Open-Vocab Detection Speed vs. Accuracy

Grounding DINO is more accurate but slower (~1-2 fps on Jetson even with TensorRT). For real-time use at ≥8 fps, YOLO-World is the pragmatic choice. Consider running Grounding DINO in parallel only for VLM keyframes.

### VLM Hallucinations

VLMs can confidently describe objects that aren't there, especially with low-quality or ambiguous frames. Mitigation:
- Use VLM only as a high-level interpreter; trust the structured detector for precise positions
- Filter VLM outputs against the structured detection list (if VLM mentions an object not in the detection list, flag it as unverified)
- Use the scene graph JSON in the VLM prompt to anchor its reasoning to ground-truth detections

---

*Built for robotics research. Static perceptual system — no actuators required.*
