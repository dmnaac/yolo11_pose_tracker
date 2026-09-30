# yolo11_pose_tracker

A ROS 2 Humble package for **multi-person pose tracking** with the
[Ultralytics YOLO11](https://docs.ultralytics.com/) pose-estimation model.

The node subscribes to a camera image topic, detects all persons in the frame,
assigns each of them a **persistent tracking ID**, estimates their **17 COCO
body keypoints**, and publishes the result as a structured ROS message. The
inference device (CPU or GPU) is selectable at launch time.

---

## Features

- Real-time person detection + pose estimation (YOLO11-pose, `yolo11n/s/m/l/x-pose.pt`)
- Multi-object tracking with persistent IDs (`persist=True`); tracker backend
  selectable: ByteTrack (default), BoT-SORT, or a custom tracker YAML
- **Target re-identification**: the first frame containing exactly one person
  locks that person as the target (always published as `track_id 0`). If the
  tracker loses the target, an OSNet-x0_25 appearance model re-identifies them
  when they reappear — the search never expires
- Persons only (COCO class 0) — other object classes are filtered out
- Structured output message: track ID, bounding box, keypoints with per-point
  confidence, timestamped with the source image header
- Optional annotated debug image (bounding boxes, IDs, skeleton overlay)
- One-parameter device switch: `device:=cpu` / `device:=cuda:0`
- Lightweight: single Python node + 3 custom messages, no C++ code

## Data flow

```
sensor_msgs/Image                     yolo11_pose_tracker/PersonPoseArray
/camera/image_raw  ──►  pose_tracker_node  ──►  /pose_tracker/persons
                              │
                              └──►  sensor_msgs/Image
                                    /pose_tracker/annotated_image (optional)
```

## Topics

The node subscribes to **1 topic** and publishes **2 topics** (one of them
optional):

### Subscribed

| Topic | Type | QoS | Description |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | BEST_EFFORT, KEEP_LAST depth 1 | input camera stream (BGR8). Name set by `image_topic` — node default `/camera/image_raw`, launch default `/camera/camera/color/image_raw` |

### Published

| Topic | Type | QoS | Description |
|---|---|---|---|
| `/pose_tracker/persons` | `yolo11_pose_tracker/PersonPoseArray` | RELIABLE (default), depth 10 | tracked persons per frame: `track_id` + bbox + 17 keypoints. With `enable_reid:=true` the locked target always has `track_id 0`; empty array when no one is tracked |
| `/pose_tracker/annotated_image` | `sensor_msgs/Image` | RELIABLE (default), depth 10 | debug visualization: bboxes, IDs, skeleton overlay, ReID state banner. Only published when `publish_annotated:=true` |

All published messages carry the source image's header (timestamp + frame_id).

## Messages

```
PersonPoseArray                     PersonPose                    Keypoint
├─ std_msgs/Header header           ├─ int32 track_id             ├─ float32 x
└─ PersonPose[] persons             ├─ float32[4] bbox            ├─ float32 y
                                    ├─ float32 bbox_confidence    └─ float32 confidence
                                    └─ Keypoint[] keypoints (17)
```

Keypoint indices follow the COCO layout:

| idx | name | idx | name | idx | name |
|----|------|----|------|----|------|
| 0 | nose | 6 | right_shoulder | 12 | right_hip |
| 1 | left_eye | 7 | left_elbow | 13 | left_knee |
| 2 | right_eye | 8 | right_elbow | 14 | right_knee |
| 3 | left_ear | 9 | left_wrist | 15 | left_ankle |
| 4 | right_ear | 10 | right_wrist | 16 | right_ankle |
| 5 | left_shoulder | 11 | left_hip | | |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `image_topic` | `/camera/image_raw` | camera topic to subscribe (BEST_EFFORT QoS, depth 1) |
| `model_path` | `<pkg_share>/model/yolo11n-pose.pt` | YOLO11 pose weights (bundled in `model/`) |
| `device` | `cuda:0` | inference device: `cuda:0`, `cuda:1`, ... or `cpu` |
| `conf_threshold` | `0.25` | detection confidence threshold |
| `tracker` | `<pkg_share>/model/botsort_reid.yaml` | ultralytics tracker config (bundled: `botsort_reid.yaml`, `tracktrack_reid.yaml`, or any custom YAML) |
| `publish_annotated` | `true` | publish the annotated debug image |
| `keypoint_conf_threshold` | `0.3` | min keypoint confidence for drawing the debug skeleton |
| `enable_reid` | `true` | lock the first lone person as target (ID 0) and re-identify via OSNet-x0_25 |
| `reid_model_path` | `~/.cache/torch/checkpoints/osnet_x0_25_market1501.pth` | OSNet-x0_25 weights; falls back to ImageNet-pretrained if missing |
| `reid_threshold` | `0.75` | cosine similarity threshold to re-acquire the lost target (sample photos: same person ≈1.0, different persons 0.35–0.70) |

## Target re-identification

With `enable_reid:=true` (default) the node runs a small state machine on top
of the tracker:

```
WAITING ── exactly 1 person in frame ──► TRACKING ── tracker loses ID ──► LOST
   ▲                                        │                              │
   │                                        │  OSNet-x0_25 match >=        │
   └──────────── never re-locks ────────────┴── reid_threshold ◄───────────┘
```

- **WAITING**: no target. Locking only happens on a frame with exactly one
  person (zero or 2+ persons never lock).
- **TRACKING**: the target is published with `track_id 0`; everyone else keeps
  their raw tracker ID. No ReID features are computed in this state.
- **LOST**: every visible person is embedded with OSNet-x0_25 each frame and
  compared (cosine similarity) to the target's stored feature. A match at or
  above `reid_threshold` re-acquires the target. The search never expires and
  a new target is never locked automatically.

The annotated image shows the state banner plus a red `TARGET 0` box for the
locked person (green boxes for everyone else).

## Dependencies

- ROS 2 Humble (`rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`, `rosidl_default_generators`)
- Python: `ultralytics` (pulls in PyTorch; install the CUDA build of torch for GPU)
- Python: `torchreid` + `gdown` + `tensorboard` (target re-identification;
  OSNet-x0_25 weights download once from the torchreid model zoo)

```bash
sudo apt install ros-humble-cv-bridge
pip install ultralytics          # into the python env used by ROS 2
pip install torchreid gdown tensorboard
# Optional but recommended: person-ReID-trained OSNet-x0_25 weights
# (ImageNet weights are used automatically if this file is missing)
python3 -c "import gdown; gdown.download( \
  'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs', \
  '$HOME/.cache/torch/checkpoints/osnet_x0_25_market1501.pth')"
```

## Build

```bash
cd ~/yolo_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo11_pose_tracker
source install/setup.bash
```

## Usage

```bash
# CPU, default topic
ros2 launch yolo11_pose_tracker pose_tracker.launch.py \
  image_topic:=/camera/image_raw device:=cpu

# GPU with local weights and a custom tracker config
ros2 launch yolo11_pose_tracker pose_tracker.launch.py \
  image_topic:=/camera/image_raw device:=cuda:0 \
  model_path:=~/Workspace/yolo_test/yolo11n-pose.pt \
  tracker:=~/Workspace/yolo_test/tracktrack_reid.yaml

# Inspect the tracking output
ros2 topic echo /pose_tracker/persons

# View the debug image
ros2 run rqt_image_view rqt_image_view /pose_tracker/annotated_image
```

## Notes

- With `enable_reid:=true`, `track_id 0` is reserved for the locked target;
  all other persons carry their raw tracker IDs (which start at 1).
- A dummy warm-up pass runs at node startup so the first real frame is not
  slowed by model/graph initialization.
- Track IDs persist across frames while the node runs (`persist=True`); they
  are **not** stable across node restarts.
- For moving cameras, a BoT-SORT-style config with global motion compensation
  usually holds IDs better than plain ByteTrack.

## License

Apache-2.0
