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

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | subscribe | input camera stream (name set by `image_topic`) |
| `/pose_tracker/persons` | `yolo11_pose_tracker/PersonPoseArray` | publish | tracked persons: IDs + bboxes + keypoints |
| `/pose_tracker/annotated_image` | `sensor_msgs/Image` | publish | debug visualization (if `publish_annotated:=true`) |

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

## Dependencies

- ROS 2 Humble (`rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`, `rosidl_default_generators`)
- Python: `ultralytics` (pulls in PyTorch; install the CUDA build of torch for GPU)

```bash
sudo apt install ros-humble-cv-bridge
pip install ultralytics          # into the python env used by ROS 2
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

- A dummy warm-up pass runs at node startup so the first real frame is not
  slowed by model/graph initialization.
- Track IDs persist across frames while the node runs (`persist=True`); they
  are **not** stable across node restarts.
- For moving cameras, a BoT-SORT-style config with global motion compensation
  usually holds IDs better than plain ByteTrack.

## License

Apache-2.0
