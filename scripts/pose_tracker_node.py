#!/usr/bin/env python3
"""YOLO11 pose-estimation person tracker node for ROS 2 Humble.

Subscribes to a camera image topic, runs YOLO11 pose tracking (persons only,
persistent track IDs via the chosen tracker), and publishes the track ID,
bounding box and 17 COCO keypoints per person. The inference device
('cpu', 'cuda:0', ...) is a runtime parameter.

Target re-identification (enable_reid, default on):
  - WAITING: the first frame that contains EXACTLY ONE person locks that
    person as the target. Frames with zero or 2+ persons never lock.
  - TRACKING: while the tracker keeps the target's track ID, the target is
    published with track_id 0. No ReID features are computed (zero overhead).
  - LOST: when the tracker loses the target's ID, every visible person is
    re-embedded each frame with OSNet-x0_25 and compared against the target's
    stored feature (cosine similarity). A match >= reid_threshold re-acquires
    the target; the search NEVER expires and no new target is ever locked.
Other persons are always published with their raw tracker IDs.
"""

import os

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from ultralytics import YOLO

from yolo11_pose_tracker.msg import Keypoint, PersonPose, PersonPoseArray

# COCO skeleton (1-based pairs converted to 0-based indices) for the
# optional annotated-image output.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 1), (0, 2), (1, 3), (2, 4),           # face
]

# Default location of the person-ReID-trained OSNet-x0_25 weights
# (Market-1501; downloaded once from the torchreid model zoo).
DEFAULT_REID_WEIGHTS = os.path.expanduser(
    '~/.cache/torch/checkpoints/osnet_x0_25_market1501.pth')

# Target ReID states.
STATE_WAITING = 'WAITING'    # no target locked yet
STATE_TRACKING = 'TRACKING'  # tracker holds the target's ID
STATE_LOST = 'LOST'          # tracker lost the target; ReID search active


class PoseTrackerNode(Node):
    def __init__(self):
        super().__init__('pose_tracker_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', 'yolo11n-pose.pt')
        self.declare_parameter('device', 'cpu')          # 'cpu' or 'cuda:0' etc.
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('tracker', 'bytetrack.yaml')
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('keypoint_conf_threshold', 0.3)
        # --- Target re-identification (OSNet-x0_25) ---
        self.declare_parameter('enable_reid', True)
        self.declare_parameter('reid_model_path', DEFAULT_REID_WEIGHTS)
        self.declare_parameter('reid_threshold', 0.75)   # cosine similarity

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.conf = self.get_parameter('conf_threshold').value
        self.tracker = self.get_parameter('tracker').value
        self.publish_annotated = self.get_parameter('publish_annotated').value
        self.kp_conf = self.get_parameter('keypoint_conf_threshold').value
        self.enable_reid = self.get_parameter('enable_reid').value
        self.reid_model_path = os.path.expanduser(
            self.get_parameter('reid_model_path').value)
        self.reid_threshold = self.get_parameter('reid_threshold').value

        # ---------------- Model ----------------
        self.get_logger().info(
            f'Loading YOLO pose model "{model_path}" on device "{self.device}" '
            f'with tracker "{self.tracker}"')
        self.model = YOLO(model_path)
        # Warm-up so the first real frame is not slowed by init/graph build.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.track(dummy, persist=True, device=self.device,
                         classes=[0], verbose=False)

        # ---------------- Target ReID (OSNet-x0_25) ----------------
        self.reid_extractor = None
        self.target_track_id = None    # raw tracker ID of the locked target
        self.target_feature = None     # L2-normalized 512-d feature tensor
        self.reid_state = STATE_WAITING
        if self.enable_reid:
            try:
                from torchreid.reid.utils import FeatureExtractor
                weights = (self.reid_model_path
                           if os.path.isfile(self.reid_model_path) else '')
                if not weights:
                    self.get_logger().warn(
                        f'ReID weights not found at "{self.reid_model_path}", '
                        'falling back to ImageNet-pretrained OSNet-x0_25 '
                        '(weaker person-ReID features).')
                self.get_logger().info(
                    f'Loading OSNet-x0_25 ReID model '
                    f'({weights or "imagenet-auto"}) on "{self.device}"')
                self.reid_extractor = FeatureExtractor(
                    model_name='osnet_x0_25', model_path=weights,
                    device=self.device, verbose=False)
                # Warm-up with a dummy 256x128 person crop.
                warm = np.zeros((256, 128, 3), dtype=np.uint8)
                self._normalize(self.reid_extractor(warm))
            except Exception as e:
                self.get_logger().error(
                    f'Failed to init OSNet ReID ({e}); target re-ID disabled.')
                self.enable_reid = False

        self.bridge = CvBridge()

        # ---------------- ROS interfaces ----------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.sub = self.create_subscription(
            Image, image_topic, self.image_callback, qos)
        self.pose_pub = self.create_publisher(
            PersonPoseArray, '/pose_tracker/persons', 10)
        if self.publish_annotated:
            self.img_pub = self.create_publisher(
                Image, '/pose_tracker/annotated_image', 10)

        self.get_logger().info(
            f'Ready. Subscribing "{image_topic}", publishing '
            f'"/pose_tracker/persons"'
            + (' and "/pose_tracker/annotated_image"' if self.publish_annotated else '')
            + (f' [target ReID ON, threshold={self.reid_threshold}]'
               if self.enable_reid else ' [target ReID OFF]'))

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(feats: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(feats, p=2, dim=1)

    def _extract_features(self, frame, xyxy: np.ndarray):
        """Crop each bbox from the frame and embed with OSNet.

        Returns (kept_indices, features) where features is an (M, 512)
        L2-normalized tensor on the inference device, or ([], None) when no
        crop is usable. Input crops are converted BGR -> RGB because
        torchvision's ToPILImage treats uint8 HWC arrays as RGB.
        """
        h, w = frame.shape[:2]
        crops, kept = [], []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h))
            if x2 - x1 < 16 or y2 - y1 < 16:   # too small to embed reliably
                continue
            crop = frame[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            kept.append(i)
        if not crops:
            return [], None
        feats = self._normalize(self.reid_extractor(crops))
        return kept, feats

    def _update_target(self, frame, ids, xyxy):
        """Advance the WAITING -> TRACKING -> LOST target state machine."""
        if self.reid_state == STATE_WAITING:
            # Lock only when exactly one person is visible.
            if len(ids) == 1:
                kept, feats = self._extract_features(frame, xyxy)
                if feats is not None:
                    self.target_track_id = int(ids[kept[0]])
                    self.target_feature = feats[0]
                    self.reid_state = STATE_TRACKING
                    self.get_logger().info(
                        f'Target locked: tracker ID {self.target_track_id} '
                        '-> published as ID 0')

        elif self.reid_state == STATE_TRACKING:
            if self.target_track_id in ids:
                pass   # tracker still holds the target; zero ReID overhead
            else:
                self.reid_state = STATE_LOST
                self.get_logger().warn(
                    f'Target (tracker ID {self.target_track_id}) lost; '
                    'starting OSNet re-identification search')

        if self.reid_state == STATE_LOST:
            # Search every visible person, every frame, forever.
            kept, feats = self._extract_features(frame, xyxy)
            if feats is None:
                return
            sims = torch.mv(feats, self.target_feature)   # cosine (normalized)
            best = int(torch.argmax(sims).item())
            best_sim = float(sims[best].item())
            if best_sim >= self.reid_threshold:
                self.target_track_id = int(ids[kept[best]])
                self.reid_state = STATE_TRACKING
                self.get_logger().info(
                    f'Target re-identified: tracker ID {self.target_track_id} '
                    f'(similarity {best_sim:.3f}) -> published as ID 0')

    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        results = self.model.track(
            frame,
            persist=True,               # keep track IDs across frames
            tracker=self.tracker,
            device=self.device,
            conf=self.conf,
            classes=[0],                # COCO class 0 = person
            verbose=False,
        )
        result = results[0]

        out = PersonPoseArray()
        out.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)

        boxes = result.boxes
        kpts = result.keypoints
        if boxes is not None and boxes.id is not None and kpts is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            bconf = boxes.conf.cpu().numpy()
            kxy = kpts.xy.cpu().numpy()          # (N, 17, 2)
            kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else None

            if self.enable_reid:
                self._update_target(frame, ids, xyxy)

            for i, track_id in enumerate(ids):
                track_id = int(track_id)
                is_target = (self.enable_reid
                             and track_id == self.target_track_id)
                person = PersonPose()
                person.track_id = 0 if is_target else track_id
                person.bbox = [float(v) for v in xyxy[i]]
                person.bbox_confidence = float(bconf[i])
                for j in range(kxy.shape[1]):
                    kp = Keypoint()
                    kp.x = float(kxy[i, j, 0])
                    kp.y = float(kxy[i, j, 1])
                    kp.confidence = float(kconf[i, j]) if kconf is not None else 0.0
                    person.keypoints.append(kp)
                out.persons.append(person)

        self.pose_pub.publish(out)

        if self.publish_annotated:
            annotated = self.draw_annotations(frame, out)
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header = out.header
            self.img_pub.publish(img_msg)

    # ------------------------------------------------------------------
    def draw_annotations(self, frame, persons_msg: PersonPoseArray):
        img = frame.copy()

        # ReID status banner.
        if self.enable_reid:
            state_text = {
                STATE_WAITING: 'ReID: WAITING FOR TARGET',
                STATE_TRACKING: 'ReID: TARGET LOCKED (ID 0)',
                STATE_LOST: 'ReID: TARGET LOST - SEARCHING',
            }[self.reid_state]
            state_color = {
                STATE_WAITING: (0, 200, 255),
                STATE_TRACKING: (0, 0, 255),
                STATE_LOST: (0, 165, 255),
            }[self.reid_state]
            cv2.putText(img, state_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

        for person in persons_msg.persons:
            is_target = (self.enable_reid and person.track_id == 0)
            box_color = (0, 0, 255) if is_target else (0, 200, 0)
            label = 'TARGET 0' if is_target else f'ID {person.track_id}'

            x1, y1, x2, y2 = [int(v) for v in person.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color,
                          3 if is_target else 2)
            cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            kps = [(kp.x, kp.y, kp.confidence) for kp in person.keypoints]
            for a, b in SKELETON:
                if a < len(kps) and b < len(kps):
                    xa, ya, ca = kps[a]
                    xb, yb, cb = kps[b]
                    if ca > self.kp_conf and cb > self.kp_conf:
                        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)),
                                 (255, 128, 0), 2)
            for x, y, c in kps:
                if c > self.kp_conf:
                    cv2.circle(img, (int(x), int(y)), 3,
                               (0, 255, 255) if is_target else (0, 0, 255), -1)
        return img


def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
