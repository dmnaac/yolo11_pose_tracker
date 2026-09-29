#!/usr/bin/env python3
"""YOLO11 pose-estimation person tracker node for ROS 2 Humble.

Subscribes to a camera image topic, runs YOLO11 pose tracking (persons only,
persistent track IDs via the chosen tracker), and publishes the track ID,
bounding box and 17 COCO keypoints per person. The inference device
('cpu', 'cuda:0', ...) is a runtime parameter.
"""

import numpy as np
import rclpy
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

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.conf = self.get_parameter('conf_threshold').value
        self.tracker = self.get_parameter('tracker').value
        self.publish_annotated = self.get_parameter('publish_annotated').value
        self.kp_conf = self.get_parameter('keypoint_conf_threshold').value

        # ---------------- Model ----------------
        self.get_logger().info(
            f'Loading YOLO pose model "{model_path}" on device "{self.device}" '
            f'with tracker "{self.tracker}"')
        self.model = YOLO(model_path)
        # Warm-up so the first real frame is not slowed by init/graph build.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.track(dummy, persist=True, device=self.device,
                         classes=[0], verbose=False)

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
            + (' and "/pose_tracker/annotated_image"' if self.publish_annotated else ''))

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

            for i, track_id in enumerate(ids):
                person = PersonPose()
                person.track_id = int(track_id)
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
        import cv2
        img = frame.copy()
        for person in persons_msg.persons:
            x1, y1, x2, y2 = [int(v) for v in person.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, f'ID {person.track_id}', (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

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
                    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
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
