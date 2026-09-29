# One tracked person: persistent track ID + bounding box + keypoints.
int32 track_id
# Bounding box in pixels: [x_min, y_min, x_max, y_max]
float32[4] bbox
float32 bbox_confidence
# COCO 17 keypoints (see Keypoint.msg for the index map)
yolo11_pose_tracker/Keypoint[] keypoints
