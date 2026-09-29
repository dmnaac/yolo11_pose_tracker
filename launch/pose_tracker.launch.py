import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Bundled assets installed from the package's model/ folder:
    #   model/yolo11n-pose.pt       - YOLO11 pose weights
    #   model/botsort_reid.yaml     - BoT-SORT tracker with ReID (default)
    #   model/tracktrack_reid.yaml  - TrackTrack tracker with ReID (alternative)
    model_dir = os.path.join(
        get_package_share_directory('yolo11_pose_tracker'), 'model')

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/camera/color/image_raw',
                              description='Input camera image topic'),
        DeclareLaunchArgument('model_path',
                              default_value=os.path.join(model_dir, 'yolo11n-pose.pt'),
                              description='Path to YOLO11 pose .pt model'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description="Inference device: 'cpu' or 'cuda:0' etc."),
        DeclareLaunchArgument('conf_threshold', default_value='0.25',
                              description='Detection confidence threshold'),
        DeclareLaunchArgument('tracker',
                              default_value=os.path.join(model_dir, 'tracktrack_reid.yaml'),
                              description='Ultralytics tracker config (bundled: '
                                          'botsort_reid.yaml / tracktrack_reid.yaml, '
                                          'or any custom yaml path)'),
        DeclareLaunchArgument('publish_annotated', default_value='true',
                              description='Publish annotated debug image'),
        Node(
            package='yolo11_pose_tracker',
            executable='pose_tracker_node.py',
            name='pose_tracker_node',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'conf_threshold': ParameterValue(
                    LaunchConfiguration('conf_threshold'), value_type=float),
                'tracker': LaunchConfiguration('tracker'),
                'publish_annotated': ParameterValue(
                    LaunchConfiguration('publish_annotated'), value_type=bool),
            }],
        ),
    ])
