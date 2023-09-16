from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()
    ld.add_action(
        Node(
            package="gelsight_ros2",
            executable="gelsight_showdepth",
            output="screen",
            name="gsmini_left",
            parameters=[{"device_name": "GelSight Mini R0B 28GH-EJZ8"}],
            remappings=[('/gsmini/depth', '/gsmini_left/depth'),
                        ('/gsmini/rgb', '/gsmini_left/rgb')],
        )
    )
    ld.add_action(
        Node(
            package="gelsight_ros2",
            executable="gelsight_showdepth",
            output="screen",
            name="gsmini_right",
            parameters=[{"device_name":"GelSight Mini R0B 2871-46HU"}],
            remappings=[('/gsmini/depth', '/gsmini_right/depth'),
                        ('/gsmini/rgb', '/gsmini_right/rgb')],
        )
    )
    ld.add_action(
        Node(
            package="bye140_ros2",
            executable="start_node",
            output="screen",
            name="bye140",
            parameters=[{'status_frequency': 5.0}],
        )
    )
    ld.add_action(
        Node(
            package="tactile_adaptive_grasp",
            executable="adaptive_grasp.py",
            output="screen",
            name="grasp_control"
        )
    )
    return ld
