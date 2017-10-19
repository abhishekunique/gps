CUDA_VISIBLE_DEVICES=0 ROBOT_TYPE=RobotType.PR2 GRID=pr2_reach MODE=training REG='dropout' python -u python/gps/gps_main.py tasks_ros --config color_reach_config/ros_push.py
