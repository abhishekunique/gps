SHOW_VIEWER = False
MODE = "training-trajectories"
ARMS_3D = False
USE_IMAGES = False
import os
inner_radius = float(os.environ['INNER_RADIUS'])
outer_radius = inner_radius + float(os.environ['DIFF_RADIUS'])
robot_type = eval(os.environ['ROBOT_TYPE'])
ROBOT_TYPES = (robot_type, False),
TASK_TYPES = [BlockPush]
VIDEO_PATH = None

BLOCKPUSH_BLOCK_LOCATIONS = [[theta, theta + d_theta] for theta in np.linspace(-3, 4, 8) for d_theta in [-0.2, 0, 0.2]]
BLOCKPUSH_BLOCK_LOCATIONS = [[np.array([np.cos(th), 0, np.sin(th)]) * r for th, r in zip(thetas, (inner_radius, outer_radius))] for thetas in BLOCKPUSH_BLOCK_LOCATIONS]

NAME = "blockpush_pos_tool_round_%s_%s_%s" % (inner_radius, outer_radius, os.environ['ROBOT_TYPE'])
