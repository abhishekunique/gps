SHOW_VIEWER = False
MODE = os.environ['MODE']
ARMS_3D = False
USE_IMAGES = False
import os
inner_radius = 0.75
diff_radius = 0.3
z_location = -0.2
stem = os.environ['STEM'] if 'STEM' in os.environ else ""

ROBOT_TYPES = [(robot_type, False) for robot_type in [RobotType.PR2, RobotType.PEGGY, RobotType.THREE_DF_BLOCK, RobotType.FOUR_SEVEN]]
TASK_TYPES = [BlockPush(color, np.linspace(-2, 2, 6), [-0.4, 0, 0.4], inner_radius, diff_radius, z_location) for color in "red", "yellow", "green"]
VIDEO_PATH = None #"/home/kavi/Videos/pos%s" % inner_radius

#####
INNER_RADIUS = 0.9
DIFF_RADIUS = 0.3
INITIAL_ANGLES = [-2.2, -1.8, -1.2, 1.2, 1.8, 2.2]
DIFF_ANGLES = np.linspace(-np.pi/4, np.pi/4, 3)

stem = os.environ['STEM'] if 'STEM' in os.environ else ""
VIDEO_PATH = None # "/home/kavi/Videos/pr2vel"

#ROBOT_TYPES += [(robot_type, False) for robot_type in [RobotType.PR2, RobotType.PEGGY, RobotType.THREE_DF_BLOCK, RobotType.FOUR_SEVEN]]
TASK_TYPES += [BlockVelocityPush(color, INITIAL_ANGLES, DIFF_ANGLES, INNER_RADIUS, DIFF_RADIUS) for color in "red", "yellow", "green"]

INNER_RADIUS = 0.75
ANGLES = [-2.2, -1.8, -1.2, 1.2, 1.8, 2.2]
TASK_TYPES += [ColorReachRYG(color, ANGLES, INNER_RADIUS) for color in "red", "yellow", "green"]
VIDEO_PATH = None #"/home/kavi/Videos/pos%s" % inner_radius

NAME = "all_envs%s" % stem
