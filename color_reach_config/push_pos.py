SHOW_VIEWER = False
MODE = "training-trajectories"#os.environ['MODE']
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

NAME = "push_pos%s" % stem

# VIDEO_PATH =  "/home/kavi/Videos/%s" % NAME
