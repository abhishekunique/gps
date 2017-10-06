SHOW_VIEWER = False
MODE = os.environ['MODE']
ARMS_3D = False
USE_IMAGES = False
INNER_RADIUS = 0.75
ANGLES = [-2.2, -1.8, -1.2, 1.2, 1.8, 2.2]
stem = os.environ['STEM'] if 'STEM' in os.environ else ""
ROBOT_TYPES = [(robot_type, False) for robot_type in [RobotType.PR2, RobotType.PEGGY, RobotType.THREE_DF_BLOCK, RobotType.FOUR_SEVEN]]
TASK_TYPES = [ColorReachRYG(color, ANGLES, INNER_RADIUS) for color in "red", "yellow", "green"]
VIDEO_PATH = None #"/home/kavi/Videos/pos%s" % inner_radius

NAME = "color_reach_ryg%s" % stem

if MODE == "check-all-traj":
    VIDEO_PATH =  "/home/kavi/Videos/%s" % NAME
