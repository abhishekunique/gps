SHOW_VIEWER = True
MODE = "training"
ARMS_3D = False
USE_IMAGES = True
ROBOT_TYPES = RobotType.BAXTER, RobotType.PEGGY, RobotType.PR2
TASK_TYPES = REACHERS
VIDEO_PATH = None

NEURAL_NET_ITERATIONS *= 0
NEURAL_NET_ITERATIONS += 2 * 10 ** 5
