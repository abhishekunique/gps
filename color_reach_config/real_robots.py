SHOW_VIEWER = True
MODE = "training"
ARMS_3D = False
USE_IMAGES = True
ROBOT_TYPES = (RobotType.PEGGY, False), (RobotType.PR2, True)
ROBOT_TYPES = reversed(ROBOT_TYPES)
TASK_TYPES = LegoReach("red"), LegoReach("yellow")
VIDEO_PATH = None

SAMPLES = 2

NEURAL_NET_ITERATIONS *= 0
NEURAL_NET_ITERATIONS += 2 * 10 ** 5
