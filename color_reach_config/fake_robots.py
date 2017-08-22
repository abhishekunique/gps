SHOW_VIEWER = True
MODE = "check-traj"
ARMS_3D = False
USE_IMAGES = False
ROBOT_TYPES = (RobotType.PEGGY, False), (RobotType.PR2, False)
TASK_TYPES = LegoReach("black"), LegoReach("yellow"), LegoReach("red")
VIDEO_PATH = None

# SAMPLES = 4

NEURAL_NET_ITERATIONS *= 0
NEURAL_NET_ITERATIONS += 2 * 10 ** 2
