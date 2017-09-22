SHOW_VIEWER = False
MODE = "check-traj"
ARMS_3D = False
USE_IMAGES = False
ROBOT_TYPES = (RobotType.PR2, False), (RobotType.BAXTER, False), (RobotType.PEGGY, False), (RobotType.THREE_LINK, False), (RobotType.FOUR_LINK, False), (RobotType.FIVE_LINK, False)
TASK_TYPES = [ColorReach(color, 6) for color in ("red", "green", "yellow", "black", "magenta", "cyan")]
VIDEO_PATH = None

BLOCK_LOCATIONS = [np.array([np.cos(theta), 0, np.sin(theta)]) * 0.75 for theta in np.linspace(0, 4, 6)]

NAME = "six_by_six_try_2"
