SHOW_VIEWER = True
MODE = "check-model"
ARMS_3D = False
USE_IMAGES = True
ROBOT_TYPES = (RobotType.THREE_LINK, False), (RobotType.THREE_LINK_SHORT_JOINT, False)
TASK_TYPES = [ColorReach(color, 6) for color in ("red", "green", "yellow", "black", "magenta", "cyan")]
VIDEO_PATH = None

BLOCK_LOCATIONS = [np.array([np.cos(theta), 0, np.sin(theta)]) * 0.5 for theta in np.linspace(0, 4, 6)]
