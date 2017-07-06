SHOW_VIEWER = False
MODE = "training"
USE_IMAGES = False
ARMS_3D = False
COLOR_BLOCKS_3D = False
ROBOT_TYPES = RobotType.THREE_LINK, RobotType.THREE_LINK_SHORT_JOINT, RobotType.FOUR_LINK
TASK_TYPES = CleaningSingleObject(smoothing=True), ColorReach("red"), ColorReach("green"), ColorReach("yellow")
VIDEO_PATH = None

NAME = "cleaning_single_object_fixed"
