from os import environ
SHOW_VIEWER = False
MODE = "training"
USE_IMAGES = False
ARMS_3D = True
ROBOT_TYPES = RobotType.THREE_LINK, RobotType.PR2, RobotType.FOUR_LINK
TASK_TYPES = REACHERS
NAME = "3_pr2"
VIDEO_PATH = None
# LEAVE_ONE_OUT = int(environ["LEAVE_ONE_OUT"])
RANDOM_SEED = 0xB0BACAFE # not sure if this is what was used at the time
