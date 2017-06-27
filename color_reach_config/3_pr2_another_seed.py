from os import environ
SHOW_VIEWER = False
MODE = "training"
USE_IMAGES = False
ARMS_3D = True
ROBOT_TYPES = RobotType.THREE_LINK, RobotType.PR2, RobotType.FOUR_LINK
TASK_TYPES = REACHERS
NAME = "3_pr2_another_seed"
VIDEO_PATH = None
RANDOM_SEED = 0x123ABC # not sure if this is what was used at the time
# LEAVE_ONE_OUT = environ["LEAVE_ONE_OUT"]
