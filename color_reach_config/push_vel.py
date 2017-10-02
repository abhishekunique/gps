SHOW_VIEWER = False
MODE = os.environ['MODE']
ARMS_3D = True
USE_IMAGES = False
import os
inner_radius = float(os.environ['INNER_RADIUS'])
diff_radius = float(os.environ['DIFF_RADIUS'])
robot_type = eval(os.environ['ROBOT_TYPE'])
stem = os.environ['STEM'] if 'STEM' in os.environ else ""
ROBOT_TYPES = (robot_type, False),
VIDEO_PATH = None # "/home/kavi/Videos/pr2vel"

TASK_TYPES = BlockVelocityPush([-2.2, -1.8, -1.2, 1.2, 1.8, 2.2], np.linspace(-np.pi/4, np.pi/4, 3), inner_radius, diff_radius, "red"),

NAME = "push_vel_small%s_%s_%s_%s" % (stem, inner_radius, diff_radius, os.environ['ROBOT_TYPE'])

if MODE == "check-all-traj":
    VIDEO_PATH = "/home/kavi/Videos/0930-%s" % NAME
