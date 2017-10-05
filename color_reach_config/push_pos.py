SHOW_VIEWER = False
MODE = os.environ['MODE']
ARMS_3D = False
USE_IMAGES = False
import os
inner_radius = float(os.environ['INNER_RADIUS'])
diff_radius = float(os.environ['DIFF_RADIUS'])
z_location = float(os.environ['Z_LOCATION'])
robot_type = eval(os.environ['ROBOT_TYPE'])
stem = os.environ['STEM'] if 'STEM' in os.environ else ""
ROBOT_TYPES = (robot_type, False),
TASK_TYPES = [BlockPush("red", np.linspace(-2, 2, 6), [-0.4, 0, 0.4], inner_radius, diff_radius, -z_location)]
VIDEO_PATH = None #"/home/kavi/Videos/pos%s" % inner_radius

NAME = "push_pos_a%s_%s_%s_%s_%s" % (stem, inner_radius, diff_radius, z_location, os.environ['ROBOT_TYPE'])

# VIDEO_PATH =  "/home/kavi/Videos/%s" % NAME
