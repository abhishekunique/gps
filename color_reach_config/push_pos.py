SHOW_VIEWER = False
MODE = "training-trajectories"#os.environ['MODE']
ARMS_3D = False
USE_IMAGES = False
import os
inner_radius = 0.75
diff_radius = 0.3
z_location = -0.2
stem = os.environ['STEM'] if 'STEM' in os.environ else ""

ROBOT_TYPES = [(robot_type, False) for robot_type in [eval(os.environ['ROBOT_TYPE'])]]
TASK_TYPES = [BlockPush(color, np.linspace(-2, 2, 6), [-0.4, 0, 0.4], inner_radius, diff_radius, z_location) for color in "red", "yellow", "green"]
VIDEO_PATH = None #"/home/kavi/Videos/pos%s" % inner_radius

str_seed = ""

if 'RANDOM_SEED' in os.environ:
    RANDOM_SEED = int(os.environ['RANDOM_SEED'])
    str_seed = "seed=" + os.environ['RANDOM_SEED']


NAME = "push_pos%s%s%s" % (stem, ROBOT_TYPES, str_seed)

if 'NAME' in os.environ:
    NAME = os.environ['NAME']
