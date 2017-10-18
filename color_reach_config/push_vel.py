SHOW_VIEWER = False
MODE = os.environ['MODE']
ARMS_3D = True
USE_IMAGES = False

INNER_RADIUS = 0.9
DIFF_RADIUS = 0.3
INITIAL_ANGLES = [-2.2, -1.8, -1.2, 1.2, 1.8, 2.2]
DIFF_ANGLES = np.linspace(-np.pi/4, np.pi/4, 3)

stem = os.environ['STEM'] if 'STEM' in os.environ else ""
VIDEO_PATH = None # "/home/kavi/Videos/pr2vel"

ROBOT_TYPES = [(robot_type, False) for robot_type in [eval(os.environ['ROBOT_TYPE'])]]
TASK_TYPES = [BlockVelocityPush(color, INITIAL_ANGLES, DIFF_ANGLES, INNER_RADIUS, DIFF_RADIUS) for color in "red", "yellow", "green"]

str_seed = ""

if 'RANDOM_SEED' in os.environ:
    RANDOM_SEED = int(os.environ['RANDOM_SEED'])
    str_seed = "seed=" + os.environ['RANDOM_SEED']


NAME = "push_vel_small%s%s%s" % (stem, ROBOT_TYPES, str_seed)

print "*" * 20
print NAME
print "*" * 20

if MODE == "check-all-traj":
    VIDEO_PATH = "/home/kavi/Videos/1015-vel-%s" % os.environ['ROBOT_TYPE']
