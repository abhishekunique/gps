SHOW_VIEWER = False
MODE = os.environ['MODE']
ARMS_3D = False
USE_IMAGES = False
import os
inner_radius = float(os.environ['INNER_RADIUS'])
diff_radius = float(os.environ['DIFF_RADIUS'])
z_location = float(os.environ['Z_LOCATION'])
robot_type = eval(os.environ['ROBOT_TYPE'])
ROBOT_TYPES = (robot_type, False),
TASK_TYPES = [BlockPush]
VIDEO_PATH = None #"/home/kavi/Videos/pos%s" % inner_radius

BLOCKPUSH_ANGLES = [[theta, theta + d_theta] for theta in np.linspace(-2, 2, 6) for d_theta in [-0.4, 0, 0.4]]

def to_cartesian(r, theta, z=0):
    return np.array([np.cos(theta), z, np.sin(theta)]) * r

BLOCK_START = [to_cartesian(inner_radius, th, z=-z_location) for th, _ in BLOCKPUSH_ANGLES]
VELOCITIES  = [to_cartesian(diff_radius, th) for _, th in BLOCKPUSH_ANGLES]
BLOCKPUSH_BLOCK_LOCATIONS = [[x, x + v] for x, v in zip(BLOCK_START, VELOCITIES)]



NAME = "push_pos_%s_%s_%s_%s" % (inner_radius, diff_radius, z_location, os.environ['ROBOT_TYPE'])

# VIDEO_PATH =  "/home/kavi/Videos/%s" % NAME
