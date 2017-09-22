SHOW_VIEWER = False
MODE = "training-trajectories"
ARMS_3D = True
USE_IMAGES = False
import os
inner_radius = float(os.environ['INNER_RADIUS'])
diff_radius = float(os.environ['DIFF_RADIUS'])
robot_type = eval(os.environ['ROBOT_TYPE'])
ROBOT_TYPES = (robot_type, False),
VIDEO_PATH = None # "/home/kavi/Videos/pr2vel"

SAMPLES = 4

BLOCKPUSH_ANGLES = [[theta, theta + d_theta] for theta in [-2.2, -1.8, -1.2, 1.2, 1.8, 2.2] for d_theta in np.linspace(-np.pi/4, np.pi/4, 3)]

def to_cartesian(r, theta):
    return np.array([np.cos(theta), 0, np.sin(theta)]) * r

BLOCK_START = [to_cartesian(inner_radius, th) for th, _ in BLOCKPUSH_ANGLES]
VELOCITIES  = [to_cartesian(diff_radius * 5, th) for _, th in BLOCKPUSH_ANGLES]
BLOCKPUSH_BLOCK_LOCATIONS = [[x, x + v] for x, v in zip(BLOCK_START, VELOCITIES)]


TASK_TYPES = BlockVelocityPush([v * 4 for v in VELOCITIES]),

NAME = "push_vel_%s_%s_%s" % (inner_radius, diff_radius, os.environ['ROBOT_TYPE'])
