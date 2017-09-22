import os

SHOW_VIEWER = False
MODE = "check-all-traj"
ARMS_3D = False
USE_IMAGES = False
ROBOT_TYPES = (eval(os.environ['ROBOT_TYPE']), False),
VIDEO_PATH = "/home/kavi/Videos/pr2catch"

positions = [np.array([np.cos(theta), 10, np.sin(theta)]) * 0.20 for theta in np.linspace(0, 2 * np.pi, 7)[:-1]]
velocities = [[0, -4, 0] for x in positions]

TASK_TYPES = BlockCatch([pos + [0.7, 0, 0] for pos in positions], velocities),
