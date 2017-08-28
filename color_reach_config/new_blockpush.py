SHOW_VIEWER = False
MODE = "training-trajectories"
ARMS_3D = False
USE_IMAGES = False
ROBOT_TYPES = (RobotType.THREE_LINK, False), (RobotType.PR2, False), (RobotType.BAXTER, False)
TASK_TYPES = [BlockPush]
VIDEO_PATH = None

BLOCK_LOCATIONS = [np.array([np.cos(theta), 0, np.sin(theta)]) * 0.5 for theta in np.linspace(0, 4, 6)]

BLOCKPUSH_BLOCK_LOCATIONS = [[theta, theta + d_theta] for theta in [-2.5, -1.5, 1.5, 2.5] for d_theta in [-0.2, 0, 0.2]]
BLOCKPUSH_BLOCK_LOCATIONS = [[np.array([np.cos(th), 0, np.sin(th)]) * r for th, r in zip(thetas, (0.60, 0.80))] for thetas in BLOCKPUSH_BLOCK_LOCATIONS]
