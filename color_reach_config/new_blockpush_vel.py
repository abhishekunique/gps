SHOW_VIEWER = False
MODE = "check-model"
ARMS_3D = False
USE_IMAGES = False
ROBOT_TYPES = (RobotType.THREE_LINK, False),
VIDEO_PATH = None

BLOCK_LOCATIONS = [np.array([np.cos(theta), 0, np.sin(theta)]) * 0.5 for theta in np.linspace(0, 4, 6)]

BLOCKPUSH_BLOCK_LOCATIONS = [[theta, theta + d_theta] for theta in np.linspace(-2, 2, 4) for d_theta in [-0.2, 0, 0.2]]
BLOCKPUSH_BLOCK_LOCATIONS = [[np.array([np.cos(th), 0, np.sin(th)]) * r for th, r in zip(thetas, (0.45, 0.70))] for thetas in BLOCKPUSH_BLOCK_LOCATIONS]

TASK_TYPES = BlockVelocityPush([target - start for start, target in BLOCKPUSH_BLOCK_LOCATIONS]),
