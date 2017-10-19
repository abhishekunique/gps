SHOW_VIEWER = True
MODE = os.environ['MODE']
ARMS_3D = False
USE_IMAGES = False

ROBOT_TYPES = [(robot_type, False) for robot_type in [RobotType.PR2, RobotType.PEGGY, RobotType.THREE_DF_BLOCK, RobotType.FOUR_SEVEN]]
TASK_TYPES = [BlockPush(color, np.linspace(-2, 2, 6), [-0.4, 0, 0.4], 0.75, 0.3, -0.2) for color in "red", "yellow", "green"]
VIDEO_PATH = None #"/home/kavi/Videos/pos%s" % inner_radius
stem = os.environ['STEM'] if 'STEM' in os.environ else ""

#####
INNER_RADIUS = 0.9
DIFF_RADIUS = 0.3
INITIAL_ANGLES = [-2.2, -1.8, -1.2, 1.2, 1.8, 2.2]
DIFF_ANGLES = np.linspace(-np.pi/4, np.pi/4, 3)

TASK_TYPES += [BlockVelocityPush(color, INITIAL_ANGLES, DIFF_ANGLES, INNER_RADIUS, DIFF_RADIUS) for color in "red", "yellow", "green"]
INNER_RADIUS = 0.75
ANGLES = [-2.2, -1.8, -1.2, 1.2, 1.8, 2.2]

TASK_TYPES += [ColorReachRYG(color, ANGLES, INNER_RADIUS) for color in "red", "yellow", "green"]


ROBOT_INDEX = eval(os.environ['ROBOT_INDEX'])
TASK_INDEX = eval(os.environ['TASK_INDEX'])

ROBOT_TYPES = [ROBOT_TYPES[ROBOT_INDEX]]
TASK_TYPES = [TASK_TYPES[TASK_INDEX]]

NAME = "nonmodular_%s_%s" % (ROBOT_INDEX, TASK_INDEX)

NEURAL_NET_ITERATIONS *= 0
NEURAL_NET_ITERATIONS += 100 * 1000

DONE_AFTER_SUCCESSES = eval(os.environ['EVALUATE'])

assert type(DONE_AFTER_SUCCESSES) == bool

POLICY_TRIALS = 30 * DONE_AFTER_SUCCESSES

ITERATIONS = 1

TASKOUT_SIZE = 80
