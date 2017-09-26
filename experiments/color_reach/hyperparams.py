from __future__ import division

from datetime import datetime
import os.path
from sys import argv
import numpy as np
from itertools import product

from gps import __file__ as gps_filepath
gps_filepath = os.path.abspath(gps_filepath)
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example_multirobot import multitask_multirobot_conv_supervised

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

from gps.generalized_agents.reacher_by_color_and_type import *

REACHERS = map(ColorReach, COLOR_ORDER)
BLOCKPUSH_FIRST = [BlockPush] + REACHERS[:3]
BLOCKPUSH_SECOND = ColorReach("red"), BlockPush, ColorReach("green"), ColorReach("yellow")

ARMS = (RobotType.THREE_LINK, False), (RobotType.FOUR_LINK, False), (RobotType.FIVE_LINK, False)
PEGGY_VS_ARMS = (RobotType.PEGGY, False), (RobotType.THREE_LINK, False), (RobotType.FOUR_LINK, False)
PR2_VS_ARMS = (RobotType.PR2, False), (RobotType.THREE_LINK, False), (RobotType.FOUR_LINK, False)
BAXTER_VS_ARMS = (RobotType.BAXTER, False), (RobotType.THREE_LINK, False), (RobotType.FOUR_LINK, False)

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

PASS_ENVIRONMENT_EFFECTORS_TO_ROBOT = False # False for mixing blockpush and color reachers. True to load old models
LEGACY_BLOCK_POSITIONS = False
LOAD_OLD_WEIGHTS = True
NEURAL_NET_ITERATIONS = 20000
ITERATIONS = 100

LEAVE_ONE_OUT = 0
N_BLOCK_CONDITIONS = 6
NAME = None
COLOR_BLOCKS_3D = None
RANDOM_SEED = 0x123ABC
SAMPLES = None
BLOCK_LOCATIONS = BLOCKPUSH_BLOCK_LOCATIONS = None
TORQUE_COSTS = True

CONFIG_FILE = argv[argv.index("--config") + 1]
execfile(CONFIG_FILE)

if NAME is None:
    NAME = os.path.basename(CONFIG_FILE)[:-3]
    print "Name", NAME

SHOW_VIEWER, MODE, USE_IMAGES, ARMS_3D, ROBOT_TYPES, TASK_TYPES, VIDEO_PATH # ensure that all these names exist

if MODE == "testing" or MODE == "taskout-print":
    SHOW_VIEWER = True
    IS_TESTING = True
    if SAMPLES is None:
        SAMPLES = 20
    VERBOSE_TRIALS = False
    VIEW_TRAJECTORIES = True
elif MODE == "check-traj" or MODE == "check-all-traj":
    IS_TESTING = False
    if SAMPLES is None:
        SAMPLES = 1
    VERBOSE_TRIALS = True
    VIEW_TRAJECTORIES = False
    SHOW_VIEWER = True
elif MODE == "training" or MODE == "training-trajectories":
    IS_TESTING = False
    if SAMPLES is None:
        SAMPLES = 20
    VERBOSE_TRIALS = SHOW_VIEWER
    VIEW_TRAJECTORIES = False
elif MODE == "check-model":
    LEAVE_ONE_OUT = 1
    SHOW_VIEWER = True
    IS_TESTING = False
    if SAMPLES is None:
        SAMPLES = 1
    VERBOSE_TRIALS = SHOW_VIEWER
    VIEW_TRAJECTORIES = False
elif MODE == "view-traj":
    IS_TESTING = False
    if SAMPLES is None:
        SAMPLES = 20
    VERBOSE_TRIALS = False
    VIEW_TRAJECTORIES = True
else:
    raise RuntimeError

taskout_print = MODE == "taskout-print"

if COLOR_BLOCKS_3D is None:
    COLOR_BLOCKS_3D = ARMS_3D

if BLOCK_LOCATIONS is None:
    BLOCK_LOCATIONS = [np.asarray(loc) / 2 + (1 - LEGACY_BLOCK_POSITIONS) * np.array([0.1, 0, 0.2]) for loc in ([-0.3, 0., -1.65], [0.4, 0., -1.3], [0.45, 0., 0.45], [-0.4, 0.0, 0.7])]
first_row_y, second_row_y = -1, -0.5
first_row_dx, second_row_dx = 0.3, 0.5
LEGO_BLOCK_LOCATIONS = [np.asarray(loc) for loc in ([-second_row_dx, 0., second_row_y], [0, 0., second_row_y], [second_row_dx, 0., second_row_y], [-first_row_dx, 0, first_row_y], [first_row_dx, 0, first_row_y])]
BLOCK_VERTICAL_LOCATIONS = [-0.5, 0, 0.5] if COLOR_BLOCKS_3D else [0]

if BLOCKPUSH_BLOCK_LOCATIONS is None:
    BLOCKPUSH_BLOCK_LOCATIONS = [[np.array([x, 0, z]) / 2 for x, z in both] for both in
        [[[np.cos(theta), np.sin(theta)], [np.cos(theta) + np.cos(theta + off), np.sin(theta) + np.sin(theta + off)]]
            for theta in [np.pi / N_BLOCK_CONDITIONS * i - np.pi/2 for i in range(N_BLOCK_CONDITIONS)]
            for off in (-np.pi/2, np.pi/2)
    ]]

task_values, robot_values, arguments = [], [], []
for robot_n, robot_type in enumerate(ROBOT_TYPES):
    for task_n, task_type in enumerate(TASK_TYPES):
        task_values.append(task_n)
        robot_values.append(robot_n)
        arguments.append((task_type, robot_type))

leave_one_out = LEAVE_ONE_OUT
if MODE == "training-trajectories" or MODE == "check-all-traj":
    task_values, robot_values, arguments = zip(*((task, robot, arg) for task, robot, arg in zip(task_values, robot_values, arguments) if not arg[1][1]))
elif IS_TESTING:
    task_values     = [task_values[leave_one_out]]
    robot_values    = [robot_values[leave_one_out]]
    arguments       = [arguments[leave_one_out]]
else:
    task_values     = task_values[:leave_one_out]+task_values[leave_one_out+1:]
    robot_values    = robot_values[:leave_one_out]+robot_values[leave_one_out+1:]
    arguments       = arguments[:leave_one_out]+arguments[leave_one_out+1:]

agents = [reacher_by_color_and_type(i,
                                    len(arguments),
                                    ARMS_3D,
                                    BLOCK_LOCATIONS,
                                    BLOCK_VERTICAL_LOCATIONS,
                                    LEGO_BLOCK_LOCATIONS,
                                    BLOCKPUSH_BLOCK_LOCATIONS,
                                    robot_type,
                                    USE_IMAGES,
                                    task_type,
                                    torque_costs=TORQUE_COSTS,
                                    pass_environment_effectors_to_robot=PASS_ENVIRONMENT_EFFECTORS_TO_ROBOT,
                                    number_samples=SAMPLES,
                                    IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT=IMAGE_HEIGHT, IMAGE_CHANNELS=IMAGE_CHANNELS)
            for i, (task_type, robot_type) in enumerate(arguments)]

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/color_reach/'
INIT_POLICY_DIR = '/home/abhigupta/gps/'
common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 8,
    'train_conditions': [0,1,2,3],
    'test_conditions': [4,5,6,7],
    'num_robots':len(agents),
    'policy_opt': {
        'type': PolicyOptTf,
        'network_model': lambda *args, **kwargs: multitask_multirobot_conv_supervised(*args, use_image=USE_IMAGES, is_testing=IS_TESTING, **kwargs),
        'network_params': {
            'task_list': task_values,
            'robot_list': robot_values,
            'agent_params':[a['network_params'] for a in agents],
        },
        #'val_agents': [1],
        'iterations': NEURAL_NET_ITERATIONS,
        'fc_only_iterations': 5000,
        'checkpoint_prefix': EXP_DIR + 'data_files/policy',
        'print_task_out' : taskout_print,
        # 'restore_all_wts':'/home/abhigupta/gps/allweights_push_4link.npy'
    }
}


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [a['agent'] for a in agents]
for a in agent:
    a.update({
        'show_viewer' : SHOW_VIEWER
    })
for a in agents:
    a['agent']['write_video'] = VIDEO_PATH
algorithm = [a['algorithm'] for a in agents]

config = {
    'iterations': ITERATIONS,
    'is_testing' : IS_TESTING,
    'load_old_weights' : LOAD_OLD_WEIGHTS,
    'view_trajectories' : VIEW_TRAJECTORIES,
    'nn_dump_path' : "dump/nn_weights_%s" % NAME,
    'traj_distr_dump' : "dump/traj_distr_%s.pkl" % NAME,
    'num_samples': SAMPLES,
    'verbose_trials': SAMPLES * VERBOSE_TRIALS,
    'verbose_policy_trials': int(IS_TESTING),
    'save_wts': True,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'inner_iterations': 4,
    'robot_iters': [range(25), range(0,25,2)],
    'to_log': [END_EFFECTOR_POINTS, JOINT_ANGLES, ACTION],
    'random_seed' : RANDOM_SEED
    #'val_agents': [1],
}

common['info'] = generate_experiment_info(config)
