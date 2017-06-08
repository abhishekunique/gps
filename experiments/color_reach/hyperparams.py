from __future__ import division

from datetime import datetime
import os.path
import numpy as np
from itertools import product

from gps import __file__ as gps_filepath
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example_multirobot import multitask_multirobot_conv_supervised

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

from gps.generalized_agents.reacher_by_color_and_type import RobotType, reacher_by_color_and_type, BlockPush, ColorReach, COLOR_ORDER

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

PASS_ENVIRONMENT_EFFECTORS_TO_ROBOT = False # False for mixing blockpush and color reachers. True to load old models
SHOW_VIEWER = False
MODE = "training"
USE_IMAGES = False
LOAD_OLD_WEIGHTS = True
NEURAL_NET_ITERATIONS = 20000
ITERATIONS = 100
ARMS_3D = True
ROBOT_TYPES = RobotType.BAXTER, RobotType.THREE_LINK, RobotType.FOUR_LINK
TASK_TYPES = map(ColorReach, COLOR_ORDER)
NAME = "baxter_demonstration"

if MODE == "testing":
    IS_TESTING = True
    SAMPLES = 10
    VERBOSE_TRIALS = False
    VIEW_TRAJECTORIES = True
elif MODE == "check-traj":
    IS_TESTING = False
    SAMPLES = 1
    VERBOSE_TRIALS = True
    VIEW_TRAJECTORIES = False
elif MODE == "training":
    IS_TESTING = False
    SAMPLES = 5
    VERBOSE_TRIALS = False
    VIEW_TRAJECTORIES = False
elif MODE == "view-traj":
    IS_TESTING = False
    SAMPLES = 10
    VERBOSE_TRIALS = False
    VIEW_TRAJECTORIES = True
else:
    raise RuntimeError

BLOCK_LOCATIONS = [np.asarray(loc) / 2 for loc in ([-0.3, 0., -1.65], [0.4, 0., -1.3], [0.45, 0., 0.45], [-0.4, 0.0, 0.7])]
BLOCK_VERTICAL_LOCATIONS = [-0.5, 0, 0.5]
INIT_OFFSET = np.array([0.8, 0.0, 0.5]) / 2

task_values, robot_values, arguments = [], [], []
for robot_n, robot_type in enumerate(ROBOT_TYPES):
    for task_n, task_type in enumerate(TASK_TYPES):
        task_values.append(task_n)
        robot_values.append(robot_n)
        arguments.append((task_type, robot_type))

leave_one_out = 0
if IS_TESTING:
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
                                    INIT_OFFSET,
                                    BLOCK_LOCATIONS,
                                    BLOCK_VERTICAL_LOCATIONS,
                                    robot_type,
                                    USE_IMAGES,
                                    task_type,
                                    pass_environment_effectors_to_robot=PASS_ENVIRONMENT_EFFECTORS_TO_ROBOT)
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
        'print_task_out' : False,
        # 'restore_all_wts':'/home/abhigupta/gps/allweights_push_4link.npy'
    }
}


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [a['agent'] for a in agents]
for a in agent:
    a.update({
        'offsets': [x + INIT_OFFSET for x in BLOCK_LOCATIONS],
        'vertical_offsets' : BLOCK_VERTICAL_LOCATIONS,
        'show_viewer' : SHOW_VIEWER
    })
algorithm = [a['algorithm'] for a in agents]

config = {
    'iterations': ITERATIONS,
    'is_testing' : IS_TESTING,
    'load_old_weights' : LOAD_OLD_WEIGHTS,
    'view_trajectories' : VIEW_TRAJECTORIES,
    'nn_dump_path' : "nn_weights_%s" % NAME,
    'traj_distr_dump' : "traj_distr_%s.pkl" % NAME,
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
    'random_seed' : 0xB0BACAFE
    #'val_agents': [1],
}

common['info'] = generate_experiment_info(config)
