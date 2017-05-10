from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example_multirobot import multitask_multirobot_conv_supervised

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

MODE = "training"
USE_IMAGES = False
LOAD_OLD_WEIGHTS = False
NEURAL_NET_ITERATIONS = 20000
ITERATIONS = 10
ARMS_3D = True
NAME = "PR2_vs_reachers"

if MODE == "testing":
    IS_TESTING = True
    SAMPLES = 10
    VERBOSE_TRIALS = False
elif MODE == "check-traj":
    IS_TESTING = False
    SAMPLES = 1
    VERBOSE_TRIALS = True
elif MODE == "training":
    IS_TESTING = False
    SAMPLES = 10
    VERBOSE_TRIALS = False
else:
    raise RuntimeError

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

from gps.generalized_agents.reacher_by_color_and_type import RobotType, reacher_by_color_and_type

BLOCK_LOCATIONS = [np.asarray(loc) / 2 for loc in ([-0.3, 0., -1.65], [0.4, 0., -1.3], [0.45, 0., 0.45], [-0.4, 0.0, 0.7])]
INIT_OFFSET = np.array([0.8, 0.0, 0.5]) / 2

task_values, robot_values, arguments = [], [], []
for robot_n, robot_type in enumerate((RobotType.FOUR_LINK, RobotType.PEGGY, RobotType.THREE_LINK_SHORT_JOINT)):
    for task_n, color in enumerate(("red", "green", "yellow", "black")):
        task_values.append(task_n)
        robot_values.append(robot_n)
        arguments.append((color, robot_type))

leave_one_out = 11
if IS_TESTING:
    task_values     = [task_values[leave_one_out]]
    robot_values    = [robot_values[leave_one_out]]
    arguments       = [arguments[leave_one_out]]
else:
    task_values     = task_values[:leave_one_out]+task_values[leave_one_out+1:]
    robot_values    = robot_values[:leave_one_out]+robot_values[leave_one_out+1:]
    arguments       = arguments[:leave_one_out]+arguments[leave_one_out+1:]

agents = [reacher_by_color_and_type(i, len(arguments), ARMS_3D, INIT_OFFSET, BLOCK_LOCATIONS, color, robot_type, USE_IMAGES) for i, (color, robot_type) in enumerate(arguments)]

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
        # 'restore_all_wts':'/home/abhigupta/gps/allweights_push_4link.npy'
    }
}


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [a['agent'] for a in agents]
for a in agent:
    a.update({'offsets': [x + INIT_OFFSET for x in BLOCK_LOCATIONS]})
algorithm = [a['algorithm'] for a in agents]

config = {
    'iterations': ITERATIONS,
    'is_testing' : IS_TESTING,
    'load_old_weights' : LOAD_OLD_WEIGHTS,
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
    #'val_agents': [1],
}

common['info'] = generate_experiment_info(config)
