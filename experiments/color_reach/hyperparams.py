from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd, init_from_file
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_imbalanced import model_fc_shared
from gps.algorithm.policy_opt.tf_model_example_multirobot import example_tf_network_multi, multitask_multirobot_conv_supervised
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3
USE_IMAGES = False
IS_TESTING = False
NEURAL_NET_ITERATIONS = 20000
ITERATIONS = 10
SAMPLES = 10

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

from gps.generalized_agents.reacher_by_color_and_type import RobotType, reacher_by_color_and_type

task_values, robot_values, arguments = [], [], []
for robot_n, robot_type in enumerate(RobotType):
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

agents = [reacher_by_color_and_type(i, len(arguments), color, robot_type, USE_IMAGES) for i, (color, robot_type) in enumerate(arguments)]

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

init_offset = np.array([0.8, 0.0, 0.5])
agent = [a['agent'] for a in agents]
for a in agent:
    a.update({'offsets': [np.asarray([-0.3, 0., -1.65]) + init_offset, np.asarray([0.4, 0., -1.3]) + init_offset,  np.asarray([0.45, 0., 0.45]) + init_offset, np.asarray([-0.4, 0.0, 0.7]) + init_offset, \
                np.asarray([-0.3, 0., -1.65]) + init_offset, np.asarray([0.4, 0., -1.3]) + init_offset,  np.asarray([0.45, 0., 0.45]) + init_offset, np.asarray([-0.4, 0.0, 0.7]) + init_offset


        ]})
algorithm = [a['algorithm'] for a in agents]

config = {
    'iterations': ITERATIONS,
    'is_testing' : IS_TESTING,
    'num_samples': SAMPLES * (1 - IS_TESTING),
    'verbose_trials': 0,
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
