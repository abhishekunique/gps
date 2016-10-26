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
from gps.algorithm.cost.cost_fk_dev import CostFKDev
from gps.algorithm.cost.cost_dev_rs_strike import CostDevRs
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_imbalanced import model_fc_shared
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, RAMP_MIDDLE_DRAWER
from gps.utility.data_logger import DataLogger

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = [{
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 3,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}]

PR2_GAINS = [np.array([1.0, 1.0, 1.0]), np.array([ 1.0, 1.0, 1.0, 1.0])]

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/fruit_obstacle/'

OBS_INCLUDE =  [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES]

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 2,
    'train_conditions': [0],
    'test_conditions': [1],
    'num_robots':1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [{
    'type': AgentMuJoCo,
    'filename': './mjc_models/3link_obstacle_fruit.xml',
    'x0': np.zeros((6,)),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': [
                        [np.array([0.5, 0.0, 0.8]),np.array([0.8, 0.0, 0.5])],
                        [np.array([0.3, 0.0, 1.0]),np.array([0.6, 0.0, 0.6])],
                        # [np.array([-0.9, 0.0, 0.7]),np.array([0.5, 0.0, 1.3])],
                        # [np.array([-0.7, 0.0, -0.6]),np.array([0.8, 0.0, -1.2])],

                        # [np.array([-0.3, 0.0, 0.6]),np.array([0.5, 0.0, 0.9])],
                        # [np.array([-0.4, 0.0, -0.5]),np.array([0.5, 0.0, -0.75])],
                        # [np.array([-0.3, 0.0, 0.6]),np.array([0.6, 0.0, 0.85])],
                        # [np.array([-0.4, 0.0, -0.6]),np.array([0.45, 0.0, -0.95])],
                        ],
    'pos_body_idx': np.array([6,7]),
    'conditions': 2,
    'train_conditions': [0],
    'test_conditions': [1],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 100,
    'sensor_dims': SENSOR_DIMS[0],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),

}
]

algorithm = [{
    'type': AlgorithmTrajOpt,
    'conditions': agent[0]['conditions'],
    'train_conditions': agent[0]['train_conditions'],
    'test_conditions': agent[0]['test_conditions'],
    'num_robots': common['num_robots'],
    'iterations': 25,
}]


# DIFFERENT!!!!
algorithm[0]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 50.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[0][ACTION],
    'dt': agent[0]['dt'],
    'T': agent[0]['T'],
}


fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0.05, 0.05, 0.05]) + agent[0]['pos_body_offset'][i][0], np.array([0.0, 0.0, 0.0])]),
    'wp': np.array([ 1, 1, 1,0,0,0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'ramp_option': RAMP_QUADRATIC
} for i in agent[0]['train_conditions']]

# fk_waypt = [{
#     'type': CostFK,
#     'target_end_effector': np.concatenate((agent[0]['pos_body_offset'][i][1]- np.array([0.15, 0., 0.15]), np.array([0.0, 0.0, 0.0]))),
#     'wp': np.array([ 1, 1, 1,0,0,0]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
#     'ramp_option': RAMP_MIDDLE_DRAWER
# } for i in agent[0]['train_conditions']]

algorithm[0]['cost'] = [{
    'type': CostSum,
    'costs': [fk_cost_1[i]],
    'weights': [1.0],
} for i in agent[0]['train_conditions']]



algorithm[0]['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}



algorithm[0]['traj_opt'] = {
    'type': TrajOptLQRPython,
    'robot_number':0
}


algorithm[0]['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
    'robot_number':0
}


config = {
    'iterations': 25,
    'num_samples': 7,
    'verbose_trials': 7,
    'save_wts': True,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'inner_iterations': 4,
    'to_log': [],
    'robot_iters': [range(25), range(0,25,2)],
}

common['info'] = generate_experiment_info(config)
