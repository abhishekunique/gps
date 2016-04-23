""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network



IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 3,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}

PR2_GAINS = np.array([1.0, 1.0, 1.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_3link_reach/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 3,   
    'train_conditions': [0,1],
    'test_conditions':[2]
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/arm_3link_reach.xml',
    'x0': np.zeros(6),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': [np.array([0, 0.0, 0]), np.array([0, 0., 0.4]), np.array([0, 0., 0.2])],
    'pos_body_idx': np.array([6]),
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],
    'meta_include': [RGB_IMAGE_SIZE],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'iterations': 25,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1
}

# algorithm = {
#     'type': AlgorithmTrajOpt,
#     'conditions': common['conditions'],
#     'iterations': 25,
# }

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost_1 = [{
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
},
{
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
}]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.array([0.8, 0.0, 0.5]),
    'wp': np.array([1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
},
{
    'type': CostFK,
    'target_end_effector': np.array([0.8, 0.0, 0.9]),
    'wp': np.array([1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}]



algorithm['cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [1.0, 1.0],
} for i in common['train_conditions']]

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}



algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}


algorithm['policy_opt']= {
        'type': PolicyOptTf,
        'network_model': multi_modal_network,
        'network_params': {
            'dim_hidden': [10],
            'num_filters': [10, 20],
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],
            'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],
            'obs_image_data':[RGB_IMAGE],
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
            'sensor_dims': SENSOR_DIMS,
            'batch_size': 25,
        },
        'iterations': 500,
        'fc_only_iterations': 5000,
        'weights_file_prefix': EXP_DIR + 'policy',
    }



config = {
    'iterations': 25,
    'num_samples': 5,
    'verbose_trials': 5,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
}

common['info'] = generate_experiment_info(config)

