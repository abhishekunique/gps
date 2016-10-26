""" Hyperparameters for MJC peg insertion policy optimization. """
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
# from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, RAMP_MIDDLE_DRAWER


SENSOR_DIMS = [{
    JOINT_ANGLES: 4,
    JOINT_VELOCITIES: 4,
    END_EFFECTOR_POINTS: 12,
    END_EFFECTOR_POINT_VELOCITIES: 12,
    ACTION: 3,
}]

PR2_GAINS = np.array([1.0, 1.0, 1.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/3link_drawer_horizontal/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 8,
    'train_conditions': [0],
    'test_conditions': [1,2,3,4,5,6,7],
    'num_robots':1
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [{
    'type': AgentMuJoCo,
    'filename': './mjc_models/3link_drawer_horizontal.xml',
    'x0': np.concatenate([np.array([np.pi/8, 0.0, 0.0, 0.0]), np.zeros((4,))]),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': [
                        # [np.array([1.2, 0.0, -0.75]),np.array([1.2, 0.0, -0.35])],
                        [np.array([-0.25, 0.0, -1.25]),np.array([-0.25, 0.0, -0.45])],
                        [np.array([-0.25, 0.0, 0.85]),np.array([-0.25, 0.0, 0.45])],
                        [np.array([-0.25, 0.0, 1.15]),np.array([-0.25, 0.0, 0.45])],
                        [np.array([1.2, 0.0, -0.75]),np.array([1.2, 0.0, -0.35])],
                        [np.array([0.0, 0.0, -0.85]),np.array([0.0, 0.0, -0.55])],
                        [np.array([0.0, 0.0, -1.15]),np.array([0.0, 0.0, -0.55])],
                        [np.array([0.0, 0.0, 0.85]),np.array([0.0, 0.0, 0.55])],
                        [np.array([0.0, 0.0, 1.15]),np.array([0.0, 0.0, 0.55])],
                        ],
    'pos_body_idx': np.array([6,8]),
    'conditions': 8,
    'train_conditions': [0],
    'test_conditions': [1,2,3, 4,5,6,7],
    'T': 150,
    'sensor_dims': SENSOR_DIMS[0],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
    'smooth_noise': True,
    'smooth_noise_var': 1.0,
    'smooth_noise_renormalize': True,
}]

algorithm = [{
    'type': AlgorithmTrajOpt,
    'conditions': 8,
    'train_conditions': [0],
    'test_conditions': [1,2,3, 4,5,6,7],
    'iterations': 25,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
}]

algorithm[0]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 40.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[0][ACTION],
    'dt': agent[0]['dt'],
    'T': agent[0]['T'],
}

torque_cost_0 = [{
    'type': CostAction,
    'wu': 1e-1 / PR2_GAINS,
} for i in agent[0]['train_conditions']]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0,0,0]), np.array([0,0,0]),
                                           agent[0]['pos_body_offset'][i][1],
                                           np.array([0,0,0])]),
    'wp': np.array([0, 0, 0, 0, 0, 0, 1, 1, 1,0,0,0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'ramp_option': RAMP_QUADRATIC
} for i in agent[0]['train_conditions']]

fk_cost_2 = [
#{
#         'type': CostFK,
#         'target_end_effector': np.concatenate([np.array([-0.15, 0.0, -0.95]),  
#                                                np.array([0.05, 0.05, 0.05]),
#                                                np.array([0,0,0]),np.array([0,0,0])]),
#         'wp': np.array([1, 1, 1, 0, 0, 0,0,0,0,0,0,0]),
#         'l1': 0.1,
#         'l2': 10.0,
#         'alpha': 1e-5,
#         'ramp_option': RAMP_MIDDLE_DRAWER
#     },
    {
        'type': CostFK,
        'target_end_effector': np.concatenate([np.array([-0.15, 0.0, -1.35]), np.array([-0.15, 0.0, -1.75]), 
                                               np.array([0.05, 0.05, 0.05]),
                                               np.array([0,0,0])]),
        'wp': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0,0,0,0]),
        'l1': 0.1,
        'l2': 10.0,
        'alpha': 1e-5,
        'ramp_option': RAMP_MIDDLE_DRAWER
    }
    
#     {
#         'type': CostFK,
#         'target_end_effector': np.concatenate([np.array([-0.15, 0.0, 1.35]), np.array([-0.15, 0.0, 0.95]), 
#                                                np.array([0.05, 0.05, 0.05]),
#                                                np.array([0,0,0])]),
#         'wp': np.array([1, 1, 5, 1, 1, 5, 0, 0, 0,0,0,0]),
#         'l1': 0.1,
#         'l2': 10.0,
#         'alpha': 1e-5,
#         'ramp_option': RAMP_MIDDLE_DRAWER
#     },
#     {
#         'type': CostFK,
#         'target_end_effector': np.concatenate([np.array([-0.15, 0.0, 1.75]), np.array([-0.15, 0.0, 1.35]), 
#                                                np.array([0.05, 0.05, 0.05]),
#                                                np.array([0,0,0])]),
#         'wp': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0,0,0,0]),
#         'l1': 0.1,
#         'l2': 10.0,
#         'alpha': 1e-5,
#         'ramp_option': RAMP_MIDDLE_DRAWER
# }
]

# demo_waypoints = np.load("demo_waypoints.npy")
# fk_cost_2 = [{
#     'type': CostFK,
#     'target_end_effector': demo_waypoints[i],
#     'wp': np.array([1, 1, 1, 0,0,0, 0, 0, 0]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
#     'ramp_option': RAMP_QUADRATIC
# } for i in agent[0]['train_conditions']]


algorithm[0]['cost'] = [{
        'type': CostSum,
        'costs': [fk_cost_1[i],torque_cost_0[i]],
        'weights': [3.0, 1.0],
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
}


algorithm[0]['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm[0]['iterations'],
    'num_samples': 5,
    'verbose_trials': 5,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': 8,
    'train_conditions': [0],
    'test_conditions': [1,2,3, 4,5,6,7],
    'to_log': []
}

common['info'] = generate_experiment_info(config)
