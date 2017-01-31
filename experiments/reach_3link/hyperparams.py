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
from gps.algorithm.policy_opt.tf_model_example import example_tf_network


IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 3,
}

PR2_GAINS = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])]
BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/reach_3link/'

#close to the blockstrike positions
# all_offsets = [[np.array([-0.8, 0.0, 0.25])],
#                 [np.array([-0.8, 0.0, -1.3])],
#                 [np.array([0.0, 0.0, 0.75])],
#                 [np.array([0.1, 0.0, -0.75])]]


all_offsets = [[np.asarray([-0.3, 0., -1.5])],
               [np.asarray([0.3, 0., 0.3])],
               [np.asarray([-0.4, 0.0, 0.6])],
               [np.asarray([0.3, 0., -1.2])], 
               [np.asarray([.5, 0.0, 0.3])],
               [np.asarray([.7, 0.0, -0.3])],
               [np.array([0., 0., -1.2])],
               [np.array([0.4, 0., -0.9])]]

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 8,
    'train_conditions': [0,1,2,3],
    'test_conditions':[4,5,6,7],
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/mjc_3link_reach.xml',
    'x0': np.zeros(6),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': all_offsets,
    'pos_body_idx': np.array([6]),
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
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
    'sample_increase_var': 0.1,
}


algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 1.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}




torque_cost_1 = [{
    'type': CostAction,
    'wu': 5e-1 / PR2_GAINS[0],
} for i in common['train_conditions']]

ee_traj = np.load("arm_tip_traj.pkl")
# fk_cost_1 = [{
#     'type': CostFK,
#     'target_end_effector': np.concatenate([np.array([0.8, 0.0, 0.5])+ agent['pos_body_offset'][i][0], np.array([0., 0., 0.])]),
#     'wp': np.array([1, 1, 1, 0, 0, 0]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
# } for i in common['train_conditions']
# ]
fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.mean(ee_traj[i], axis=0),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in common['train_conditions']
]

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

algorithm['policy_opt'] = {
        'type': PolicyOptTf,
        'network_model': example_tf_network,
        'run_feats': False,
        'load_weights': False,
        'network_params': {
            'dim_hidden': [10],
            'num_filters': [10, 20],
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_image_data':[],
            'sensor_dims': SENSOR_DIMS,
            'batch_size': 25,
            # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
        },
        'iterations': 2000,
        'checkpoint_prefix': EXP_DIR + 'data_files/policy',
    }

config = {
    'iterations': 25,
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'save_wts': True,
    'agent': agent,
    'gui_on': False,
    'verbose_policy_trials': 5,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'inner_iterations': 4,
    'to_log': [],
}

common['info'] = generate_experiment_info(config)
