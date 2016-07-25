""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_dev_rs_3d import CostDevRs3D
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_fk_dev import CostFKDev

from gps.algorithm.cost.cost_sum_decreasing import CostSumDecrease
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example_multirobot import example_tf_network_multi, model_fc_shared
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC
from gps.utility.data_logger import DataLogger

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 18,
    END_EFFECTOR_POINT_VELOCITIES: 18,
    ACTION: 5,
}

PR2_GAINS = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/3d_push_4link/'


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
    'num_robots': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [{
    'type': AgentMuJoCo,
    'filename': './mjc_models/3d_pusher_4link.xml',
    'x0': np.concatenate([np.zeros(7), #np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([7,8,9,10]),
    'pos_body_offset': [[np.array([0.0, 0.5, -0.4]), np.array([0.8, 0.5, -0.4]), np.array([0.0, 1.1, -0.4]),  np.array([0.8, 1.1, -0.4])],
                        [np.array([0, 0.1, 0]), np.array([0, 0.2, 0]), np.array([1.0, 0.5, 0.8]), np.array([0.6, 0.5, 0.4])]                        # np.array([-.1, -0.2, 0]),
                        # np.array([0, 0.1, 0.2]),
                        # np.array([0, -0.1, 0]),
                        # np.array([0, -0.2, 0]),
                        # np.array([0.1, -0.1, 0]),
                        # np.array([0, 0.2, -0.1])
                    ],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
}]

algorithm = [{
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 25,
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
}]

algorithm[0]['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': .5,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent[0]['dt'],
    'T': agent[0]['T'],
}

fk_cost_blocktouch = {
    'type': CostFKBlock,
    # 'wp': np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.concatenate([np.zeros(3), np.zeros(3), agent[0]['pos_body_offset'][0][2], agent[0]['pos_body_offset'][0][3], np.zeros(3), np.zeros(3)]),
    'wp': np.array([0,0,0, 0,0,0, 1, 1, 1, 1, 1, 1, 0,0,0, 0,0,0,]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'ramp_option': RAMP_QUADRATIC
}

# data_logger = DataLogger()
# data_traj = data_logger.unpickle('/home/abhigupta/gps/experiments/3d_push_3link/data_files/traj_sample_itr_24_rn_00.pkl')
# # import IPython
# # IPython.embed()
# fk_cost_2 = {
#     'type': CostFKDev,
#     'traj': data_traj[0][0]._data[END_EFFECTOR_POINTS],
#     'wp': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
# }

load_trajs = np.load('/home/abhigupta/gps/experiments/3d_push_3link/data_files/fps_16_rn_00.pkl.npy')
test_cost = {
    'type': CostDevRs3D,
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'target_feats': load_trajs[0][0],
    'load_file': '/home/abhigupta/gps/subspace_3dwts_v2.pkl',
}


algorithm[0]['cost'] = {
    'type': CostSum,
    'costs': [fk_cost],
    'weights': [1.0],
}

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

algorithm[0]['policy_opt'] = {}

config = {
    'iterations': algorithm[0]['iterations'],
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'num_samples': 10,
    'verbose_trials': 10,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'to_log': [ACTION, END_EFFECTOR_POINTS, JOINT_ANGLES],
}

common['info'] = generate_experiment_info(config)