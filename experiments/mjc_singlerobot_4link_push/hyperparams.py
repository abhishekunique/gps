from __future__ import division

from datetime import datetime
import os.path
import numpy as np

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
from gps.algorithm.policy_opt.tf_model_example_multirobot import multi_input_multi_output_images_shared


IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = [{
    JOINT_ANGLES: 5,
    JOINT_VELOCITIES: 5,
    END_EFFECTOR_POINTS: 9,
    END_EFFECTOR_POINT_VELOCITIES: 9,
    ACTION: 4,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}]

PR2_GAINS = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])]

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_singlerobot_4link_push/'
INIT_POLICY_DIR = '/home/abhigupta/gps/'
common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 8,
    'train_conditions': [0,1,2,3,],
    'test_conditions':[4,5,6,7],
    'num_robots':1,
    # 'policy_opt': {
    #     'type': PolicyOptTf,
    #     'network_model': multi_input_multi_output_images_shared,
    #     'network_params': [{
    #         'dim_hidden': [10],
    #         'num_filters': [10, 20],
    #         'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],
    #         'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],
    #         'obs_image_data':[RGB_IMAGE],
    #         'image_width': IMAGE_WIDTH,
    #         'image_height': IMAGE_HEIGHT,
    #         'image_channels': IMAGE_CHANNELS,
    #         'sensor_dims': SENSOR_DIMS[0],
    #         'batch_size': 25,
    #     }],
    #     'iterations': 500,
    #     'fc_only_iterations': 5000,
    #     'checkpoint_prefix': EXP_DIR + 'data_files/policy',
    #     'restore_all_wts':'/home/abhigupta/gps/allweights_push_4link.npy'
    # }
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [{
    'type': AgentMuJoCo,
    'filename': './mjc_models/arm_4link_push.xml',
    'x0': np.zeros(10),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': [
                        [np.array([1.0, 0.0, 0.45]),np.array([0.4, 0.0, 0.45])],
                        [np.array([1.0, 0.0, -0.5]),np.array([0.6, 0.0, -0.5])],
                        [np.array([0.6, 0.0, 0.65]),np.array([0.2, 0.0, 0.65])],
                        [np.array([0.8, 0.0, -0.65]),np.array([0.6, 0.0, -0.65])],

                        [np.array([0.8, 0.0, 0.5]),np.array([0.3, 0.0, 0.5])],
                        [np.array([0.8, 0.0, -0.5]),np.array([0.4, 0.0, -0.5])],
                        [np.array([0.7, 0.0, 0.6]),np.array([0.4, 0.0, 0.6])],
                        [np.array([0.75, 0.0, -0.55]),np.array([0.45, 0.0, -0.55])],
                        ],
    'pos_body_idx': np.array([7,9]),
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 100,
    'sensor_dims': SENSOR_DIMS[0],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],
    'meta_include': [RGB_IMAGE_SIZE],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
    # 'smooth_noise': True,
    # 'smooth_noise_var': 2.0,
    # 'smooth_noise_renormalize': True,
}]

# algorithm = [{
#     'type': AlgorithmBADMM,
#     'conditions': common['conditions'],
#     'train_conditions': common['train_conditions'],
#     'test_conditions': common['test_conditions'],
#     'num_robots': common['num_robots'],
#     'iterations': 25,
#     'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
#     'policy_dual_rate': 0.2,
#     'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
#     'fixed_lg_step': 3,
#     'kl_step': 5.0,
#     'min_step_mult': 0.01,
#     'max_step_mult': 1.0,
#     'sample_decrease_var': 0.05,
#     'sample_increase_var': 0.1,
#     # 'init_pol_wt': 0.001,
# }]

algorithm = [{
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'iterations': 25,
    'num_robots': common['num_robots'],
}]

# DIFFERENT!!!!
algorithm[0]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[0][ACTION],
    'dt': agent[0]['dt'],
    'T': agent[0]['T'],
}
# algorithm[0]['init_traj_distr'] = {
#     'type': init_from_file,
#     'filename': [INIT_POLICY_DIR + ('traj_dist%d.pkl' % i)
#             for i in common['train_conditions']]
# }

# algorithm[0]['init_traj_distr'] = {
#     'type': init_lqr,
#     'init_gains':  1.0 / PR2_GAINS[1],
#     'init_acc': np.zeros(SENSOR_DIMS[0][ACTION]),
#     'init_var': 1.0,
#     'stiffness': 0.001,
#     'stiffness_vel': 0.001,
#     'dt': agent[0]['dt'],
#     'T': agent[0]['T'],
# }
torque_cost_1 = [{
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS[1],
} for i in common['train_conditions']]

# fk_cost_1 = [{
#     'type': CostFKBlock,
#     'wp': np.array([0, 0, 0, 1, 1, 1]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
# } for i in common['train_conditions']]

fkblock_cost_1 = [{
    'type': CostFKBlock,
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in common['train_conditions']]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0,0,0]), np.array([0.05, 0.05, 0.05]) + agent[0]['pos_body_offset'][i][1]]),
    'wp': np.array([0, 0, 0, 1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in common['train_conditions']]

#NO TORQUE COST!!
algorithm[0]['cost'] = [{
    'type': CostSum,
    'costs': [fkblock_cost_1[i], fk_cost_1[i]],
    'weights': [0.5, 1.0],
} for i in common['train_conditions']]


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
    'verbose_trials': 1,
    'verbose_policy_trials': 5,
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
}

common['info'] = generate_experiment_info(config)
