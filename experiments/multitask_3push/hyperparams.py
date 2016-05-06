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
from gps.algorithm.policy_opt.tf_model_example_multirobot import example_tf_network_multi, multitask_multirobot_fc
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC

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
},
{
    JOINT_ANGLES: 4,
    JOINT_VELOCITIES: 4,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 4,
},
{
    JOINT_ANGLES: 5,
    JOINT_VELOCITIES: 5,
    END_EFFECTOR_POINTS: 9,
    END_EFFECTOR_POINT_VELOCITIES: 9,
    ACTION: 3,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
}, 
{
    JOINT_ANGLES: 6,
    JOINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS: 9,
    END_EFFECTOR_POINT_VELOCITIES: 9,
    ACTION: 4,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
},
{
    JOINT_ANGLES: 3,
    JOINT_VELOCITIES: 3,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 3,
},
{
    JOINT_ANGLES: 4,
    JOINT_VELOCITIES: 4,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 4,
}]


PR2_GAINS = [np.array([1.0, 1.0, 1.0]), np.array([ 1.0, 1.0, 1.0, 1.0])]

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/multitask_3push/'
#INIT_POLICY_DIR = '/home/abhigupta/gps/'
all_offsets = [[np.asarray([0., 0., -1.4])],
               [np.asarray([0.3, 0., 0.])],
               [np.asarray([0.2, 0.0, 0.4])],
               [np.asarray([0.4, 0., -0.7])], 
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
    'test_conditions': [4,5,6,7],
    'num_robots':5,
    'policy_opt': {
        'type': PolicyOptTf,
        'network_model': multitask_multirobot_fc,
        'network_params': [{
            'dim_hidden': [10],
            'num_filters': [10, 20],
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_image_data':[],
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
            'sensor_dims': SENSOR_DIMS[0],
            'batch_size': 25,
            # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
        },
        {
            'dim_hidden': [10],
            'num_filters': [10, 20],
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_image_data':[],
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
            'sensor_dims': SENSOR_DIMS[1],
            'batch_size': 25,
            # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
        },
        # {
        #     'dim_hidden': [10],
        #     'num_filters': [10, 20],
        #     'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        #     'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        #     'obs_image_data':[],
        #     'image_width': IMAGE_WIDTH,
        #     'image_height': IMAGE_HEIGHT,
        #     'image_channels': IMAGE_CHANNELS,
        #     'sensor_dims': SENSOR_DIMS[2],
        #     'batch_size': 25,
        #     # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
        # },
        {
            'dim_hidden': [10],
            'num_filters': [10, 20],
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_image_data':[],
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
            'sensor_dims': SENSOR_DIMS[3],
            'batch_size': 25,
            # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
        },
        {
            'dim_hidden': [10],
            'num_filters': [10, 20],
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_image_data':[],
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
            'sensor_dims': SENSOR_DIMS[4],
            'batch_size': 25,
            # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
        },
        {
            'dim_hidden': [10],
            'num_filters': [10, 20],
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'obs_image_data':[],
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
            'sensor_dims': SENSOR_DIMS[5],
            'batch_size': 25,
            # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
        }],

        'iterations': 4000,
        'fc_only_iterations': 5000,
        'checkpoint_prefix': EXP_DIR + 'data_files/policy',
        # 'restore_all_wts':'/home/abhigupta/gps/allweights_push_4link.npy'
    }
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [{
    'type': AgentMuJoCo,
    'filename': './mjc_models/arm_3link_reach.xml',
    'x0': np.zeros(6),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': all_offsets,
    'pos_body_idx': np.array([6]),
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
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
    },
    {
    'type': AgentMuJoCo,
    'filename': './mjc_models/arm_4link_reach.xml',
    'x0': np.zeros(8),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': all_offsets,
    'pos_body_idx': np.array([7]),
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 100,
    'sensor_dims': SENSOR_DIMS[1],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
    },
    # {
    # 'type': AgentMuJoCo,
    # 'filename': './mjc_models/3link_gripper_push.xml',
    # 'x0': np.zeros(10),
    # 'dt': 0.05,
    # 'substeps': 5,
    # # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    # 'pos_body_offset': [
    #                     [np.array([0.0, 0.0, 0.0]),np.array([0.4, 0.0, 0.65])],
    #                     [np.array([0.0, 0.0, 0.0]),np.array([0.8, 0.0, -0.75])],
    #                     [np.array([0.0, 0.0, 0.0]),np.array([0.2, 0.0, 0.95])],
    #                     [np.array([0.0, 0.0, 0]),np.array([0.6, 0.0, -0.85])],

    #                     [np.array([0.0, 0.0, 0.0]),np.array([0.3, 0.0, 0.75])],
    #                     [np.array([0.0, 0.0, 0]),np.array([0.5, 0.0, -0.75])],
    #                     [np.array([0.0, 0.0, 0]),np.array([0.5, 0.0, 0.8])],
    #                     [np.array([0.0, 0.0, 0.0]),np.array([0.45, 0.0, -0.95])],
    #                     ],
    # 'pos_body_idx': np.array([6,8]),
    # 'conditions': 8,
    # 'train_conditions': [0,1,2 ,3 ],
    # 'test_conditions': [4,5,6,7],
    # 'image_width': IMAGE_WIDTH,
    # 'image_height': IMAGE_HEIGHT,
    # 'image_channels': IMAGE_CHANNELS,
    # 'T': 100,
    # 'sensor_dims': SENSOR_DIMS[2],
    # 'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
    #                   END_EFFECTOR_POINT_VELOCITIES],
    #                   #include the camera images appropriately here
    # 'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    # 'meta_include': [],
         # 'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
    #     },
    {
    'type': AgentMuJoCo,
    'filename': './mjc_models/4link_gripper_push.xml',
    'x0': np.zeros(12),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': [
                        [np.array([0.0, 0.0, 0.0]),np.array([0.4, 0.0, 0.65])],
                        [np.array([0.0, 0.0, 0.0]),np.array([0.8, 0.0, -0.75])],
                        [np.array([0.0, 0.0, 0.0]),np.array([0.2, 0.0, 0.95])],
                        [np.array([0.0, 0.0, 0]),np.array([0.6, 0.0, -0.85])],

                        [np.array([0.0, 0.0, 0.0]),np.array([0.3, 0.0, 0.75])],
                        [np.array([0.0, 0.0, 0]),np.array([0.5, 0.0, -0.75])],
                        [np.array([0.0, 0.0, 0]),np.array([0.5, 0.0, 0.8])],
                        [np.array([0.0, 0.0, 0.0]),np.array([0.45, 0.0, -0.95])],
                        ],
    'pos_body_idx': np.array([7,9]),
    'conditions': 8,
    'train_conditions': [0,1,2 ,3 ],
    'test_conditions': [4,5,6,7],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 100,
    'sensor_dims': SENSOR_DIMS[3],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),

         },
    {
    'type': AgentMuJoCo,
    'filename': './mjc_models/3link_peg_insert.xml',
    'x0': np.zeros(6),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': [[np.array([1.3, 0.0, 0.4])], [np.array([1.2, 0.0, 0.7])], [np.array([1.4, 0.0, 0.6])],
                        [np.array([0.8, 0.0, 1.0])], [np.array([0.6, 0.0, 1.4])], [np.array([1.4 , 0.0, 0.4])], 
                        [np.array([1.1, 0.0, 0.7])], [np.array([1.3, 0.0, 0.6])]
                        ],
    'pos_body_idx': np.array([6]),
    'conditions': 8,
    'train_conditions': [0,1,2,3],
    'test_conditions': [4,5,6,7],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 100,
    'sensor_dims': SENSOR_DIMS[4],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
        'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
},
{
    'type': AgentMuJoCo,
    'filename': './mjc_models/4link_peg_insert.xml',
    'x0': np.zeros(8),
    'dt': 0.05,
    'substeps': 5,
    # [np.array([1.2, 0.0, 0.4]),np.array([1.2, 0.0, 0.9])]
    'pos_body_offset': [[np.array([1.3, 0.0, 0.4])], [np.array([1.2, 0.0, 0.7])], [np.array([1.4, 0.0, 0.6])],
                        [np.array([0.8, 0.0, 1.0])], [np.array([0.6, 0.0, 1.4])], [np.array([1.4 , 0.0, 0.4])], 
                        [np.array([1.1, 0.0, 0.7])], [np.array([1.3, 0.0, 0.6])]
                        ],
    'pos_body_idx': np.array([7]),
    'conditions': 8,
    'train_conditions': [0,1,2,3],
    'test_conditions': [4,5,6,7],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'T': 100,
    'sensor_dims': SENSOR_DIMS[5],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
                      #include the camera images appropriately here
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
    }
]

algorithm = [{
    'type': AlgorithmBADMM,
    'conditions': agent[0]['conditions'],
    'train_conditions': agent[0]['train_conditions'],
    'test_conditions': agent[0]['test_conditions'],
    'num_robots': common['num_robots'],
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
    'init_pol_wt': 0.005,
},
{
    'type': AlgorithmBADMM,
    'conditions': agent[1]['conditions'],
    'train_conditions': agent[1]['train_conditions'],
    'test_conditions': agent[1]['test_conditions'],
    'num_robots': common['num_robots'],
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
    'init_pol_wt': 0.005,
},
# {
#     'type': AlgorithmBADMM,
#     'conditions': agent[2]['conditions'],
#     'train_conditions': agent[2]['train_conditions'],
#     'test_conditions': agent[2]['test_conditions'],
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
#     'init_pol_wt': 0.005,
# },
{
    'type': AlgorithmBADMM,
    'conditions': agent[2]['conditions'],
    'train_conditions': agent[2]['train_conditions'],
    'test_conditions': agent[2]['test_conditions'],
    'num_robots': common['num_robots'],
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
    'init_pol_wt': 0.005,
},
{
    'type': AlgorithmBADMM,
    'conditions': agent[3]['conditions'],
    'train_conditions': agent[3]['train_conditions'],
    'test_conditions': agent[3]['test_conditions'],
    'num_robots': common['num_robots'],
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
    'init_pol_wt': 0.005,
},
{
    'type': AlgorithmBADMM,
    'conditions': agent[4]['conditions'],
    'train_conditions': agent[4]['train_conditions'],
    'test_conditions': agent[4]['test_conditions'],
    'num_robots': common['num_robots'],
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
    'init_pol_wt': 0.005,
},
]


# DIFFERENT!!!!
algorithm[0]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[0][ACTION],
    'dt': agent[0]['dt'],
    'T': agent[0]['T'],
}
algorithm[1]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[1][ACTION],
    'dt': agent[1]['dt'],
    'T': agent[1]['T'],
}
algorithm[2]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[2][ACTION],
    'dt': agent[2]['dt'],
    'T': agent[2]['T'],
}
# algorithm[3]['init_traj_distr'] = {
#     'type': init_pd,
#     'init_var': 10.0,
#     'pos_gains': 10.0,
#     'dQ': SENSOR_DIMS[3][ACTION],
#     'dt': agent[3]['dt'],
#     'T': agent[3]['T'],
# }
algorithm[3]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[4][ACTION],
    'dt': agent[4]['dt'],
    'T': agent[4]['T'],
}
algorithm[4]['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'dQ': SENSOR_DIMS[5][ACTION],
    'dt': agent[4]['dt'],
    'T': agent[4]['T'],
}

torque_cost_0 = [{
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS[0],
} for i in common['train_conditions']]

fk_cost_0 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0.8, 0.0, 0.5])+ agent[0]['pos_body_offset'][i][0], np.array([0., 0., 0.])]),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in common['train_conditions']
]

algorithm[0]['cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_0[i], fk_cost_0[i]],
    'weights': [1.0, 1.0],
} for i in common['train_conditions']]


torque_cost_1 = [{
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS[1],
} for i in common['train_conditions']]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0.8, 0.0, 0.5])+ agent[1]['pos_body_offset'][i][0], np.array([0., 0., 0.])]),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in common['train_conditions']
]

algorithm[1]['cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [1.0, 1.0],
} for i in common['train_conditions']]

# torque_cost_2 = [{
#     'type': CostAction,
#     'wu': 5e-5 / PR2_GAINS[0],
# } for i in agent[2]['train_conditions']]

# fk_cost_2 = [{
#     'type': CostFK,
#     'target_end_effector': np.concatenate([np.array([0,0,0]), 
#                                            np.array([0.05, 0.05, 0.05]) + agent[2]['pos_body_offset'][i][1],
#                                            np.array([0,0,0])]),
#     'wp': np.array([0, 0, 0, 1, 1, 1,0,0,0]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
# } for i in agent[2]['train_conditions']]

# algorithm[2]['cost'] = [{
#     'type': CostSum,
#     'costs': [fk_cost_2[i], torque_cost_2[i]],
#     'weights': [1.0, 0.5],
#     'ramp_option': RAMP_FINAL_ONLY
# } for i in agent[2]['train_conditions']]


torque_cost_3 = [{
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS[1],
} for i in agent[2]['train_conditions']]

fk_cost_3 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0,0,0]), 
                                           np.array([0.05, 0.05, 0.05]) + agent[2]['pos_body_offset'][i][1],
                                           np.array([0,0,0])]),
    'wp': np.array([0, 0, 0, 1, 1, 1,0,0,0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in agent[2]['train_conditions']]

fk_cost_3_gripper = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([0.05, 0.05, 0.05]) + agent[2]['pos_body_offset'][i][1],
                                           np.array([0,0,0]), 
                                           np.array([0,0,0])]),
    'wp': np.array([1, 1, 1, 0, 0, 0, 0,0,0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'ramp_option': RAMP_QUADRATIC
} for i in agent[2]['train_conditions']]

algorithm[2]['cost'] = [{
    'type': CostSum,
    'costs': [fk_cost_3[i], fk_cost_3_gripper[i], torque_cost_3[i]],
    'weights': [1.0, 0.5, 0.5],
    'ramp_option': RAMP_QUADRATIC
} for i in agent[2]['train_conditions']]


fk_cost_4 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([agent[3]['pos_body_offset'][i][0], np.array([0,0,0])]),
    'wp': np.array([1, 1, 1,0,0,0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in agent[3]['train_conditions']]
 
algorithm[3]['cost'] = [{
    'type': CostSum,
    'costs': [ fk_cost_4[i]],
    'weights': [1.0],
} for i in agent[3]['train_conditions']]

fk_cost_5 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([agent[4]['pos_body_offset'][i][0], np.array([0,0,0])]),
    'wp': np.array([1, 1, 1,0,0,0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
} for i in agent[4]['train_conditions']]
 
algorithm[4]['cost'] = [{
    'type': CostSum,
    'costs': [ fk_cost_5[i]],
    'weights': [1.0],
} for i in agent[4]['train_conditions']]

        
for iter_var in range(5):
    algorithm[iter_var]['dynamics'] = {
        'type': DynamicsLRPrior,
        'regularization': 1e-6,
        'prior': {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 20,
        },
    }



    algorithm[iter_var]['traj_opt'] = {
        'type': TrajOptLQRPython,
        'robot_number':iter_var
    }


    algorithm[iter_var]['policy_prior'] = {
        'type': PolicyPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
        'robot_number':iter_var
    }



config = {
    'iterations': 25,
    'num_samples': 10,
    'verbose_trials': 10,
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
    'robot_iters': [range(25), range(0,25,2)],
    'to_log': [END_EFFECTOR_POINTS, JOINT_ANGLES, ACTION],
}

common['info'] = generate_experiment_info(config)