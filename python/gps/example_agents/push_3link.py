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

PR2_GAINS = [np.array([1.0, 1.0, 1.0]), np.array([ 1.0, 1.0, 1.0, 1.0])]

def push_3link(robot_number, num_robots):
    SENSOR_DIMS = {
        JOINT_ANGLES: 5,
        JOINT_VELOCITIES: 5,
        END_EFFECTOR_POINTS: 9,
        END_EFFECTOR_POINT_VELOCITIES: 9,
        ACTION: 3,
        RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
        RGB_IMAGE_SIZE: 3,
    }
    agent_dict= {}
    agent_dict['network_params']= {
        'dim_hidden': [10],
        'num_filters': [10, 20],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_image_data':[RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
        'batch_size': 25,
        'robot_specific_idx': range(3)+range(5,8)+range(10,13)+range(19,22),
        'task_specific_idx': range(10,13) + range(19,22),
        'dim_output':3,
        # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
    }
    agent_dict['agent'] = {
        'type': AgentMuJoCo,
        'filename': './mjc_models/3link_gripper_push.xml',
        'x0': np.zeros(10),
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
        'pos_body_idx': np.array([6,8]),
        'conditions': 8,
        'train_conditions': [0,1,2 ,3 ],
        'test_conditions': [4,5,6,7],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'T': 100,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                          END_EFFECTOR_POINT_VELOCITIES],
        #include the camera images appropriately here
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE],
        'meta_include': [],
        'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
    }
    agent_dict['algorithm'] = {
        'type': AlgorithmBADMM,
        'conditions': agent_dict['agent']['conditions'],
        'train_conditions': agent_dict['agent']['train_conditions'],
        'test_conditions': agent_dict['agent']['test_conditions'],
        'num_robots': num_robots,
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
    }
    agent_dict['algorithm']['init_traj_distr'] = {
        'type': init_pd,
        'init_var': 10.0,
        'pos_gains': 10.0,
        'dQ': SENSOR_DIMS[ACTION],
        'dt':  agent_dict['agent']['dt'],
        'T':  agent_dict['agent']['T'],
    }

    torque_cost_0 = [{
        'type': CostAction,
        'wu': 5e-5 / PR2_GAINS[0],
    } for i in agent_dict['agent']['train_conditions']]

    fk_cost_0 = [{
        'type': CostFK,
        'target_end_effector': np.concatenate([np.array([0,0,0]), 
                                               np.array([0.05, 0.05, 0.05]) + agent_dict['agent']['pos_body_offset'][i][1],
                                               np.array([0,0,0])]),
        'wp': np.array([0, 0, 0, 1, 1, 1,0,0,0]),
        'l1': 0.1,
        'l2': 10.0,
        'alpha': 1e-5,
        'ramp_option': RAMP_QUADRATIC
    } for i in agent_dict['agent']['train_conditions']]

    fk_cost_gripper = [{
        'type': CostFK,
        'target_end_effector': np.concatenate([np.array([0.05, 0.05, 0.05]) +agent_dict['agent']['pos_body_offset'][i][1],
                                           np.array([0,0,0]), 
                                           np.array([0,0,0])]),
        'wp': np.array([1, 1, 1, 0, 0, 0, 0,0,0]),
        'l1': 0.1,
        'l2': 10.0,
        'alpha': 1e-5,
        'ramp_option': RAMP_QUADRATIC
    } for i in agent_dict['agent']['train_conditions']]

    agent_dict['algorithm']['cost'] = [{
        'type': CostSum,
        'costs': [torque_cost_0[i], fk_cost_0[i], fk_cost_gripper[i]],
        'weights': [0.5,1.0, 0.5],
    } for i in agent_dict['agent']['train_conditions']]

    agent_dict['algorithm']['dynamics'] = {
        'type': DynamicsLRPrior,
        'regularization': 1e-6,
        'prior': {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 20,
        },
}

    agent_dict['algorithm']['traj_opt'] = {
        'type': TrajOptLQRPython,
        'robot_number': robot_number
    }

    agent_dict['algorithm']['policy_prior'] = {
        'type': PolicyPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
        'robot_number': robot_number
    }
    return agent_dict
 