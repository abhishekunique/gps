from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example_multirobot import multitask_multirobot_fc

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

from enum import Enum

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

PR2_GAINS = [np.array([1.0, 1.0, 1.0]), np.array([ 1.0, 1.0, 1.0, 1.0])]

class RobotType(Enum):
    THREE_LINK = 1
    THREE_LINK_SHORT_JOINT = 2
    FOUR_LINK = 3
    PEGGY = 4
    KINOVA = 5
    def is_arm(self):
        if self == RobotType.THREE_LINK or self == RobotType.THREE_LINK_SHORT_JOINT or self == RobotType.FOUR_LINK:
            return True
        elif self == RobotType.PEGGY or self == RobotType.KINOVA:
            return False
        else:
            raise RuntimeError
    def number_links(self):
        if self == RobotType.THREE_LINK or self == RobotType.THREE_LINK_SHORT_JOINT:
            return 3
        elif self == RobotType.FOUR_LINK:
            return 4
        elif self == RobotType.PEGGY:
            return 7
        elif self == RobotType.KINOVA:
            return 9
        else:
            raise RuntimeError
    def bodies_before_color_blocks(self):
        if self.is_arm():
            return self.number_links() + 2
        elif self == RobotType.PEGGY:
            return 16
        elif self == RobotType.KINOVA:
            return 10
        else:
            raise RuntimeError
    def gains(self):
        if self.is_arm() or self == RobotType.KINOVA:
            return np.ones(self.number_links())
        elif self == RobotType.PEGGY:
            return np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
        else:
            raise RuntimeError

UNCHANGED_OBJECT_BY_COLOR = {
    "black": 4,
    "green": 2,
    "yellow": 3,
    "red": 1
}

XML_BY_ROBOT_TYPE = {
    RobotType.THREE_LINK_SHORT_JOINT : './mjc_models/arm_3link_reach_colors_shortjoint.xml',
    RobotType.THREE_LINK : './mjc_models/arm_3link_reach_colors.xml',
    RobotType.FOUR_LINK : './mjc_models/arm_4link_reach_colors.xml',
    RobotType.PEGGY : './mjc_models/pr2_arm3d_reach_colors.xml',
    RobotType.KINOVA : './mjc_models/jaco.xml'
}

def reacher_by_color_and_type(robot_number, num_robots, init_offset, offsets, color, robot_type, enable_images):
    number_links = robot_type.number_links()
    bodies_before_color_blocks = robot_type.bodies_before_color_blocks()
    SENSOR_DIMS = {
        JOINT_ANGLES: number_links,
        JOINT_VELOCITIES: number_links,
        END_EFFECTOR_POINTS: 15,
        END_EFFECTOR_POINT_VELOCITIES: 15,
        ACTION: number_links
    }
    if enable_images:
        SENSOR_DIMS.update({
            RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
            RGB_IMAGE_SIZE: 3
        })
    image_data = [RGB_IMAGE] if enable_images else []
    agent_dict= {}
    start_of_end_eff_pts = SENSOR_DIMS[JOINT_ANGLES] + SENSOR_DIMS[JOINT_VELOCITIES]
    start_of_end_eff_vel = start_of_end_eff_pts + SENSOR_DIMS[END_EFFECTOR_POINTS]
    end_of_end_eff_vel = start_of_end_eff_vel + SENSOR_DIMS[END_EFFECTOR_POINT_VELOCITIES]
    if enable_images:
        image_dims = {
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
        }
        robot_specific_indices = range(3 + start_of_end_eff_pts)+range(start_of_end_eff_vel,start_of_end_eff_vel + 3)
        task_specific_indices = range(start_of_end_eff_pts,3 + start_of_end_eff_pts) + range(start_of_end_eff_vel,start_of_end_eff_vel + 3)
    else:
        image_dims = {}
        robot_specific_indices = range(end_of_end_eff_vel)
        task_specific_indices = range(start_of_end_eff_pts, end_of_end_eff_vel)
    agent_dict['network_params']= {
        'dim_hidden': [10],
        'num_filters': [10, 20],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES] + image_data,
        'obs_image_data':image_data,
        'sensor_dims': SENSOR_DIMS,
        'batch_size': 25,
        'robot_specific_idx': robot_specific_indices,
        'task_specific_idx': task_specific_indices,
        'dim_robot_specific':len(robot_specific_indices),
        'dim_output':number_links,
        # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
    }
    agent_dict['network_params'].update(image_dims)
    agent_dict['agent'] = {
        'type': AgentMuJoCo,
        'filename': XML_BY_ROBOT_TYPE[robot_type],
        'x0': np.zeros(2 * number_links),
        'dt': 0.05,
        'substeps': 5,
        'pos_body_offset': None,
        'pos_body_idx': np.array(range(bodies_before_color_blocks + 1, bodies_before_color_blocks + 5)),
        'conditions': 8,
        'train_conditions': range(4),
        'test_conditions': range(4,8),
        'T': 100,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                          END_EFFECTOR_POINT_VELOCITIES],
        #include the camera images appropriately here
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES] + image_data,
        'meta_include': [],
        'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
        'unchanged_object': UNCHANGED_OBJECT_BY_COLOR[color] + bodies_before_color_blocks
    }
    agent_dict['agent'].update(image_dims)
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
        'init_var': 10.0, # TODO can be useful to use
        'pos_gains': 10.0,
        'dQ': SENSOR_DIMS[ACTION],
        'dt':  agent_dict['agent']['dt'],
        'T':  agent_dict['agent']['T'],
    }

    torque_cost_0 = [{
        'type': CostAction,
        'wu': 1e-1 / robot_type.gains(),
    } for i in agent_dict['agent']['train_conditions']]

    fk_cost_0 = [{
        'type': CostFK,
        'target_end_effector': np.concatenate([np.array(init_offset) + offsets[i], np.zeros(12)]),
        'wp': np.array([1, 1, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0]),
        'l1': 0.1,
        'l2': 10.0,
        'alpha': 1e-5,
    } for i in agent_dict['agent']['train_conditions']]


    agent_dict['algorithm']['cost'] = [{
        'type': CostSum,
        'costs': [torque_cost_0[i], fk_cost_0[i]],
        'weights': [1.0, 1.0],
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
