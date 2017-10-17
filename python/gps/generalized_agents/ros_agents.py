from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps.agent.ros.agent_ros import AgentROS
from gps.agent.recorded.agent_recorded import AgentRecorded
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example_multirobot import multitask_multirobot_fc

from enum import Enum

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, IMAGE_FEAT, ACTION, \
    TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE

CAMERA_POS = [0, 5., 0., 0.3, 0., 0.3]
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info

class BlockPush(object):
    additional_joints = 2
    number_end_effectors = 3
    cost_weights = [1, 10, 5]
    camera_pos = CAMERA_POS
    @staticmethod
    def body_indices(robot_type):
        start = robot_type.bodies_before_color_blocks()
        return [start + 1, start + 3]
    @staticmethod
    def nconditions(n_offs, n_verts, n_blocks):
        del n_offs, n_verts
        return n_blocks
    @classmethod
    def offset_generator(cls, offsets, vert_offs, block_locs, condition):
        condition = condition % cls.nconditions(len(offsets), len(vert_offs), len(block_locs))
        return block_locs[condition]
    @staticmethod
    def target(robot_type):
        filename = {
            RobotType.PR2 : './targets/sweep.npz',
        }[robot_type]
        return filename
    @classmethod
    def task_specific_cost(cls, targets, train_conditions):
        return [[{
            'type': CostFK,
            'target_end_effector': targets[i],
            'wp': np.array([1,1,1,1,1,1,1, 1, 1]),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }] for i in train_conditions]

def to_cartesian(r, theta):
    return np.array([np.cos(theta), 0, np.sin(theta)]) * r

class RobotType(Enum):
    THREE_DF_BLOCK = 0
    THREE_LINK = 1
    THREE_LINK_SHORT_JOINT = 2
    FOUR_LINK = 3
    FIVE_LINK = 3.5
    PEGGY = 4
    KINOVA = 5
    BAXTER = 6
    BAXTER_CYAN = 7
    PR2 = 8
    PR2_WIDER = 8.5
    PR2_MAGENTA = 9
    FOUR_SIX = 10
    FOUR_SEVEN = 11
    def is_arm(self):
        if self in {RobotType.THREE_LINK, RobotType.THREE_LINK_SHORT_JOINT, RobotType.FOUR_LINK, RobotType.FIVE_LINK}:
            return True
        elif self in {RobotType.PEGGY, RobotType.THREE_DF_BLOCK, RobotType.FOUR_SIX, RobotType.FOUR_SEVEN, RobotType.KINOVA, RobotType.BAXTER, RobotType.BAXTER_CYAN, RobotType.PR2, RobotType.PR2_WIDER, RobotType.PR2_MAGENTA}:
            return False
        else:
            raise RuntimeError
    def number_links(self):
        if self == RobotType.THREE_LINK or self == RobotType.THREE_LINK_SHORT_JOINT:
            return 3
        elif self == RobotType.FOUR_LINK:
            return 4
        elif self == RobotType.FIVE_LINK:
            return 5
        elif self == RobotType.PEGGY:
            return 7
        elif self == RobotType.KINOVA:
            return 9
        elif self in {RobotType.BAXTER, RobotType.BAXTER_CYAN}:
            return 10
        elif self in {RobotType.PR2, RobotType.PR2_WIDER, RobotType.PR2_MAGENTA}:
            return 7
        elif self == RobotType.THREE_DF_BLOCK:
            return 3
        elif self == RobotType.FOUR_SIX:
            return 6
        elif self == RobotType.FOUR_SEVEN:
            return 7
        else:
            raise RuntimeError
    def bodies_before_color_blocks(self):
        if self.is_arm():
            return self.number_links() + 2
        elif self == RobotType.FOUR_SIX or self == RobotType.FOUR_SEVEN:
            return 6
        elif self == RobotType.THREE_DF_BLOCK:
            return 2
        elif self == RobotType.PEGGY:
            return 16
        elif self == RobotType.KINOVA:
            return 10
        elif self in {RobotType.BAXTER, RobotType.BAXTER_CYAN}:
            return 21
        elif self in {RobotType.PR2, RobotType.PR2_WIDER, RobotType.PR2_MAGENTA}:
            return 11
        else:
            raise RuntimeError
    def gains(self):
        if self.is_arm() or self in {RobotType.KINOVA, RobotType.BAXTER, RobotType.BAXTER_CYAN, RobotType.FOUR_SIX, RobotType.THREE_DF_BLOCK}:
            return np.ones(self.number_links())
        elif self == RobotType.FOUR_SEVEN:
            return np.array([1] * 6 + [1e-3])
        elif self in {RobotType.PEGGY, RobotType.PR2, RobotType.PR2_WIDER, RobotType.PR2_MAGENTA}:
            return np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
        else:
            raise RuntimeError

COLOR_ORDER = ("red", "green", "yellow", "black", "magenta", "cyan")

def ros_agent(robot_number, num_robots, is_3d, offsets, vert_offs, lego_offsets, blockpush_locations, (robot_type, is_real), enable_images, task_type, torque_costs, pass_environment_effectors_to_robot=False, number_samples=None, IMAGE_WIDTH=80, IMAGE_HEIGHT=64, IMAGE_CHANNELS=3):
    number_links = robot_type.number_links()
    number_joints = number_links #+ task_type.additional_joints
    end_effector_points = 3 * 3#task_type.number_end_effectors
    SENSOR_DIMS = {
        JOINT_ANGLES: number_joints,
        JOINT_VELOCITIES: number_joints,
        END_EFFECTOR_POINTS: end_effector_points,
        END_EFFECTOR_POINT_VELOCITIES: end_effector_points,
        ACTION: number_links
    }
    if enable_images:
        SENSOR_DIMS.update({
            RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
            RGB_IMAGE_SIZE: 3,
            IMAGE_FEAT: 0
        })
    image_data = [RGB_IMAGE] if enable_images else []

    start_of_robot_vel = SENSOR_DIMS[JOINT_ANGLES]
    start_of_end_eff_pts = start_of_robot_vel + SENSOR_DIMS[JOINT_VELOCITIES]
    start_of_end_eff_vel = start_of_end_eff_pts + SENSOR_DIMS[END_EFFECTOR_POINTS]
    end_of_end_eff_vel = start_of_end_eff_vel + SENSOR_DIMS[END_EFFECTOR_POINT_VELOCITIES]

    robot_joint_idx = range(number_links) + range(start_of_robot_vel, start_of_robot_vel + number_links)
    env_joint_idx = range(number_links, start_of_robot_vel) + range(start_of_robot_vel + number_links, start_of_end_eff_pts)
    robot_end_effector_idx = range(start_of_end_eff_pts, start_of_end_eff_pts + 3) + range(start_of_end_eff_vel, start_of_end_eff_vel + 3)
    env_end_effector_idx = range(start_of_end_eff_pts + 3, start_of_end_eff_vel) + range(start_of_end_eff_vel + 3, end_of_end_eff_vel)

    robot_specific_indices = robot_joint_idx + robot_end_effector_idx
    task_specific_indices = robot_end_effector_idx
    if enable_images:
        image_dims = {
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'image_channels': IMAGE_CHANNELS,
        }
    else:
        image_dims = {}
        if pass_environment_effectors_to_robot:
            robot_specific_indices += env_end_effector_idx
        task_specific_indices += env_joint_idx + env_end_effector_idx
    robot_specific_indices, task_specific_indices = map(sorted, (robot_specific_indices, task_specific_indices))

    EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, -0.05],
                          [0.02, 0.05, 0.0]])

    SENSOR_DIMS = {
        JOINT_ANGLES: 7,
        JOINT_VELOCITIES: 7,
        END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
        END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
        ACTION: 7,
    }

    PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
    x0s = []
    ee_tgts = []
    reset_conditions = []
    nconditions = task_type.nconditions(len(offsets), len(vert_offs), len(blockpush_locations))
    print "conds", nconditions
    common = {
        'target_filename': task_type.target(robot_type),
        'conditions': nconditions*2,
    }
    for i in xrange(common['conditions']):

        ja_x0, ee_pos_x0, ee_rot_x0 = load_pose_from_npz(
            common['target_filename'], 'trial_arm', str(i), 'initial'
        )
        ja_aux, _, _ = load_pose_from_npz(
            common['target_filename'], 'auxiliary_arm', str(i), 'initial'
        )
        _, ee_pos_tgt, ee_rot_tgt = load_pose_from_npz(
            common['target_filename'], 'trial_arm', str(i), 'target'
        )

        x0 = np.zeros(32)
        x0[:7] = ja_x0
        x0[14:(14+9)] = np.ndarray.flatten(
            get_ee_points(EE_POINTS, ee_pos_x0, ee_rot_x0).T
        )

        ee_tgt = np.ndarray.flatten(
            get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
        )

        aux_x0 = np.zeros(7)
        aux_x0[:] = ja_aux

        reset_condition = {
            TRIAL_ARM: {
                'mode': JOINT_SPACE,
                'data': x0[0:7],
            },
            AUXILIARY_ARM: {
                'mode': JOINT_SPACE,
                'data': aux_x0,
            },
        }

        x0s.append(x0)
        ee_tgts.append(ee_tgt)
        reset_conditions.append(reset_condition)
    print "eetgts", len(ee_tgts)
    agent_dict = {}
    agent_dict['network_params']= {
        'dim_hidden': [10],
        'num_filters': [10, 20],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES] + image_data,
        'obs_image_data': image_data,
        'sensor_dims': SENSOR_DIMS,
        'batch_size': 25,
        'robot_specific_idx': robot_specific_indices,
        'task_specific_idx': task_specific_indices,
        'dim_robot_specific':len(robot_specific_indices),
        'dim_output':number_links,
        # 'dim_input': reduce(operator.mul, [SENSOR_DIMS[0][s] for s in OBS_INCLUDE]),
    }
    agent_dict['network_params'].update(image_dims)
    offset_generator = lambda condition: task_type.offset_generator(offsets, vert_offs, blockpush_locations, condition)
    agent_dict['agent'] = {
        'type': AgentROS,
        'dt': 0.05,
        #'conditions': common['conditions'],
        'filename': 'pr2_ros',
        'T': 100,
        'x0': x0s,
        'ee_points_tgt': ee_tgts,
        'reset_conditions': reset_conditions,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                          END_EFFECTOR_POINT_VELOCITIES],
        'end_effector_points': EE_POINTS,
        'conditions': nconditions * 2,
        'train_conditions': range(nconditions),
        'test_conditions': range(nconditions, nconditions * 2),
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES] + image_data,
        'meta_include': [],
    }
    if is_real:
        assert isinstance(task_type, LegoReach)
        agent_dict['agent']['type'] = AgentRecorded
        truecolor = "blue" if task_type.color == "black" else task_type.color
        agent_dict['agent']['real_obs_path'] = "/home/abhigupta/output/result_" + truecolor
        agent_dict["agent"]['number_samples'] = number_samples
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
        'init_var': 10.0, # TODO can be useful to use (50 in blockpush)
        'pos_gains': 10.0,
        'dQ': SENSOR_DIMS[ACTION],
        'dt':  agent_dict['agent']['dt'],
        'T':  agent_dict['agent']['T'],
    }

    torque_cost_0 = [{
        'type': CostAction,
        'wu': 1e-3 / robot_type.gains(),
    } for i in agent_dict['agent']['train_conditions']]


    agent_dict['algorithm']['cost'] = [{
        'type': CostSum,
        'costs': ([torque_cost_0[i]] if torque_costs else []) + task_type.task_specific_cost(ee_tgts, agent_dict['agent']['train_conditions'])[i],
        'weights': task_type.cost_weights if torque_costs else task_type.cost_weights[1:],
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
