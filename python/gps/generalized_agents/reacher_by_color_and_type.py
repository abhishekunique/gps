from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
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
    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, IMAGE_FEAT, ACTION

class BlockPush(object):
    additional_joints = 2
    number_end_effectors = 3
    cost_weights = [0.5, 0.5, 4]
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
    def xml(is_3d, robot_type):
        filename = {
            RobotType.THREE_LINK : './mjc_models/3link_gripper_push',
            RobotType.THREE_LINK_SHORT_JOINT : './mjc_models/3link_gripper_push_shortjoint',
            RobotType.FOUR_LINK : './mjc_models/4link_gripper_push',
            RobotType.FIVE_LINK : './mjc_models/5link_gripper_push',
            RobotType.PR2 : './mjc_models/pr2/pr2_arm_blockpush',
            RobotType.PEGGY : './mjc_models/peggy_arm3d_blockpush',
        }[robot_type]
        if robot_type.is_arm() and is_3d:
            filename += "_3d"
        return filename + ".xml"
    @classmethod
    def task_specific_cost(cls, offset_generator, train_conditions):
        return [[{
            'type': CostFK,
            'target_end_effector': np.concatenate([np.array([0,0,0]),
                                                   offset_generator(i)[1],
                                                   np.array([0,0,0])]),
            'wp': np.array([0, 0, 0, 1, 1, 1,0,0,0]),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }, {
            'type': CostFKBlock,
            'wp': np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }] for i in train_conditions]

class ColorReach(object):
    cost_weights = [1, 1]
    additional_joints = 0
    number_end_effectors = 5
    def __init__(self, color):
        self.color = color
    @staticmethod
    def body_indices(robot_type):
        start = robot_type.bodies_before_color_blocks()
        return range(start + 1, start + 5)
    @staticmethod
    def nconditions(n_offs, n_verts, _):
        return n_offs * n_verts
    def offset_generator(self, offsets, vert_offs, _, condition):
        num_offsets = len(offsets)
        num_verts = len(vert_offs)
        vert_idx = condition % num_verts
        cond_idx = (condition // num_verts) % num_offsets
        vertical = [np.random.choice(vert_offs) for _ in range(len(offsets))]
        vertical[cond_idx] = vert_offs[vert_idx]
        to_shuffle = [i for i in range(num_offsets) if i != cond_idx]
        unchanged = COLOR_ORDER.index(self.color)
        shuffled = list(to_shuffle)
        np.random.shuffle(shuffled)
        indices = np.arange(num_offsets)
        indices[to_shuffle] = shuffled
        indices[[unchanged, cond_idx]] = indices[[cond_idx, unchanged]]
        return [offsets[i] + [0, vertical[i], 0] for i in indices]
    @staticmethod
    def xml(is_3d, robot_type):
        filename = {
            RobotType.THREE_LINK_SHORT_JOINT : './mjc_models/arm_3link_reach_colors_shortjoint',
            RobotType.THREE_LINK : './mjc_models/arm_3link_reach_colors',
            RobotType.FOUR_LINK : './mjc_models/arm_4link_reach_colors',
            RobotType.FIVE_LINK : './mjc_models/arm_5link_reach_colors',
            RobotType.PEGGY : './mjc_models/peggy_arm3d_reach_colors',
            RobotType.KINOVA : './mjc_models/kinova/jaco',
            RobotType.BAXTER : './mjc_models/baxter/baxter',
            RobotType.PR2 : './mjc_models/pr2/pr2_arm'
        }[robot_type]
        if robot_type.is_arm() and is_3d:
            filename += "_3d"
        return filename + ".xml"
    def task_specific_cost(self, offset_generator, train_conditions):
        zero_padding = self.number_end_effectors * 3 - 3
        target_effector_index = COLOR_ORDER.index(self.color)
        return[[{
            'type': CostFK,
            'target_end_effector': np.concatenate([offset_generator(i)[target_effector_index], np.zeros(zero_padding)]),
            'wp': np.array([1, 1, 1] + [0] * zero_padding),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }] for i in train_conditions]

class ColorPush(ColorReach):
    additional_joints = 8
    cost_weights = BlockPush.cost_weights
    def __init__(self, color_to, color_from):
        ColorReach.__init__(self, color_to)
        self.color_from = color_from
    @staticmethod
    def xml(is_3d, robot_type):
        filename = ColorReach.xml(is_3d, robot_type)
        assert "reach" in filename
        return filename.replace("reach", "push")
    def task_specific_cost(self, offset_generator, train_conditions):
        movable_block = COLOR_ORDER.index(self.color_from)
        target_block = COLOR_ORDER.index(self.color)
        costs = []
        for i in train_conditions:
            final_locations = offset_generator(i)
            final_locations[movable_block] = final_locations[target_block]
            costs.append([
                {
                    'type': CostFK,
                    'target_end_effector': np.concatenate([[0, 0, 0]] + final_locations),
                    'wp': np.array([0] * 3 + [1] * (len(final_locations) * 3)),
                    'l1': 0.1,
                    'l2': 10.0,
                    'alpha': 1e-5,
                }, {
                    'type': lambda hyper: CostFKBlock(hyper, first_effector=0, second_effector=movable_block),
                    'wp': np.array([1] * 3 + [0] * (len(final_locations) * 3)),
                    'l1': 0.1,
                    'l2': 10.0,
                    'alpha': 1e-5,
                }
            ])
        return costs

class RobotType(Enum):
    THREE_LINK = 1
    THREE_LINK_SHORT_JOINT = 2
    FOUR_LINK = 3
    FIVE_LINK = 3.5
    PEGGY = 4
    KINOVA = 5
    BAXTER = 6
    PR2 = 7
    def is_arm(self):
        if self in {RobotType.THREE_LINK, RobotType.THREE_LINK_SHORT_JOINT, RobotType.FOUR_LINK, RobotType.FIVE_LINK}:
            return True
        elif self in {RobotType.PEGGY, RobotType.KINOVA, RobotType.BAXTER, RobotType.PR2}:
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
        elif self == RobotType.BAXTER:
            return 10
        elif self == RobotType.PR2:
            return 7
        else:
            raise RuntimeError
    def bodies_before_color_blocks(self):
        if self.is_arm():
            return self.number_links() + 2
        elif self == RobotType.PEGGY:
            return 16
        elif self == RobotType.KINOVA:
            return 10
        elif self == RobotType.BAXTER:
            return 21
        elif self == RobotType.PR2:
            return 11
        else:
            raise RuntimeError
    def gains(self):
        if self.is_arm() or self in {RobotType.KINOVA, RobotType.BAXTER}:
            return np.ones(self.number_links())
        elif self in {RobotType.PEGGY, RobotType.PR2}:
            return np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
        else:
            raise RuntimeError

COLOR_ORDER = ("red", "green", "yellow", "black")

def reacher_by_color_and_type(robot_number, num_robots, is_3d, offsets, vert_offs, blockpush_locations, robot_type, enable_images, task_type, pass_environment_effectors_to_robot=False):
    number_links = robot_type.number_links()
    number_joints = number_links + task_type.additional_joints
    end_effector_points = 3 * task_type.number_end_effectors
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
    nconditions = task_type.nconditions(len(offsets), len(vert_offs), len(blockpush_locations))
    offset_generator = lambda condition: task_type.offset_generator(offsets, vert_offs, blockpush_locations, condition)
    agent_dict['agent'] = {
        'type': AgentMuJoCo,
        'filename': task_type.xml(is_3d, robot_type),
        'x0': np.zeros(2 * number_joints), # TODO start with pi/2?
        'dt': 0.05,
        'substeps': 5,
        'pos_body_offset': None,
        'pos_body_idx': np.array(task_type.body_indices(robot_type)),
        'conditions': nconditions * 2,
        'train_conditions': range(nconditions),
        'test_conditions': range(nconditions, nconditions * 2),
        'T': 100,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                          END_EFFECTOR_POINT_VELOCITIES],
        #include the camera images appropriately here
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES] + image_data,
        'meta_include': [],
        'camera_pos': np.array([0, 5., 0., 0.3, 0., 0.3]),
        'offs_to_use': offset_generator
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
        'costs': [torque_cost_0[i]] + task_type.task_specific_cost(offset_generator, agent_dict['agent']['train_conditions'])[i],
        'weights': task_type.cost_weights,
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
