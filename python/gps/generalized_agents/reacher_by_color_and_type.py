from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps.agent.mjc.agent_mjc import AgentMuJoCo
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
    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, IMAGE_FEAT, ACTION

CAMERA_POS = [0, 5., 0., 0.3, 0., 0.3]

RYG = "red", "yellow", "green"

class BlockPush(object):
    additional_joints = 2
    number_end_effectors = 5
    cost_weights = [1, 10, 5]
    camera_pos = CAMERA_POS
    def __init__(self, color, initial_angles, diff_angles, inner_radius, diff_radius, z_location):
        self.color = color
        self.initial_angles = initial_angles
        self.diff_angles = diff_angles
        self.inner_radius = inner_radius
        self.diff_radius = diff_radius
        self.z_location = z_location
    @staticmethod
    def body_indices(robot_type):
        start = robot_type.bodies_before_color_blocks()
        return [start + 1] + range(start + 3, start + 6)
    def nconditions(self, n_offs, n_verts, n_blocks):
        del n_offs, n_verts
        return self.ninitials * self.nvels
    @staticmethod
    def modify_initial_state(state, _):
        # state[:len(state) // 2 - 2] += np.pi/4
        return state
    @property
    def nvels(self):
        return len(self.diff_angles)
    @property
    def ninitials(self):
        return len(self.initial_angles)
    def vel_index(self, condition):
        return condition % self.nvels
    def ini_index(self, condition):
        return (condition // self.nvels) % self.ninitials
    def offset_generator(self, offsets, vert_offs, block_locs, condition):
        vel_index = self.vel_index(condition)
        ini_index = self.ini_index(condition)
        x = to_cartesian(self.inner_radius, self.initial_angles[ini_index], self.z_location)
        vs = [to_cartesian(self.diff_radius, self.initial_angles[ini_index] + v_theta) for v_theta in self.diff_angles]
        indices = range(self.nvels)
        while True:
            np.random.shuffle(indices)
            if indices[RYG.index(self.color)] == vel_index:
                break
        return [x] + [x + vs[i] for i in indices]
    @staticmethod
    def xml(is_3d, robot_type):
        filename = {
            RobotType.THREE_LINK : './mjc_models/3link_gripper_push',
            RobotType.THREE_LINK_SHORT_JOINT : './mjc_models/3link_gripper_push_shortjoint',
            RobotType.FOUR_LINK : './mjc_models/4link_gripper_push',
            RobotType.FOUR_SIX : './mjc_models/4link_6joint_push',
            RobotType.FOUR_SEVEN : './mjc_models/4link_7joint_push',
            RobotType.FIVE_LINK : './mjc_models/5link_gripper_push',
            RobotType.THREE_DF_BLOCK : './mjc_models/3df_push',
            RobotType.PR2 : './mjc_models/pr2/pr2_arm_blockpush',
            RobotType.PR2_WIDER : './mjc_models/pr2/pr2_arm_wider_tool_blockpush',
            RobotType.PEGGY : './mjc_models/peggy_arm3d_blockpush',
            RobotType.BAXTER : './mjc_models/baxter/baxter_push',
        }[robot_type]
        if robot_type.is_arm() and is_3d:
            filename += "_3d"
        return filename + ".xml"
    def task_specific_cost(self, offset_generator, train_conditions):
        return [[{
            'type': CostFK,
            'target_end_effector': np.concatenate([np.array([0,0,0]),
                                                   offset_generator(i)[1 + RYG.index(self.color)],
                                                   np.zeros(9)]),
            'wp': np.array([0] * 3 + [1] * 3 + [0] * 9),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }, {
            'type': CostFKBlock,
            'wp': np.array([0] * 3 + [1] * 3 + [0] * 9),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }] for i in train_conditions]
    @staticmethod
    def modify_initial_state(state, _):
        return state

def to_cartesian(r, theta):
    return np.array([np.cos(theta), 0, np.sin(theta)]) * r

class BlockVelocityPush(BlockPush):
    COLOR_ORDER = "red", "green", "yellow"
    camera_pos = [0, 15., 0., 0.3, 0., 0.3]
    cost_weights = [1, 2, 40]
    def __init__(self, color, initial_angles, diff_angles, inner_radius, diff_radius):
        BlockPush.__init__(self, color, initial_angles, diff_angles, inner_radius, diff_radius * 5, z_location=0)
        self.velocities = [to_cartesian(self.diff_radius * 5, th0 + th) for th0 in initial_angles for th in diff_angles]
    @staticmethod
    def xml(is_3d, robot_type):
        path = BlockPush.xml(is_3d, robot_type)
        assert "push" in path
        return path.replace("push", "push_vel")
    def task_specific_cost(self, offset_generator, train_conditions):
        return [[{
            'type': CostFKBlock,
            'wp': np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }, {
            'type': CostState,
            'data_types' : {
                END_EFFECTOR_POINT_VELOCITIES: {
                    'wp': np.concatenate([np.zeros(3), np.ones(3), np.zeros(9)]),
                    'target_state': np.concatenate([np.zeros(3), self.velocities[i], np.zeros(9)]),
                },
            },
        }] for i in train_conditions]

class BlockCatch(object):
    def __init__(self, start_positions, velocities):
        self.start_positions = start_positions
        self.velocities = velocities
        assert len(start_positions) == len(velocities)
    additional_joints = 3
    number_end_effectors = 2
    cost_weights = [2, 4, 2]
    camera_pos = CAMERA_POS
    @staticmethod
    def body_indices(robot_type):
        start = robot_type.bodies_before_color_blocks()
        return [start + 1]
    def nconditions(self, n_offs, n_verts, n_blocks):
        del n_offs, n_verts
        return len(self.start_positions)
    def offset_generator(self, offsets, vert_offs, block_locs, condition):
        condition = condition % len(self.start_positions)
        return [self.start_positions[condition]]
    @staticmethod
    def xml(is_3d, robot_type):
        return BlockPush.xml(is_3d, robot_type).replace("push", "catch")
    @classmethod
    def task_specific_cost(cls, offset_generator, train_conditions):
        return [[{
            'type': CostFKBlock,
            'wp': np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5,
        }, {
            'type': CostState,
            'data_types' : {
                END_EFFECTOR_POINTS: {
                    'wp': np.array([0, 1, 0] * 2),
                    'target_state': np.array([0, 10, 0] * 2)
                },
            },
        }] for i in train_conditions]
    def modify_initial_state(self, state, condition):
        state[-3:] = self.velocities[condition % len(self.start_positions)]
        return state

class CleaningPerObject(object):
    camera_pos = CAMERA_POS
    def __init__(self, num_objects, file_tail, smoothing, goal_z=0):
        self.file_tail = file_tail
        self.num_objects = num_objects
        self.smoothing = smoothing
        self.goal_z = goal_z
        self.additional_joints = (self.num_objects + 1) * 2
        self.number_end_effectors = self.num_objects + 3
    cost_weights = [1, 1]
    x_offs = np.linspace(-0.15, 0.15, 4)
    y_offs = np.linspace(-0.15, 0.15, 4)
    def body_indices(self, robot_type):
        start = robot_type.bodies_before_color_blocks()
        return range(start + 1, start + self.num_objects + 2)
    @classmethod
    def nconditions(cls, n_offs, n_verts, n_blocks):
        return len(cls.x_offs) * len(cls.y_offs)
    def offset_generator(self, offsets, vert_offs, block_locs, condition):
        xpos = self.x_offs[condition % len(self.x_offs)] - 1.8
        ypos = self.y_offs[condition // len(self.x_offs)] - 1.5
        ball_location = 0.5
        goal_loc = [xpos, self.goal_z, ypos]
        obj_center = np.array([xpos, -0.25, ypos * ball_location])
        if self.num_objects == 1:
            items = [obj_center]
        else:
            items = [obj_center + [np.cos(x) / 4, 0, np.sin(x) / 4] for x in np.linspace(0, np.pi, self.num_objects)]
        return items + [goal_loc]
    def xml(self, is_3d, robot_type):
        filename = {
            RobotType.THREE_LINK : './mjc_models/3link_cleaning_task%s',
            RobotType.THREE_LINK_SHORT_JOINT : './mjc_models/3link_cleaning_task%s_shortjoint',
            RobotType.FOUR_LINK : './mjc_models/4link_cleaning_task%s',
        }[robot_type] % self.file_tail
        if robot_type.is_arm() and is_3d:
            filename += "_3d"
        return filename + ".xml"
    def task_specific_cost(self, offset_generator, train_conditions):
        cost_components = []
        for i in train_conditions:
            target = list(np.array(offset_generator(i)[-1] * 2) + [-0.5, 0, 0, 0.5, 0, 0]) if self.smoothing else [0] * 6
            for _ in range(self.number_end_effectors - 3):
                target += offset_generator(i)[-1]
            target += [0, 0, 0]
            current = {
                "type" : CostFK,
                "target_end_effector" : np.array(target),
                "wp" : np.array([1 if self.smoothing else 0] * 6 + [1] * (3 * self.number_end_effectors - 9) + [0] * 3),
                "l1" : 0.1,
                "l2" : 10.0,
                "alpha" : 1e-5
            }
            cost_components.append([current])
        return cost_components
    @staticmethod
    def modify_initial_state(state, _):
        return state

Cleaning = lambda smoothing, **kwargs: CleaningPerObject(5, "", smoothing, **kwargs)
CleaningSingleObject = lambda smoothing, **kwargs: CleaningPerObject(1, "_single_object", smoothing, **kwargs)

class ColorReach(object):
    cost_weights = [1, 1]
    additional_joints = 0
    camera_pos = CAMERA_POS
    def __init__(self, color, number_bodies=4):
        self.color = color
        self.number_bodies = number_bodies
    @property
    def number_end_effectors(self):
        return self.number_bodies + 1
    def body_indices(self, robot_type):
        start = robot_type.bodies_before_color_blocks()
        return range(start + 1, start + 1 + self.number_bodies)
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
        return [np.array(offsets[i]) + [0, vertical[i], 0] for i in indices]
    def xml(self, is_3d, robot_type):
        filename = {
            RobotType.THREE_LINK_SHORT_JOINT : './mjc_models/arm_3link_reach_colors_shortjoint',
            RobotType.THREE_LINK : './mjc_models/arm_3link_reach_colors',
            RobotType.FOUR_LINK : './mjc_models/arm_4link_reach_colors',
            RobotType.FIVE_LINK : './mjc_models/arm_5link_reach_colors',
            RobotType.PEGGY : './mjc_models/peggy_arm3d_reach_colors',
            RobotType.KINOVA : './mjc_models/kinova/jaco',
            RobotType.BAXTER : './mjc_models/baxter/baxter',
            RobotType.BAXTER_CYAN : './mjc_models/baxter/baxter_cyan',
            RobotType.PR2 : './mjc_models/pr2/pr2_arm',
            RobotType.PR2_MAGENTA : './mjc_models/pr2/pr2_arm_magenta'
        }[robot_type]
        if self.number_bodies != 4:
            filename = "{filename}_bodies_{number_bodies}".format(filename=filename, number_bodies=self.number_bodies)
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
    @staticmethod
    def modify_initial_state(state, _):
        return state

class LegoReach(ColorReach):
    camera_pos = [0, 5., 0., -3, 0., 0]
    @staticmethod
    def xml(is_3d, robot_type):
        xml_file = ColorReach("red").xml(is_3d, robot_type)
        return xml_file.replace("reach_colors", "reach_lego")

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

def reacher_by_color_and_type(robot_number, num_robots, is_3d, offsets, vert_offs, lego_offsets, blockpush_locations, (robot_type, is_real), enable_images, task_type, torque_costs, pass_environment_effectors_to_robot=False, number_samples=None, IMAGE_WIDTH=80, IMAGE_HEIGHT=64, IMAGE_CHANNELS=3):
    if isinstance(task_type, LegoReach):
        offsets = lego_offsets
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
        'camera_pos': np.array(task_type.camera_pos),
        'offs_to_use': offset_generator,
        'modify_initial_state' : task_type.modify_initial_state,
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
        'costs': ([torque_cost_0[i]] if torque_costs else []) + task_type.task_specific_cost(offset_generator, agent_dict['agent']['train_conditions'])[i],
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
