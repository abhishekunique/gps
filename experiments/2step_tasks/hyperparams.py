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
from gps.algorithm.policy_opt.tf_model_example_multirobot import example_tf_network_multi, multitask_multirobot_fc_supervised
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

from gps.example_agents.door_3link import door_3link
from gps.example_agents.door_4link import door_4link
from gps.example_agents.door_5link import door_5link

from gps.example_agents.blockpushfree_3link import blockpushfree_3link
from gps.example_agents.blockpushfree_4link import blockpushfree_4link
from gps.example_agents.blockpushfree_5link import blockpushfree_5link

from gps.example_agents.lockkey_3link import lockkey_3link
from gps.example_agents.lockkey_4link import lockkey_4link
from gps.example_agents.lockkey_5link import lockkey_5link

from gps.example_agents.horizdrawer_3link import horizdrawer_3link
from gps.example_agents.horizdrawer_4link import horizdrawer_4link
from gps.example_agents.horizdrawer_5link import horizdrawer_5link
from gps.example_agents.vertdrawer_3link import vertdrawer_3link
from gps.example_agents.vertdrawer_4link import vertdrawer_4link
from gps.example_agents.vertdrawer_5link import vertdrawer_5link

from gps.example_agents.blockpushfree_2step_3link import blockpushfree_2step_3link
from gps.example_agents.blockpushfree_2step_4link import blockpushfree_2step_4link
from gps.example_agents.blockpushfree_2step_5link import blockpushfree_2step_5link



# agent_funs =[door_4link, door_5link, \
#             blockpushfree_3link, blockpushfree_4link, blockpushfree_5link,\
#             lockkey_3link, lockkey_4link, lockkey_5link]
# task_values = [0, 0, 1, 1, 1, 2, 2, 2]
# robot_values =[1, 2, 0, 1, 2, 0, 1, 2]

# agent_funs =[vertdrawer_3link]
# task_values = [0]
# robot_values =[0]

# agent_funs =[blockpushfree_2step_3link, blockpushfree_2step_4link, blockpushfree_2step_5link,
# vertdrawer_3link, vertdrawer_4link, vertdrawer_5link, horizdrawer_4link, horizdrawer_5link]
agent_funs = [horizdrawer_3link]
task_values = [2]#, 0, 0, 1, 1, 1, 2, 2]
robot_values =[0]#, 1, 2, 0, 1, 2, 1, 2]

agents = []
num_agents = len(agent_funs)
for i in range(num_agents):
    agents.append(agent_funs[i](i, num_agents))
# val_agent =  {'agent': reach_4link(num_agents),
#               'task': 0,
#               'robot': 1,}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/2step_tasks/'
INIT_POLICY_DIR = '/home/abhigupta/gps/'
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
    'num_robots':len(agents),
    'policy_opt': {
        'type': PolicyOptTf,
        'network_model': multitask_multirobot_fc_supervised,
        'network_params': {
            'task_list': task_values,
            'robot_list': robot_values,
            'agent_params':[a['network_params'] for a in agents],
        },
        #'val_agents': [1],
        'iterations': 15000,
        'fc_only_iterations': 5000,
        'checkpoint_prefix': EXP_DIR + 'data_files/policy',
        # 'restore_all_wts':'/home/abhigupta/gps/allweights_push_4link.npy'
    }
}


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = [a['agent'] for a in agents]
algorithm = [a['algorithm'] for a in agents]

config = {
    'iterations': 50,
    'num_samples': 10,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'save_wts': True,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'inner_iterations': 4,
    'robot_iters': [range(25), range(0,25,2)],
    'to_log': [END_EFFECTOR_POINTS, JOINT_ANGLES, ACTION],
    #'val_agents': [1],
}

common['info'] = generate_experiment_info(config)
