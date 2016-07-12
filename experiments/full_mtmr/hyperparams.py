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

from gps.example_agents.reach_3link import reach_3link
from gps.example_agents.reach_4link import reach_4link
from gps.example_agents.push_3link import push_3link
from gps.example_agents.push_4link import push_4link
from gps.example_agents.peg_3link import peg_3link
from gps.example_agents.peg_4link import peg_4link
from gps.example_agents.peg_right_3link import peg_right_3link
from gps.example_agents.peg_right_4link import peg_right_4link
from gps.example_agents.push_out_3link import push_out_3link
from gps.example_agents.push_out_4link import push_out_4link

# from gps.example_agents.reach_avoid_3link import reach_avoid_3link
# from gps.example_agents.reach_avoid_4link import reach_avoid_4link

from gps.example_agents.reach_3link_shortjoint import reach_3link_shortjoint
from gps.example_agents.reach_4link_shortjoint import reach_4link_shortjoint
from gps.example_agents.push_3link_shortjoint import push_3link_shortjoint
from gps.example_agents.push_4link_shortjoint import push_4link_shortjoint
from gps.example_agents.peg_3link_shortjoint import peg_3link_shortjoint
from gps.example_agents.peg_4link_shortjoint import peg_4link_shortjoint
from gps.example_agents.peg_right_3link_shortjoint import peg_right_3link_shortjoint
from gps.example_agents.peg_right_4link_shortjoint import peg_right_4link_shortjoint
from gps.example_agents.push_out_3link_shortjoint import push_out_3link_shortjoint
from gps.example_agents.push_out_4link_shortjoint import push_out_4link_shortjoint

# from gps.example_agents.reach_avoid_3link import reach_avoid_3link
# from gps.example_agents.reach_avoid_4link import reach_avoid_4linkp

agent_funs =[ reach_3link, reach_4link, push_3link, push_4link, push_out_3link, push_out_4link, peg_3link, peg_4link, peg_right_3link, peg_right_4link,
              reach_3link_shortjoint, reach_4link_shortjoint, push_3link_shortjoint, push_4link_shortjoint, push_out_3link_shortjoint,
              push_out_4link_shortjoint, peg_3link_shortjoint, peg_4link_shortjoint, peg_right_3link_shortjoint]# peg_right_4link_shortjoint,]
task_values = [0,0,1,1,2,2,3,3,4,4,
               0,0,1,1,2,2,3,3,4,]#,4]
robot_values = [0,1,0,1,0,1,0,1,0,1,
                2,3,2,3,2,3,2,3,2]#,3]

agents = []
num_agents = len(agent_funs)
for i in range(num_agents):
    agents.append(agent_funs[i](i, num_agents))


BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/full_mtmr/'
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
        'network_model': multitask_multirobot_fc,
        'network_params': {
            'task_list': task_values,
            'robot_list': robot_values,
            'agent_params':[a['network_params'] for a in agents],
        },
        'iterations': 4000,
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
    'iterations': 25,
    'num_samples': 7,
    'verbose_trials': 1,
    'verbose_policy_trials': 3,
    'save_wts': True,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'inner_iterations': 2,
    'robot_iters': [range(25), range(0,25,2)],
    'to_log': [END_EFFECTOR_POINTS, JOINT_ANGLES, ACTION],
}

common['info'] = generate_experiment_info(config)
