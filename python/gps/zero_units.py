""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import pickle
import tensorflow as tf
# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.proto.gps_pb2 import ACTION, RGB_IMAGE, END_EFFECTOR_POINTS,END_EFFECTOR_POINT_VELOCITIES
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
import IPython
###from queue import Queue
from threading import Thread
from multiprocessing import Pool
def parallel_traj_samples(info):
    """info is [cond, num_samples, agent, pol, samples_out]"""
    cond = info[0]
    for i in range(info[1]):
       info[4].append(info[2].sample(info[3], cond, verbose=True))


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        self._hyperparams = config
        self._conditions = [] #config['common']['conditions']

        self._train_idx = [] #config['common']['train_conditions']
        self._test_idx = []#config['common']['test_conditions']
        # else:
        #     self._train_idx = range(self._conditions)
        #     config['common']['train_conditions'] = config['common']['conditions']
        #     self._hyperparams=config
        #     self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']
        self.num_robots = config['common']['num_robots']
        self.agent = []
        self.gui = []
        for robot_number in range(self.num_robots):
            self.agent.append(config['agent'][robot_number]['type'](config['agent'][robot_number]))
            self._conditions.append(config['agent'][robot_number]['conditions'])
            if 'train_conditions' in config['agent'][robot_number]:
                self._train_idx.append(config['agent'][robot_number]['train_conditions'])
                self._test_idx.append(config['agent'][robot_number]['train_conditions'])
            else:
                self._train_idx.append(range(self._conditions[robot_number]))
                self._test_idx.append(range(self._conditions[robot_number]))
                config['agent'][robot_number]['train_conditions'] = self._train_idx[robot_number]
                self._hyperparams = config
            if config['gui_on']:
                self.gui.append(GPSTrainingGUI(config['common']))
            else:
                self.gui = None
        self.data_logger = DataLogger()

        self.pol_data_logs = [{key:[] for key in self._hyperparams['to_log']} for r in range(self.num_robots)]
        self.traj_data_logs = [{key:[] for key in self._hyperparams['to_log']} for r in range(self.num_robots)]

        self.algorithm = []
        for robot_number in range(self.num_robots):
            config['algorithm'][robot_number]['agent'] = self.agent[robot_number]
            self.algorithm.append(config['algorithm'][robot_number]['type'](config['algorithm'][robot_number]))
        if 'policy_opt' in self._hyperparams['common']:
            dU = [ag.dU for ag in self.agent]
            dO = [ag.dO for ag in self.agent]
            self.policy_opt =  self._hyperparams['common']['policy_opt']['type'](
                self._hyperparams['common']['policy_opt'], dO, dU
            )
            for robot_number in range(self.num_robots):
                self.algorithm[robot_number].policy_opt = self.policy_opt
                self.algorithm[robot_number].robot_number = robot_number
        self.save_shared = False
        if 'save_shared' in self._hyperparams['common']:
            self.save_shared = self._hyperparams['save_shared']
        else: self.save_shared = False
        if 'save_wts' in self._hyperparams:
            self.save_wts = self._hyperparams['save_wts']
        else: self.save_wts = False

    def run_badmm(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        # self.collect_img_dataset(1)
        import IPython
        time1 = time.clock()
        for robot_number in range(self.num_robots):
            itr_start = self._initialize(itr_load, robot_number=robot_number)

        self.policy_opt.validation_samples = self.data_logger.unpickle('4peg_val.pkl')


        size = 18
        self.policy_opt.policy[0].scale = np.eye(size)
        self.policy_opt.policy[0].bias = np.zeros((size,))
        # self.policy_opt.var = [np.load('/home/coline/Downloads/pol_var_1.npy')[-2]]
        self.policy_opt.policy[0].x_idx = range(size)



       # traj_distr = self.data_logger.unpickle('/home/coline/Downloads/traj_distr_newest.pkl')
        # # abh_traj_distr = self.data_logger.unpickle('abh_traj_distr_mtmr_moreiters.pkl')
        # for ag in range(self.num_robots):
        #     name = self.agent[ag]._hyperparams['filename'][0]
        #     # IPython.embed()
        #     # if 'reach' in name:
        #     for cond in  self._train_idx[ag]:
        #         print ag, cond
        #         self.algorithm[ag].cur[cond].traj_distr = traj_distr[name][cond]
        self.check_itr = 1
        task = 2
        robot=0
        robot_number=0
        task_out_size =12
        costs = [[] for n in range(task_out_size)]
        features = []

        import pickle
        val_vars, pol_var = pickle.load(open('/home/coline/abhishek_gps/gps/weights_reachtest_itr9.pkl', 'rb'))
        self.policy_opt.var = pol_var#[pol_var[-2]]
        val_vars['task_weights_tn_'+str(task)] = np.ones(val_vars['task_weights_tn_'+str(task)].shape)
        for k,v in self.policy_opt.av.items():
            if k in val_vars:
                assign_op = v.assign(val_vars[k])
                self.policy_opt.sess.run(assign_op)

        print "orig sampels"
        pol_sample_lists = self._take_policy_samples(N=4,robot_number=robot_number)
        print "done"
        cost = 0
        for cond in self._train_idx[0]:
            cs = self.eval_cost(pol_sample_lists[cond], cond, itr=0)
            cost += np.sum(cs)
        print "orig cost", cost
        orig_cost = cost
        for f1 in range(1):
            for f2 in range(task_out_size):
                w = self.policy_opt.av['task_weights_tn_'+str(task)]
                wval = self.policy_opt.sess.run(w)
                #print "wval =", wval
                if wval[f2] == 1:
                    wval_new = wval.copy()
                    wval_new[f2] = 0
                    assign_op = w.assign(wval_new)
                    self.policy_opt.sess.run(assign_op)
                    pol_sample_lists = self._take_policy_samples(N=4,robot_number=robot_number)
                    cost = 0
                    for cond in self._train_idx[0]:
                        cs = self.eval_cost(pol_sample_lists[cond], cond, itr=0)
                        cost += np.sum(cs)
                    costs[f1].append((cost, f2))
                    print f1, f2, cost
                    assign_op = w.assign(wval)
                    self.policy_opt.sess.run(assign_op)
            largest_cost, feature = max(costs[f1])
            features.append(feature)
            w = self.policy_opt.av['task_weights_tn_'+str(task)]
            wval = self.policy_opt.sess.run(w)
            wval[feature] = 0
            assign_op = w.assign(wval)
            self.policy_opt.sess.run(assign_op)
            print "REMOVED feature", feature
        IPython.embed()
        self._end()

    def eval_cost(self,sample_list, cond, itr):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.algorithm[0].T, self.algorithm[0].dX, self.algorithm[0].dU
        N = len(sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample =sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.algorithm[0].cost[cond].eval(sample, itr)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        # self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        # self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        # self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).
        return cs

    def _initialize(self, itr_load, robot_number=0):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui[robot_number].set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_i_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread
                
            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('pol_sample_itr_%02d.pkl' % itr_load))
                self.gui.update(itr_load, self.algorithm, self.agent,
                    traj_sample_lists, pol_sample_lists)
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

 
    def _take_policy_samples(self, N=None, robot_number=0):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        # if 'verbose_policy_trials' not in self._hyperparams:
        #     return None
        if not N:
            N = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui[robot_number].set_status_text('Taking policy samples.')
        pol_samples = [[None for _ in range(N)] for _ in range(self._conditions[robot_number])]
        for cond in self._train_idx[robot_number]:
            for i in range(N):
                pol_samples[cond][i] = self.agent[robot_number].sample(
                    self.algorithm[robot_number].policy_opt.policy[robot_number], cond,
                    verbose=(i==0), save=False)
        return [SampleList(samples) for samples in pol_samples]

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            for robot_number in range(self.num_robots):
                self.gui[robot_number].set_status_text('Training complete.')
                self.gui[robot_number].end_mode()


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM only)')
    parser.add_argument('-m', '--multithread', action='store_true',
                        help='Perform the badmm algorithm in parallel')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    exp_dir = 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    if args.targetsetup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    elif test_policy_N:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(40)
        np.random.seed(40)
        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(40)
        np.random.seed(40)

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                run_gps = threading.Thread(
                    target=lambda: gps.run(itr_load=resume_training_itr)
                )
            else:
                if args.multithread:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_badmm_parallel(itr_load=resume_training_itr)
                    )
                else:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_badmm(itr_load=resume_training_itr)
                    )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                gps.run(itr_load=resume_training_itr)
            else:
                gps.run_badmm(itr_load=resume_training_itr)

if __name__ == "__main__":
    main()
