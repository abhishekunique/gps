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
import multiprocessing
from pathos.multiprocessing import ProcessingPool
# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):

        self._hyperparams = config
        self._conditions = config['common']['conditions']
        # TODO: add default
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']
        self.num_robots = config['common']['num_robots']
        self.agent = [ag['type'](ag) for ag in config['agent']]
        self.data_logger = DataLogger()
        self.gui = [GPSTrainingGUI(config['common']) for r in range(self.num_robots)] if config['gui_on'] else None   
        for alg, ag in zip(config['algorithm'], self.agent):
            alg['agent'] = ag
        dU = [ag.dU for ag in self.agent]
        
        dO = [ag.dO for ag in self.agent]
        self.policy_opt =  self._hyperparams['common']['policy_opt']['type'](
           self._hyperparams['common']['policy_opt'], dO, dU
        )
        self.algorithm = []
        for robot_number, alg in enumerate(config['algorithm']):
            self.algorithm.append(alg['type'](alg, self.policy_opt, robot_number))

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        #CHANGED
        for robot_number in range(self.num_robots):
            #CHANGED
            itr_start = self._initialize(itr_load, robot_number=robot_number)

        for itr in range(itr_start, self._hyperparams['iterations']):
            #CHANGED            
            traj_sample_lists = [None]*self.num_robots
            thread_samples = []
            for robot_number in range(self.num_robots):
                thread_samples.append(threading.Thread(target=self.collect_samples, args=(itr, robot_number, traj_sample_lists)))
                thread_samples[robot_number].start()
            for robot_number in range(self.num_robots):
                thread_samples[robot_number].join()

            thread_samples = []
            for robot_number in range(self.num_robots):
                thread_samples.append(threading.Thread(target=self._take_iteration_start, args=(itr, traj_sample_lists[robot_number], robot_number)))
                thread_samples[robot_number].start()
            for robot_number in range(self.num_robots):
                thread_samples[robot_number].join()
            # for robot_number in range(self.num_robots):
            #     self._take_iteration_start(itr, traj_sample_lists[robot_number], robot_number)
            self._take_iteration_shared()

            thread_samples = []
            for robot_number in range(self.num_robots):
                thread_samples.append(threading.Thread(target=self.take_policy_samples_and_log, args=(itr, robot_number, traj_sample_lists[robot_number])))
                thread_samples[robot_number].start()
            for robot_number in range(self.num_robots):
                thread_samples[robot_number].join()



            import IPython
            IPython.embed()
        self._end()

    def collect_samples(self, itr, robot_number, traj_sample_lists):
        for cond in self._train_idx:
            for i in range(self._hyperparams['num_samples']):
                #CHANGED
                self._take_sample(itr, cond, i, robot_number=robot_number)
        #CHANGED
        traj_sample_lists[robot_number] = [
            self.agent[robot_number].get_samples(cond, -self._hyperparams['num_samples'])
            for cond in self._train_idx
        ]
        return traj_sample_lists

    def take_policy_samples_and_log(self, itr, robot_number, traj_sample_list):
        pol_sample_list = self._take_policy_samples(robot_number)
        self._log_data(itr, traj_sample_list, pol_sample_list, robot_number=robot_number)


    def _initialize(self, itr_load, robot_number):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            #CHANGED
            if self.gui[robot_number]:
                #CHANGED
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
                #CHANGED
                self.gui[robot_number].update(itr_load, self.algorithm[robot_number], self.agent[robot_number],
                    traj_sample_lists, pol_sample_lists)
                #CHANGED
                self.gui[robot_number].set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i, robot_number):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        pol = self.algorithm[robot_number].cur[cond].traj_distr
        if self.gui:
            self.gui[robot_number].set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui[robot_number].mode in ('wait', 'request', 'process'):
                    if self.gui[robot_number].mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui[robot_number].request == 'reset':
                        try:
                            self.agent[robot_number].reset(cond)
                        except NotImplementedError:
                            self.gui[robot_number].err_msg = 'Agent reset unimplemented.'
                    elif self.gui[robot_number].request == 'fail':
                        self.gui[robot_number].err_msg = 'Cannot fail before sampling.'
                    self.gui[robot_number].process_mode()  # Complete request.

                self.gui[robot_number].set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d robot_number %d' %
                    (itr, cond, i, robot_number)
                )
                self.agent[robot_number].sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui[robot_number].mode == 'request' and self.gui[robot_number].request == 'fail':
                    redo = True
                    self.gui[robot_number].process_mode()
                    self.agent[robot_number].delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent[robot_number].sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            )

    def _take_iteration_start(self, itr, sample_lists, robot_number):
        if self.gui:
            self.gui[robot_number].set_status_text('Calculating.')
        self.algorithm[robot_number].iteration_start(sample_lists)

    def _take_iteration_shared(self):
        for inner_itr in range(self._hyperparams['inner_iterations']):
            #TODO: Could start from init controller.
            obs_data_full = [None]*self.num_robots
            tgt_mu_full = [None]*self.num_robots
            tgt_prc_full = [None]*self.num_robots
            tgt_wt_full = [None]*self.num_robots
            #Can parallelize this
            for r_no in range(self.num_robots):
                if self.algorithm[r_no].iteration_count > 0 or inner_itr > 0:
                    # Update the policy.
                    
                    obs_data, tgt_mu, tgt_prc, tgt_wt = self.algorithm[r_no]._write_policy_samples(self.algorithm[r_no].iteration_count, inner_itr)
                    obs_data_full[r_no] = obs_data
                    tgt_mu_full[r_no] = tgt_mu
                    tgt_prc_full[r_no] = tgt_prc
                    tgt_wt_full[r_no] = tgt_wt
            full_iteration_count = [self.algorithm[r_no].iteration_count for r_no in range(self.num_robots)]
            #TODO: Make this better
            if self.algorithm[0].iteration_count > 0 or inner_itr > 0:
                self.policy_opt.update(obs_data_full, tgt_mu_full, tgt_prc_full, tgt_wt_full,
                   full_iteration_count, inner_itr)
           
            for r_no in range(self.num_robots):
                for m in range(self._conditions):
                    self.algorithm[r_no]._update_policy_fit(m)  # Update policy priors.
            
            for r_no in range(self.num_robots):
                if self.algorithm[r_no].iteration_count > 0 or inner_itr > 0:
                    step = (inner_itr == self._hyperparams['inner_iterations'] - 1)
                    # Update dual variables.
                    for m in range(self._conditions):
                        self.algorithm[r_no]._policy_dual_step(m, step=step)

            for r_no in range(self.num_robots):
                self.algorithm[r_no]._update_trajectories()
        for r_no in range(self.num_robots):   
            self.algorithm[r_no]._advance_iteration_variables()
     


    def _take_policy_samples(self, robot_number):
        """ Take samples from the policy to see how it's doing. """
        if 'verbose_policy_trials' not in self._hyperparams:
            return None
        if self.gui:
            self.gui[robot_number].set_status_text('Taking policy samples.')
        pol_samples = [[None for _ in range(self._hyperparams['verbose_policy_trials'])]
                       for _ in self._test_idx]
        for cond in range(len(self._test_idx)):
            for i in range(self._hyperparams['verbose_policy_trials']):
                pol_samples[cond][i] = self.agent[robot_number].sample(
                    self.algorithm.policy_opt.policy[robot_number], self._test_idx[cond],
                    verbose=True, save=False)
        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None, robot_number=0):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui:
            self.gui[robot_number].set_status_text('Logging data and updating GUI.')
            self.gui[robot_number].update(itr, self.algorithm[robot_number], self.agent[robot_number],
                traj_sample_lists, pol_sample_lists)
            self.gui[robot_number].save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm[robot_number])
        )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

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
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume

    exp_dir = 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.new:
        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)
        open(hyperparams_file, 'w')
        sys.exit("Experiment '%s' created.\nhyperparams file: '%s'" %
                 (exp_name, hyperparams_file))

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
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(0)
        np.random.seed(0)

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
