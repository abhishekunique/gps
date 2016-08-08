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
import tensorflow as tf
# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.proto.gps_pb2 import ACTION, RGB_IMAGE
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        self._hyperparams = config
        self._conditions = [] 

        self._train_idx = []
        self._test_idx = []
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


    def run(self, itr_load=None, rf=False):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        for robot_number in range(self.num_robots):
            itr_start = self._initialize(itr_load, robot_number=robot_number)

        for itr in range(itr_start, self._hyperparams['iterations']):
            traj_sample_lists = {}
            feature_lists = []
            feature_lists_action = []
            for robot_number in range(self.num_robots):
                for cond in self._train_idx[robot_number]:
                    for i in range(self._hyperparams['num_samples']):
                        self._take_sample(itr, cond, i, robot_number=robot_number)

                traj_sample_lists[robot_number] = [
                    self.agent[robot_number].get_samples(cond_1, -self._hyperparams['num_samples'])
                    for cond_1 in self._train_idx[robot_number]
                ]
                if rf:
                    feature_lists.append(self.policy_opt.run_features_forward(self._extract_features(traj_sample_lists[robot_number], robot_number), robot_number))
                    # feature_lists_action.append(self.policy_opt.run_features_forward_action(self._extract_features_action(traj_sample_lists[robot_number], robot_number), robot_number))

            for robot_number in range(self.num_robots):
                self._take_iteration(itr, traj_sample_lists[robot_number], robot_number=robot_number)

            for robot_number in range(self.num_robots):
                pol_sample_lists = None #self._take_policy_samples(robot_number=robot_number)
                self._log_data(itr, traj_sample_lists[robot_number], pol_sample_lists, robot_number=robot_number)
                if rf:
                    np.save(self._data_files_dir + ('fps_%02d_rn_%02d.pkl' % (itr,robot_number)), copy.copy(np.asarray(feature_lists)))
                    # np.save(self._data_files_dir + ('actionfps_%02d_rn_%02d.pkl' % (itr,robot_number)), copy.copy(np.asarray(feature_lists_action)))
            if itr % 16 == 0 and itr > 0:
                import IPython
                IPython.embed()


        self._end()

    def run_badmm(self, itr_load=None, rf=False):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        for robot_number in range(self.num_robots):
            itr_start = self._initialize(itr_load, robot_number=robot_number)

        for itr in range(itr_start, self._hyperparams['iterations']):
            traj_sample_lists = {}
            feature_lists = []
            for robot_number in range(self.num_robots):
                for cond in self._train_idx[robot_number]:
                    for i in range(self._hyperparams['num_samples']):
                        self._take_sample(itr, cond, i, robot_number=robot_number)
                        
                traj_sample_lists[robot_number] = [
                    self.agent[robot_number].get_samples(cond_1, -self._hyperparams['num_samples'])
                    for cond_1 in self._train_idx[robot_number]
                ]
                if rf:
                    feature_lists.append(self.policy_opt.run_features_forward(self._extract_features(traj_sample_lists[robot_number], robot_number), robot_number))

            for robot_number in range(self.num_robots):
                self._take_iteration_start(itr, traj_sample_lists[robot_number], robot_number=robot_number)

            self._take_iteration_shared()

            for robot_number in range(self.num_robots):
                pol_sample_lists = self._take_policy_samples(robot_number=robot_number)
                self._log_data(itr, traj_sample_lists[robot_number], pol_sample_lists, robot_number=robot_number)
                if rf:
                    np.save(self._data_files_dir + ('fps_%02d_rn_%02d.pkl' % (itr,robot_number)), copy.copy(np.asarray(feature_lists)))
            if itr % 4 == 0 and itr > 0:
                import IPython
                IPython.embed()

        self._end()

    def _extract_features(self, pol_sample_lists, robot_number):
        dU, dO, T = self.algorithm[robot_number].dU, self.algorithm[robot_number].dO, self.algorithm[robot_number].T
        obs_data = np.zeros((len(pol_sample_lists), T, dO))
        for j, slist in enumerate(pol_sample_lists):
            for s in slist._samples:
                obs_data[j] += s.get_obs()
            obs_data[j] = obs_data[j]/float(len(slist._samples))
        # obs_data = ([obs[:, 0:3], obs[:, 4:7], obs[:, 8:11], obs[:, 17:20]], axis = 1)
        # obs_data = np.concatenate([obs_data[:, :, 0:3], obs_data[:, :, 3:6], obs_data[:, :, 6:9],  obs_data[:, :, 12:15]], axis=2) 
        obs_data = obs_data[:, :, self._hyperparams['r0_index_list']]
        return obs_data

    def _extract_features_action(self, pol_sample_lists, robot_number):
        dU, dO, T = self.algorithm[robot_number].dU, self.algorithm[robot_number].dO, self.algorithm[robot_number].T
        act_data = np.zeros((len(pol_sample_lists), T, dU))
        for j, slist in enumerate(pol_sample_lists):
            for s in slist._samples:
                act_data[j] += s.get_U()
            act_data[j] = act_data[j]/float(len(slist._samples))
        return act_data

    # def run_subspace_learning(self, itr_load=None):
    #     """
    #     Run training by iteratively sampling and taking an iteration.
    #     Args:
    #         itr_load: If specified, loads algorithm state from that
    #             iteration, and resumes training at the next iteration.
    #     Returns: None
    #     """
    #     # self.collect_img_dataset(1)
    #     obs_full = [None]*self.num_robots
    #     for robot_number in range(self.num_robots):
    #         if robot_number == 0:
    #             obs_sample = self.data_logger.unpickle(self._hyperparams['robot0_file'])
    #             for slist in obs_sample:
    #                 for s in slist._samples:
    #                     s.agent = self.agent[0]

    #         else:
    #             obs_sample = self.data_logger.unpickle(self._hyperparams['robot1_file'])
    #             for slist in obs_sample:
    #                 for s in slist._samples:
    #                     s.agent = self.agent[1]
        
    #         dU, dO, T = self.algorithm[robot_number].dU, self.algorithm[robot_number].dO, self.algorithm[robot_number].T
    #         dO = self.algorithm[robot_number].dX
    #         obs_data = np.zeros((0, T, dO))
    #         for samples in obs_sample:
    #             obs_data = np.concatenate((obs_data, samples.get_X()))

    #         if robot_number == 0:
    #             obs_data = obs_data[:, :, self._hyperparams['r0_index_list']]
    #             # obs_data = np.concatenate([obs_data[:, :, 0:3], obs_data[:, :, 3:6], obs_data[:, :, 6:9],  obs_data[:, :, 12:15]], axis=2) 
    #             # obs_data = np.concatenate([obs_data[:, :, 0:3], obs_data[:, :, 4:7], obs_data[:, :, 8:11],  obs_data[:, :, 17:20]], axis=2) 

    #         else:
    #             obs_data = obs_data[:, :, self._hyperparams['r1_index_list']]
    #             # obs_data = np.concatenate([obs_data[:, :, 0:4], obs_data[:, :, 4:8], obs_data[:, :, 8:11], obs_data[:, :, 14:17]], axis=2) 
    #             # obs_data = np.concatenate([obs_data[:, :, 0:4], obs_data[:, :, 5:9], obs_data[:, :, 10:13], obs_data[:, :, 19:22]], axis=2) 
    #         obs_full[robot_number] = obs_data


    #     # import matplotlib.pyplot as plt
    #     # from mpl_toolkits.mplot3d import Axes3D

    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # for c in range(3,4):
    #     #     for s in range(10):
    #     #         # ax.scatter(obs_full[0][c*10 + s][:,8],  obs_full[0][0][:,9], obs_full[0][0][:,10])
    #     #         # ax.scatter(obs_full[1][c*10 + s][:,10], obs_full[1][0][:,11], obs_full[1][0][:,12], marker='x')

    #     #         ax.scatter(obs_full[0][c*10 + s][:,11],  obs_full[0][0][:,12], obs_full[0][0][:,13])
    #     #         ax.scatter(obs_full[1][c*10 + s][:,13], obs_full[1][0][:,14], obs_full[1][0][:,15], marker='x')
    #     # plt.show()

    #     import IPython
    #     IPython.embed()
    #     self.policy_opt.train_invariant_autoencoder(obs_full)


    def run_subspace_learning(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        # self.collect_img_dataset(1)
        obs_full = [None]*self.num_robots
        act_full = [None]*self.num_robots
        for robot_number in range(self.num_robots):
            if robot_number == 0:
                obs_sample = self.data_logger.unpickle(self._hyperparams['robot0_file'])
                for slist in obs_sample:
                    for s in slist._samples:
                        s.agent = self.agent[0]

            else:
                obs_sample = self.data_logger.unpickle(self._hyperparams['robot1_file'])
                for slist in obs_sample:
                    for s in slist._samples:
                        s.agent = self.agent[1]
        
            dU, dO, T = self.algorithm[robot_number].dU, self.algorithm[robot_number].dO, self.algorithm[robot_number].T
            dO = self.algorithm[robot_number].dX
            obs_data = np.zeros((0, T, dO))
            act_data = np.zeros((0, T, dU))
            for samples in obs_sample:
                obs_data = np.concatenate((obs_data, samples.get_X()))
                act_data = np.concatenate((act_data, samples.get_U()))

            if robot_number == 0:
                obs_data = obs_data[:, :, self._hyperparams['r0_index_list']]
                # obs_data = np.concatenate([obs_data[:, :, 0:3], obs_data[:, :, 3:6], obs_data[:, :, 6:9],  obs_data[:, :, 12:15]], axis=2) 
                # obs_data = np.concatenate([obs_data[:, :, 0:3], obs_data[:, :, 4:7], obs_data[:, :, 8:11],  obs_data[:, :, 17:20]], axis=2) 

            else:
                obs_data = obs_data[:, :, self._hyperparams['r1_index_list']]
                # obs_data = np.concatenate([obs_data[:, :, 0:4], obs_data[:, :, 4:8], obs_data[:, :, 8:11], obs_data[:, :, 14:17]], axis=2) 
                # obs_data = np.concatenate([obs_data[:, :, 0:4], obs_data[:, :, 5:9], obs_data[:, :, 10:13], obs_data[:, :, 19:22]], axis=2) 
            obs_full[robot_number] = obs_data
            act_full[robot_number] = act_data

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for c in range(3,4):
        #     for s in range(10):
        #         # ax.scatter(obs_full[0][c*10 + s][:,8],  obs_full[0][0][:,9], obs_full[0][0][:,10])
        #         # ax.scatter(obs_full[1][c*10 + s][:,10], obs_full[1][0][:,11], obs_full[1][0][:,12], marker='x')

        #         ax.scatter(obs_full[0][c*10 + s][:,11],  obs_full[0][0][:,12], obs_full[0][0][:,13])
        #         ax.scatter(obs_full[1][c*10 + s][:,13], obs_full[1][0][:,14], obs_full[1][0][:,15], marker='x')
        # plt.show()
        # self.policy_opt.train_action_autoencoder(act_full)
        self.policy_opt.train_invariant_autoencoder(obs_full)
        

       
    def _take_iteration_start(self, itr, sample_lists, robot_number=0):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui[robot_number].set_status_text('Calculating.')
        self.algorithm[robot_number].iteration_start(sample_lists, itr)

    def _take_iteration_shared(self):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        # Run inner loop to compute new policies.
        for inner_itr in range(self._hyperparams['inner_iterations']):
            #TODO: Could start from init controller.
            obs_full = [None]*self.num_robots
            tgt_mu_full = [None]*self.num_robots
            tgt_prc_full = [None]*self.num_robots
            tgt_wt_full = [None]*self.num_robots
            itr_full = [None]*self.num_robots
            for robot_number in range(self.num_robots):
                if self.algorithm[robot_number].iteration_count > 0 or inner_itr > 0:
                    obs, tgt_mu, tgt_prc, tgt_wt = self.algorithm[robot_number]._update_policy_lists(self.algorithm[robot_number].iteration_count, inner_itr)
                    obs_full[robot_number] = obs
                    tgt_mu_full[robot_number] = tgt_mu
                    tgt_prc_full[robot_number] = tgt_prc
                    tgt_wt_full[robot_number] = tgt_wt
                    itr_full[robot_number] = self.algorithm[robot_number].iteration_count

            #May want to make this shared across robots
            if self.algorithm[0].iteration_count > 0 or inner_itr > 0:
                self.policy_opt.update(obs_full, tgt_mu_full, tgt_prc_full, tgt_wt_full, itr_full, inner_itr)
            for robot_number in range(self.num_robots):
                for m in self._train_idx[robot_number]:
                    self.algorithm[robot_number]._update_policy_fit(m)  # Update policy priors.
            for robot_number in range(self.num_robots):
                if self.algorithm[robot_number].iteration_count > 0 or inner_itr > 0:
                    step = (inner_itr == self._hyperparams['inner_iterations'] - 1)
                    # Update dual variables.
                    for m in self._train_idx[robot_number]:
                        self.algorithm[robot_number]._policy_dual_step(m, step=step)
            for robot_number in range(self.num_robots):
                self.algorithm[robot_number]._update_trajectories()
        for robot_number in range(self.num_robots):
            self.algorithm[robot_number]._advance_iteration_variables()
            if self.gui:
                self.gui[robot_number].stop_display_calculating()

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

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

    def _take_sample(self, itr, cond, i, robot_number=0):
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
                    'Sampling: iteration %d, condition %d, sample %d robot %d' %
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

    def _take_iteration(self, itr, sample_lists, robot_number=0):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui[robot_number].set_status_text('Calculating.')
            self.gui[robot_number].start_display_calculating()
        self.algorithm[robot_number].iteration(sample_lists, itr)
        if self.gui:
            self.gui[robot_number].stop_display_calculating()


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
        for cond in range(self._conditions[robot_number]):
            for i in range(N):
                pol_samples[cond][i] = self.agent[robot_number].sample(
                    self.algorithm[robot_number].policy_opt.policy[robot_number], cond,
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
        # self.data_logger.pickle(
        #     self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
        #     copy.copy(self.algorithm)
        # )
    
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d_rn_%02d.pkl' % (itr,robot_number)),
            copy.copy(traj_sample_lists)
        )

        for key in self.traj_data_logs[robot_number].keys():
            self.traj_data_logs[robot_number][key].append([samplelist.get(key) for samplelist in traj_sample_lists])

        self.data_logger.pickle(
            self._data_files_dir + ('traj_samples_combined_rn_%02d.pkl'% (robot_number)),
            copy.copy(self.traj_data_logs[robot_number]))

        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d_rn_%02d.pkl' % (itr, robot_number)),
                copy.copy(pol_sample_lists)
            )
            for key in self.pol_data_logs[robot_number].keys():
                self.pol_data_logs[robot_number][key].append([samplelist.get(key) for samplelist in pol_sample_lists])

            self.data_logger.pickle(
                self._data_files_dir + ('pol_samples_combined_rn_%02d.pkl'% (robot_number)),
                copy.copy(self.pol_data_logs[robot_number]))


        if self.gui:
            self.gui[robot_number].set_status_text('Logging data and updating GUI.')
            self.gui[robot_number].update(itr, self.algorithm[robot_number], self.agent[robot_number],
                traj_sample_lists, pol_sample_lists)

            self.gui[robot_number].save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )

        if 'no_sample_logging' in self._hyperparams['common']:
            return
 

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
    parser.add_argument('-f', '--traininvariant', action='store_true',
                        help='Train invariant subspace')
    parser.add_argument('-g', '--recordfeats', action='store_true',
                        help='Record features in feature space')
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

        random.seed(0)
        np.random.seed(0)

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

    elif args.traininvariant:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(1)
        np.random.seed(1)

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                run_gps = threading.Thread(
                    target=lambda: gps.run_subspace_learning(itr_load=resume_training_itr)
                )
            else:
                if args.multithread:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_subspace_learning(itr_load=resume_training_itr)
                    )
                else:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_subspace_learning(itr_load=resume_training_itr)
                    )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                gps.run_subspace_learning(itr_load=resume_training_itr)
            else:
                gps.run_subspace_learning(itr_load=resume_training_itr)


    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(13)
        np.random.seed(13)

        gps = GPSMain(hyperparams.config)
        if args.recordfeats:
            recordfeats = True
        else:
            recordfeats=False
        if hyperparams.config['gui_on']:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                run_gps = threading.Thread(
                    target=lambda: gps.run(itr_load=resume_training_itr, rf=recordfeats)
                )
            else:
                if args.multithread:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_badmm(itr_load=resume_training_itr, rf=recordfeats)
                    )
                else:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_badmm(itr_load=resume_training_itr, rf=recordfeats)
                    )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                gps.run(itr_load=resume_training_itr, rf=recordfeats)
            else:
                gps.run_badmm(itr_load=resume_training_itr, rf=recordfeats)

if __name__ == "__main__":
    main()
