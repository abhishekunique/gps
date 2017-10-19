""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import re
import sys
import copy
import argparse
import threading
import time
import pickle
import tensorflow as tf
import IPython
from datetime import datetime
import numpy as np
# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.proto.gps_pb2 import ACTION, RGB_IMAGE, END_EFFECTOR_POINTS,END_EFFECTOR_POINT_VELOCITIES
import IPython
###from queue import Queue
from threading import Thread
from multiprocessing import Pool
def parallel_traj_samples(info):
    """info is [cond, num_samples, agent, pol, samples_out]"""
    cond = info[0]
    for i in range(info[1]):
       info[4].append(info[2].sample(info[3], cond, verbose=True))

os.environ['GLOG_minloglevel'] = '2'

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)



class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end
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


    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        for robot_number in range(self.num_robots):
            itr_start = self._initialize(itr_load, robot_number=robot_number)

        #self.read_traj_distr('all-12-new.pkl') #'final-dump-new.pkl')
        self.read_traj_distr(self._hyperparams["traj_distr_dump"])
        print("starting")
        for itr in range(itr_start, self._hyperparams['iterations']):
            print(itr)
            traj_sample_lists = {}
            for robot_number in range(self.num_robots):
                for cond in self._train_idx[robot_number]:
                    for i in range(self._hyperparams['num_samples']):
                        self._take_sample(itr, cond, i, robot_number=robot_number)

                traj_sample_lists[robot_number] = [
                    self.agent[robot_number].get_samples(cond_1, -self._hyperparams['num_samples'])
                    for cond_1 in self._train_idx[robot_number]
                ]
            #self.dump_traj_sample_lists(traj_sample_lists)

            for robot_number in range(self.num_robots):
                self._take_iteration(itr, traj_sample_lists[robot_number], robot_number=robot_number)

            for robot_number in range(self.num_robots):
                pol_sample_lists = None #self._take_policy_samples(robot_number=robot_number)
                self._log_data(itr, traj_sample_lists[robot_number], pol_sample_lists, robot_number=robot_number)
                self.save_traj_distr()
        self._end()
    def read_traj_distr(self, traj_distr_dump):
        HAVE_TRAJ_DISTR = os.path.isfile(traj_distr_dump)
        print
        if HAVE_TRAJ_DISTR:
            print "Reading from", traj_distr_dump
            traj_distr = self.data_logger.unpickle(traj_distr_dump)
            for ag in range(self.num_robots):
                name =self.agent[ag]._hyperparams['filename'][0]
                if name in traj_distr:
                    print "found", name
                    for cond in  self._train_idx[ag]:
                        self.algorithm[ag].cur[cond].traj_distr = traj_distr[name][cond]
                else:
                    print name, "not in traj_distr"
        else:
            print("Getting traj distr")
            newtraj_distr = {}
            for ag in range(self.num_robots):
                name = self.agent[ag]._hyperparams['filename'][0]
                #print name
                newtraj_distr[name] = []
                for cond in  self._train_idx[ag]:
                    newtraj_distr[name].append(self.algorithm[ag].cur[cond].traj_distr)
            self.data_logger.pickle(traj_distr_dump, newtraj_distr)
    def save_traj_distr(self):
        traj_distr = {}
        for ag in range(self.num_robots):
            name = self.agent[ag]._hyperparams['filename'][0]
            print name
            traj_distr[name] = []
            for cond in  self._train_idx[ag]:
                traj_distr[name].append(self.algorithm[ag].cur[cond].traj_distr)
        self.data_logger.pickle(self._hyperparams["traj_distr_dump"], traj_distr)
    # def save_traj_sample(self, samples):
    #     traj_distr = {}
    #     for ag in range(self.num_robots):
    #         name = self.agent[ag]._hyperparams['filename'][0]
    #         print name
    #         traj_distr[name] = []
    #         for cond in  self._train_idx[ag]:
    #             traj_distr[name].append(self.algorithm[ag].cur[cond].traj_distr)
    #     self.data_logger.pickle(self._hyperparams["traj_sample_dump"], traj_distr)

    def run_badmm(self, testing, load_old_weights, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        # self.collect_img_dataset(1)
        # import IPython
        time1 = time.clock()
        for robot_number in range(self.num_robots):
            itr_start = self._initialize(itr_load, robot_number=robot_number)

        self.policy_opt.validation_samples = self.data_logger.unpickle('4peg_val.pkl')
        #testing=True
        if testing:
            for i in range(len(self.policy_opt.policy)):
                _, size = [x.value for x in self.policy_opt.policy[i].obs_tensor.get_shape()]
                self.policy_opt.policy[i].scale = np.eye(size)
                self.policy_opt.policy[i].bias = np.zeros((size,))
                # FIXME READ IN THE VARIANCE FOR BLOCKPUSH, etc.
                # # self.policy_opt.var = [np.load('/home/coline/Downloads/pol_var_1.npy')[-2]]
                self.policy_opt.policy[i].x_idx = range(size)
            # for r in range(11):
            #     size = [36, 36, 36, 36, 36, 36, 36, 36, 38, 38, 38][r]
            #     self.policy_opt.policy[r].scale = np.eye(size)
            #     self.policy_opt.policy[r].bias = np.zeros((size,))
            #     # self.policy_opt.var = [np.load('/home/coline/Downloads/pol_var_1.npy')[-2]]
            #     self.policy_opt.policy[r].x_idx = range(size)
        # pool = Pool()
        nn_dump_path = self._hyperparams["nn_dump_path"]
        TRAJ_DISTR_COLOR_REACH = 'all-12-new.pkl' #
        #TRAJ_DISTR_COLOR_REACH = self._hyperparams["traj_distr_dump"]
        if not os.path.exists(nn_dump_path):
            os.makedirs(nn_dump_path)

        weights_pkl_offset = 0
        #import IPython; IPython.embed()
        nn_dumps = [int(re.sub("\D", "", x)) for x in os.listdir(nn_dump_path)]
        if testing or load_old_weights and nn_dumps:
            highest_nn_dump_iteration = max(nn_dumps)
            weights_pkl_offset = highest_nn_dump_iteration + 1
            print
            print "Loading weights", '{0}/weights_itr{1}.pkl'.format(nn_dump_path, highest_nn_dump_iteration)
            val_vars, pol_var = pickle.load(open('{0}/weights_itr{1}.pkl'.format(nn_dump_path, highest_nn_dump_iteration), 'rb'))
            self.policy_opt.var = pol_var#[pol_var[-2]]
            for k,v in self.policy_opt.av.items():
                if k in val_vars:
                    print(k)
                    assign_op = v.assign(val_vars[k])
                    self.policy_opt.sess.run(assign_op)
        if not testing:
            self.read_traj_distr(TRAJ_DISTR_COLOR_REACH)


        if False: #testing: # TODO use for blockpush, etc.
            for cond in range(4):
                samples = [self.agent[0].sample(self.algorithm[0].policy_opt.policy[0], cond,
                                                verbose=True, save=False) for j in range(5)]
                self.data_logger.pickle(self._data_files_dir+'nn_list_'+str(cond)+'.pkl', samples)
            sl0 = SampleList(self.data_logger.unpickle(self._data_files_dir + 'nn_list_0.pkl'))
            sl1 = SampleList(self.data_logger.unpickle(self._data_files_dir + 'nn_list_1.pkl'))
            sl2 = SampleList(self.data_logger.unpickle(self._data_files_dir + 'nn_list_2.pkl'))
            sl3 = SampleList(self.data_logger.unpickle(self._data_files_dir + 'nn_list_3.pkl'))
            for j in range(5):
                sl0[j].agent = self.agent[0]
                sl1[j].agent = self.agent[0]
                sl2[j].agent = self.agent[0]
                sl3[j].agent = self.agent[0]
            self.algorithm[0].reinitialize_net(0, sl0)
            self.algorithm[0].reinitialize_net(1, sl1)
            self.algorithm[0].reinitialize_net(2, sl2)
            self.algorithm[0].reinitialize_net(3, sl3)

        if testing:
            samps = {
                robot_number : self._take_policy_samples(robot_number=robot_number)
                for robot_number in range(self.num_robots)
            }
            self.dump_traj_sample_lists(samps)
            self._end()
            return
        self.check_itr = 8
        import IPython
        for itr in range(itr_start, self._hyperparams['iterations']):
            time2 = time.clock()
            traj_sample_lists = {}
            thread_samples_sampling = []
            print "itr", itr
            for robot_number in range(self.num_robots):
                print "sampling robot", robot_number, datetime.time(datetime.now())
                for cond in self._train_idx[robot_number]:
                    for i in range(self._hyperparams['num_samples']):
                        self._take_sample(itr, cond, i, robot_number=robot_number)

                traj_sample_lists[robot_number] = [
                    self.agent[robot_number].get_samples(cond_1, -self._hyperparams['num_samples'])
                    for cond_1 in self._train_idx[robot_number]
                ]
            self.dump_traj_sample_lists(traj_sample_lists)
            time3 = time.clock()
            # if self.agent[robot_number].nan_flag:
            IPython.embed()

            for robot_number in range(self.num_robots):
                # self.policy_opt.prepare_solver(itr_robot_status, self.)
                print "iter", itr,"start for rn", robot_number, datetime.time(datetime.now())
                self._take_iteration_start(itr, traj_sample_lists[robot_number], robot_number=robot_number)
            if self._hyperparams['view_trajectories']:
                for robot_number in range(self.num_robots):
                    print("update traj")
                    self.algorithm[robot_number]._update_trajectories()
                    self.algorithm[robot_number]._advance_iteration_variables()
                    if itr == 15:
                        raw_input("Press enter to continue: ")
                for robot_number in range(self.num_robots):
                    self._take_sample(itr, 0, i, robot_number=robot_number, verbose=True),mm
                    self._take_sample(itr, 3, i, robot_number=robot_number, verbose=True)
                continue
            time4 = time.clock()
            self._take_iteration_shared()
            time5 = time.clock()
            pol_samples = []
            for robot_number in range(self.num_robots):
                print "pol samples", robot_number, datetime.time(datetime.now())
                pol_sample_lists = self._take_policy_samples(robot_number=robot_number)
                pol_samples.append(pol_sample_lists)
                # if self.agent[robot_number].nan_flag:
                #     IPython.embed()
                self._log_data(itr, traj_sample_lists[robot_number], pol_sample_lists, robot_number=robot_number)
            time6 = time.clock()
            # if self.save_shared:
            #     self.policy_opt.save_shared_wts()
            # if self.save_wts:
            #     self.policy_opt.save_all_wts(itr)
            vars = {}
            for k,v in self.policy_opt.av.iteritems():
                vars[k] = self.policy_opt.sess.run(v)
            data_dump =[vars, self.policy_opt.var]
            with open('{0}/weights_itr{1}.pkl'.format(nn_dump_path, itr + weights_pkl_offset),'wb') as f:
                pickle.dump(data_dump, f)
            #self.save_traj_distr()
            if itr %20 ==0:# and itr >1:
                import IPython; IPython.embed()
        self._end()

    def dump_traj_sample_lists(self, traj_sample_lists):
        success_dict = {}
        for robot_number in traj_sample_lists:
            task_type, (robot_type, _) = self._hyperparams["agent_types"][robot_number]
            success_dict[task_type, robot_type] = []
            for condition, new_sample in enumerate(traj_sample_lists[robot_number]):
                successes = []
                for obs in new_sample.get_obs():
                    without_joints = robot_type.remove_joints_from_front(obs, task_type)
                    end_effector_points = without_joints[:,:task_type.number_end_effectors * 3]
                    end_effector_point_vels = without_joints[:,task_type.number_end_effectors * 3 : task_type.number_end_effectors * 6]
                    successes += [task_type.is_success(condition, end_effector_points, end_effector_point_vels)]
                success_dict[task_type, robot_type].append(np.mean(successes))
                if self._hyperparams["sim_traj_output"] is not None:
                    path = self._hyperparams["sim_traj_output"]
                    if not os.path.exists(path):
                        os.makedirs(path)
                    with open("{path}/{robot_number}_{condition}_{index}.pkl".format(
                                    path=path,
                                    robot_number=robot_number,
                                    condition=condition,
                                index=index), "wb") as f:
                        pickle.dump([new_sample.get_obs(), new_sample.get_U()], f)
        with open(self._hyperparams["successes_dump"], "wb") as f:
            pickle.dump(success_dict, f)
        if self._hyperparams["done_after_success_measurement"]:
            sys.exit(0)
        return success_dict

    def collect_samples(self, itr, traj_sample_lists, robot_number):
        for cond in self._train_id [robot_number]:
            for i in range(self._hyperparams['num_samples']):
                self._take_sample(itr, cond, i, robot_number=robot_number)

        traj_sample_lists[robot_number] = [
            self.agent[robot_number].get_samples(cond_1, -self._hyperparams['num_samples'])
            for cond_1 in self._train_idx
        ]
        return traj_sample_lists

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
            print "inner iter", inner_itr
            #TODO: Could start from init controller.
            obs_full = [None]*self.num_robots
            tgt_mu_full = [None]*self.num_robots
            tgt_prc_full = [None]*self.num_robots
            tgt_wt_full = [None]*self.num_robots
            itr_full = [None]*self.num_robots
            next_ee_full = [None]*self.num_robots
            for robot_number in range(self.num_robots):
                ee_data = np.zeros((0, 100, 6))
                if self.algorithm[robot_number].iteration_count > 0 or inner_itr > 0:
                    print "update pol lists", robot_number
                    obs, tgt_mu, tgt_prc, tgt_wt = self.algorithm[robot_number]._update_policy_lists(self.algorithm[robot_number].iteration_count, inner_itr)
                    obs_full[robot_number] = obs
                    tgt_mu_full[robot_number] = tgt_mu
                    tgt_prc_full[robot_number] = tgt_prc
                    tgt_wt_full[robot_number] = tgt_wt
                    itr_full[robot_number] = self.algorithm[robot_number].iteration_count
                    for m in range(self.algorithm[robot_number].M):
                        samples = self.algorithm[robot_number].cur[m].sample_list
                        ee = samples.get(END_EFFECTOR_POINTS)
                        ee_vel = samples.get(END_EFFECTOR_POINT_VELOCITIES)
                        curr_ee = ee[:, :, :3]
                        next_ee = np.concatenate((ee[:,1:,:3], ee[:, -1:, :3]), axis=1)
                        curr_ee_vel = ee_vel[:, :, :3]
                        next_ee_vel = np.concatenate((ee_vel[:,1:,:3], ee_vel[:, -1:, :3]), axis=1)
                        #next_ee = ee[:,:, 3:] # the target ee
                        ee_delta = next_ee - curr_ee
                        ee_vel_delta = next_ee_vel - curr_ee_vel
                        ee_state= np.concatenate((ee_delta, ee_vel_delta), axis=2)
                        ee_data= np.concatenate((ee_data, ee_state))
                    next_ee_full[robot_number] = ee_data
            #May want to make this shared across robots
            if self.algorithm[0].iteration_count > 0 or inner_itr > 0:
                print "policy opt update"
                self.policy_opt.update_ee(obs_full, tgt_mu_full, tgt_prc_full, tgt_wt_full,
                                          next_ee_full, itr_full, inner_itr)
            for robot_number in range(self.num_robots):
                print "update pol fit", robot_number
                for m in self._train_idx[robot_number]:
                    self.algorithm[robot_number]._update_policy_fit(m)  # Update policy priors.
            for robot_number in range(self.num_robots):
                print "dual", robot_number, datetime.time(datetime.now())
                if self.algorithm[robot_number].iteration_count > 0 or inner_itr > 0:
                    step = (inner_itr == self._hyperparams['inner_iterations'] - 1)
                    # Update dual variables.
                    for m in self._train_idx[robot_number]:
                        self.algorithm[robot_number]._policy_dual_step(m, step=step)
            # for robot_number in range(self.num_robots):
            #     print "update traj", robot_number
            #     self.algorithm[robot_number]._update_trajectories()

        for robot_number in range(self.num_robots):
            # self.algorithm[robot_number]._advance_iteration_variables()
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
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                if self.algorithm.cur[0].pol_info:
                    pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                        ('pol_sample_itr_%02d.pkl' % itr_load))
                else:
                    pol_sample_lists = None
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1


    def _take_sample(self, itr, cond, i, robot_number=0, verbose=False):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm[robot_number]._hyperparams['sample_on_policy'] and self.algorithm[robot_number].iteration_count > 0:
            pol = self.algorithm[robot_number].policy_opt.policy[robot_number]
        else:
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
                verbose=(i < self._hyperparams['verbose_trials']) or verbose,
                index=i
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


    def _take_policy_samples(self, robot_number=0):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        N = self._hyperparams['policy_trials']
        if self.gui:
            self.gui[robot_number].set_status_text('Taking policy samples.')
        pol_samples = [[None for _ in range(N)] for _ in range(self._conditions[robot_number])]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        for cond in range(self._conditions[robot_number]):
            for i in range(N):
                pol_samples[cond][i] = self.agent[robot_number].sample(
                    self.algorithm[robot_number].policy_opt.policy[robot_number], cond,
                    verbose=i < verbose, save=False)

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
        # if 'no_sample_logging' in self._hyperparams['common']:
        #     return
        # self.data_logger.pickle(
        #     self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
        #     copy.copy(self.algorithm)
        # )
        # self.data_logger.pickle(
        #     self._data_files_dir + ('traj_sample_itr_%02d_rn_%02d.pkl' % (itr,robot_number)),
        #     copy.copy(traj_sample_lists)
        # )
        # for key in self.traj_data_logs[robot_number].keys():
        #     self.traj_data_logs[robot_number][key].append([samplelist.get(key) for samplelist in traj_sample_lists])
        # self.data_logger.pickle(
        #     self._data_files_dir + ('traj_samples_combined_rn_%02d.pkl'% (robot_number)),
        #     copy.copy(self.traj_data_logs[robot_number]))
        # if pol_sample_lists:
        #     self.data_logger.pickle(
        #         self._data_files_dir + ('pol_sample_itr_%02d_rn_%02d.pkl' % (itr, robot_number)),
        #         copy.copy(pol_sample_lists)
        #     )
        #     for key in self.pol_data_logs[robot_number].keys():
        #         self.pol_data_logs[robot_number][key].append([samplelist.get(key) for samplelist in pol_sample_lists])
        #     self.data_logger.pickle(
        #         self._data_files_dir + ('pol_samples_combined_rn_%02d.pkl'% (robot_number)),
        #         copy.copy(self.pol_data_logs[robot_number]))

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            for robot_number in range(self.num_robots):
                self.gui[robot_number].set_status_text('Training complete.')
                self.gui[robot_number].end_mode()
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end:
                # Quit automatically (for running sequential expts)
                os._exit(1)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-c', '--config', type=str,
                        help='ignored')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-m', '--multithread', action='store_true',
                        help='Perform the badmm algorithm in parallel')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

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

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        gps = GPSMain(hyperparams.config, args.quit)
        if hyperparams.config['gui_on']:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                run_gps = threading.Thread(
                    target=lambda: gps.run(itr_load=resume_training_itr)
                )
            else:
                if args.multithread:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_badmm_parallel(testing=hyperparams.config['is_testing'], load_old_weights=hyperparams.config['load_old_weights'], itr_load=resume_training_itr)
                    )
                else:
                    run_gps = threading.Thread(
                        target=lambda: gps.run_badmm(testing=hyperparams.config['is_testing'], load_old_weights=hyperparams.config['load_old_weights'], itr_load=resume_training_itr)
                    )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            if hyperparams.config['algorithm'][0]['type'] == AlgorithmTrajOpt:
                gps.run(itr_load=resume_training_itr)
            else:
                gps.run_badmm(testing=hyperparams.config['is_testing'], load_old_weights=hyperparams.config['load_old_weights'], itr_load=resume_training_itr)

if __name__ == "__main__":
    main()
