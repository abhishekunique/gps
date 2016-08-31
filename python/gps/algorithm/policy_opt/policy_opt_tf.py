""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver


LOGGER = logging.getLogger(__name__)


class PolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])
        self.num_robots = len(dU)
        self.checkpoint_file = self._hyperparams['checkpoint_prefix']
        self.batch_size = self._hyperparams['batch_size']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            self.gpu_device = self._hyperparams['gpu_id']
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_ops = []
        self.loss_scalars = []
        self.obs_tensors = []
        self.precision_tensors = []
        self.action_tensors = []
        self.solver = None
        self.var = []
        for i, dU_ind in enumerate(dU):
            self.act_ops.append(None)
            self.loss_scalars.append(None)
            self.obs_tensors.append(None)
            self.precision_tensors.append(None)
            self.action_tensors.append(None)
            self.var.append(self._hyperparams['init_var'] * np.ones(dU_ind))

        self.solver = None
        self.init_network()
        self.init_solver()
        self.sess = tf.Session()
        self.policy = []
        for dU_ind, ot, ap in zip(dU, self.obs_tensors, self.act_ops):
            self.policy.append(TfPolicy(dU_ind, ot, ap, np.zeros(dU_ind), self.sess, self.device_string, self.ls['keep_prob']))
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], []
        for robot_number in range(self.num_robots):
            self.x_idx.append([])
            self.img_idx.append([])
            i.append(0)

        for robot_number, robot_params in enumerate(self._hyperparams['network_params']['agent_params']):
            for sensor in robot_params['obs_include']:
                dim = robot_params['sensor_dims'][sensor]
                if sensor in robot_params['obs_image_data']:
                    self.img_idx[robot_number] = self.img_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                else:
                    self.x_idx[robot_number] = self.x_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                i[robot_number] += dim

        if not isinstance(self._hyperparams['ent_reg'], list):
            self.ent_reg = [self._hyperparams['ent_reg']]*self.num_robots
        else:
            self.ent_reg = self._hyperparams['ent_reg']

        # if 'obs_image_data' not in self._hyperparams['network_params']:
        #     self._hyperparams['network_params'].update({'obs_image_data': []})
        # for sensor in self._hyperparams['network_params']['obs_include']:
        #     dim = self._hyperparams['network_params']['sensor_dims'][sensor]
        #     if sensor in self._hyperparams['network_params']['obs_image_data']:
        #         self.img_idx = self.img_idx + list(range(i, i+dim))
        #     else:
        #         self.x_idx = self.x_idx + list(range(i, i+dim))
        #     i += dim
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        tf_maps, robot_vars, task_vars, av, ls = (
            tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
                             network_config=self._hyperparams['network_params']))
        self.obs_tensors = []
        self.action_tensors = []
        self.precision_tensors = []
        self.act_ops = []
        self.loss_scalars = []
        self.robot_vars = robot_vars
        self.task_vars = task_vars
        self.feature_points= []
        for tf_map in tf_maps:
            self.obs_tensors.append(tf_map.get_input_tensor())
            self.action_tensors.append(tf_map.get_target_output_tensor())
            self.precision_tensors.append(tf_map.get_precision_tensor())
            self.act_ops.append(tf_map.get_output_op())
            self.loss_scalars.append(tf_map.get_loss_op())
            self.feature_points.append(tf_map.feature_points)

        self.train_losses = []
        self.val_agents= self._hyperparams['val_agents']
        self.val_loss = 0#tf.add_n([self.loss_scalars[i] for i in self.val_agents])
        for i in range(len(self.loss_scalars)):
            if i  not in self.val_agents:
                self.train_losses.append(self.loss_scalars[i])
        self.combined_loss = tf.add_n(self.train_losses)
        self.av = av
        self.ls = ls
        # tf_map = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
        #                           network_config=self._hyperparams['network_params'])
        # self.obs_tensor = tf_map.get_input_tensor()
        # self.action_tensor = tf_map.get_target_output_tensor()
        # self.precision_tensor = tf_map.get_precision_tensor()
        # self.act_op = tf_map.get_output_op()
        # self.loss_scalar = tf_map.get_loss_op()

    def init_solver(self):
        """ Helper method to initialize the solver. """

        self.solver =TfSolver(loss_scalar=self.combined_loss,
                                    solver_name=self._hyperparams['solver_type'],
                                    base_lr=self._hyperparams['lr'],
                                    lr_policy=self._hyperparams['lr_policy'],
                                    momentum=self._hyperparams['momentum'],
                                    weight_decay=self._hyperparams['weight_decay'],
                                    # vars_to_opt = self.robot_vars)
                                    vars_to_opt= self.av.values())

    def update(self, obs_full, tgt_mu_full, tgt_prc_full, tgt_wt_full):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N_reshaped = []
        T_reshaped = []
        obs_reshaped = []
        tgt_mu_reshaped = []
        tgt_prc_reshaped = []
        tgt_wt_reshaped = []
        idx_reshaped = []
        batches_per_epoch_reshaped = []
        tgt_prc_orig_reshaped = []
        for robot_number in range(self.num_robots):
            obs = obs_full[robot_number]
            tgt_mu = tgt_mu_full[robot_number]
            tgt_prc = tgt_prc_full[robot_number]
            tgt_wt = tgt_wt_full[robot_number]
            N, T = obs.shape[:2]
            dU, dO = self._dU[robot_number], self._dO[robot_number]

            # TODO - Make sure all weights are nonzero?

            # Save original tgt_prc.
            tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

            # Renormalize weights.
            tgt_wt *= (float(N * T) / np.sum(tgt_wt))
            # Allow weights to be at most twice the robust median.
            mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
            for n in range(N):
                for t in range(T):
                    tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
            # Robust median should be around one.
            tgt_wt /= mn

            # Reshape inputs.
            obs = np.reshape(obs, (N*T, dO))
            tgt_mu = np.reshape(tgt_mu, (N*T, dU))
            tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
            tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

            # Fold weights into tgt_prc.
            tgt_prc = tgt_wt * tgt_prc

            # TODO: Find entries with very low weights?
            # Normalize obs, but only compute normalzation at the beginning.
            if self.policy[robot_number].scale is None or self.policy[robot_number].bias is None:
                self.policy[robot_number].x_idx = self.x_idx[robot_number]
                # 1e-3 to avoid infs if some state dimensions don't change in the
                # first batch of samples
                self.policy[robot_number].scale = np.diag(
                    1.0 / np.maximum(np.std(obs[:, self.x_idx[robot_number]], axis=0), 1e-3))
                self.policy[robot_number].bias = - np.mean(
                    obs[:, self.x_idx[robot_number]].dot(self.policy[robot_number].scale), axis=0)
            # if itr == 0 and inner_itr == 1:
            #     #TODO: may need to change this
            #     self.policy[robot_number].x_idx = self.x_idx[robot_number]
            #     self.policy[robot_number].scale = np.eye(np.diag(1.0 / (np.std(obs[:, self.x_idx[robot_number]], axis=0) + 1e-8)).shape[0])
            #     self.policy[robot_number].bias = np.zeros((-np.mean(obs[:, self.x_idx[robot_number]].dot(self.policy[robot_number].scale), axis=0)).shape)
            #     print("FIND")

            obs[:, self.x_idx[robot_number]] = obs[:, self.x_idx[robot_number]].dot(self.policy[robot_number].scale) + self.policy[robot_number].bias

            # Assuming that N*T >= self.batch_size.
            batches_per_epoch = np.floor(N*T / self.batch_size)
            idx = range(N*T)
            
            np.random.shuffle(idx)
            obs_reshaped.append(obs)
            tgt_mu_reshaped.append(tgt_mu)
            tgt_prc_reshaped.append(tgt_prc)
            tgt_wt_reshaped.append(tgt_wt)
            N_reshaped.append(N)
            T_reshaped.append(T)
            idx_reshaped.append(idx)
            batches_per_epoch_reshaped.append(batches_per_epoch)
            tgt_prc_orig_reshaped.append(tgt_prc_orig)

        average_loss = 0

        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            feed_dict = {}
            feed_dict[self.ls['keep_prob']] = 1.0
            for robot_number in range(self.num_robots):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch_reshaped[robot_number] * self.batch_size))
                idx_i = idx_reshaped[robot_number][start_idx:start_idx+self.batch_size]
                
                feed_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number][idx_i]
                feed_dict[self.action_tensors[robot_number]] = tgt_mu_reshaped[robot_number][idx_i]
                feed_dict[self.precision_tensors[robot_number]] = tgt_prc_reshaped[robot_number][idx_i]

            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)
            average_loss += train_loss
            if i % 500 == 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i, average_loss / 500)
                print 'supervised tf loss is ', (average_loss/800)
                average_loss = 0

        for robot_number in range(self.num_robots):
            # Keep track of tensorflow iterations for loading solver states.

            # Optimize variance.
            A = np.sum(tgt_prc_orig_reshaped[robot_number], 0) + 2 * N_reshaped[robot_number] * T_reshaped[robot_number] * \
                                          self.ent_reg[robot_number] * np.ones((self._dU[robot_number], self._dU[robot_number]))
            A = A / np.sum(tgt_wt_reshaped[robot_number])

            # TODO - Use dense covariance?
            self.var[robot_number] = 1 / np.diag(A)
        return self.policy
 
    def prob(self, obs, robot_number=0):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU[robot_number]
        N, T = obs.shape[:2]

        # Normalize obs.
        try:
            for n in range(N):
                if self.policy[robot_number].scale is not None and self.policy[robot_number].bias is not None:
                    obs[n, :, self.x_idx[robot_number]] = (obs[n, :, self.x_idx[robot_number]].T.dot(self.policy[robot_number].scale)
                                             + self.policy[robot_number].bias).T
        except AttributeError:
            pass  # TODO: Should prob be called before update?

        output = np.zeros((N, T, dU)) 
        for i in range(N):
            feed_dict = {self.obs_tensors[robot_number]: obs[i, :]}
            feed_dict[self.ls['keep_prob']] = 1.0
            with tf.device(self.device_string):
                output[i, :, :] = self.sess.run(self.act_ops[robot_number], feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var[robot_number]), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var[robot_number]), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var[robot_number]), [N, T])
        return output, pol_sigma, pol_prec, pol_det_sigma


    def set_ent_reg(self, ent_reg, robot_number=0):
        """ Set the entropy regularization. """
        self.ent_reg[robot_number] = ent_reg

    # For pickling.
    def __getstate__(self):
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']

        saver = tf.train.Saver()
        check_file = self.checkpoint_file
        saver.restore(self.sess, check_file)
