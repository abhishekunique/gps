""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

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
        self.num_robots = len(dU)
        self.tf_iter = [0 for r_no in range(len(dU))]
        self.checkpoint_prefix = self._hyperparams['checkpoint_prefix']
        tf.set_random_seed(self._hyperparams['random_seed'])

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
        self.init_network()
        self.init_solver()
        self.sess = tf.Session()
        self.policy = []
        for dU_ind, ot, ap in zip(dU, self.obs_tensors, self.act_ops):
            taskout = self.ls["task_output"] if ("print_task_out" in self._hyperparams and self._hyperparams["print_task_out"]) else None
            self.policy.append(TfPolicy(dU_ind, ot, ap, np.zeros(dU_ind), self.sess, self.device_string, self.ls['keep_prob'], taskout=taskout))
        # List of indices for state (vector) data and image (tensor) data in observation.

        self.x_idx = []
        self.img_idx = []
        i = []
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
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        # import pickle
        # val_vars, pol_var = pickle.load(open('/home/coline/abhishek_gps/gps/weights_bottleneck_itr0.pkl', 'rb'))
        # #val_vars = pickle.load(open('/home/coline/Downloads/weights_multitaskmultirobot_1.pkl', 'rb'))

        # self.var = pol_var#[pol_var[-2]]
        # for k,v in self.av.items():
        #     if k in val_vars:
        #         assign_op = v.assign(val_vars[k])
        #         self.sess.run(assign_op)


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

        if isinstance(av, list):
            d = {}
            for v in av:
                d[v.name] = v
            av = d
        self.av = av
        self.ls = ls

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.robot_solver =TfSolver(loss_scalar=self.combined_loss,
                                    solver_name=self._hyperparams['solver_type'],
                                    base_lr=self._hyperparams['lr'],
                                    lr_policy=self._hyperparams['lr_policy'],
                                    momentum=self._hyperparams['momentum'],
                                    weight_decay=self._hyperparams['weight_decay'],
                                    # vars_to_opt = self.robot_vars)
                                    vars_to_opt= self.av.values())

    def update(self, obs_full, tgt_mu_full, tgt_prc_full, tgt_wt_full, itr_full, inner_itr, fc_only=False):
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
        itr_reshaped = []
        idx_reshaped = []
        batches_per_epoch_reshaped = []
        tgt_prc_orig_reshaped = []
        for robot_number in range(self.num_robots):
            obs = obs_full[robot_number]
            tgt_mu = tgt_mu_full[robot_number]
            tgt_prc = tgt_prc_full[robot_number]
            tgt_wt = tgt_wt_full[robot_number]
            itr = itr_full[robot_number]
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
            if self.policy.scale is None or self.policy.bias is None:
                #TODO: may need to change this
                self.policy[robot_number].x_idx = self.x_idx[robot_number]
                self.policy[robot_number].scale = np.eye(np.diag(1.0 / (np.std(obs[:, self.x_idx[robot_number]], axis=0) + 1e-8)).shape[0])
                self.policy[robot_number].bias = np.zeros((-np.mean(obs[:, self.x_idx[robot_number]].dot(self.policy[robot_number].scale), axis=0)).shape)
                print("FIND")

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
            itr_reshaped.append(itr)
            idx_reshaped.append(idx)
            batches_per_epoch_reshaped.append(batches_per_epoch)
            tgt_prc_orig_reshaped.append(tgt_prc_orig)

        average_loss = 0
        avg_val_loss = 0
        prev_val_losses = []
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            feed_dict = {}
            for robot_number in range(self.num_robots):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch_reshaped[robot_number] * self.batch_size))
                idx_i = idx_reshaped[robot_number][start_idx:start_idx+self.batch_size]

                feed_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number][idx_i]
                feed_dict[self.action_tensors[robot_number]] = tgt_mu_reshaped[robot_number][idx_i]
                feed_dict[self.precision_tensors[robot_number]] = tgt_prc_reshaped[robot_number][idx_i]

            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)
            val_loss = self.sess.run(self.val_loss, val_dict)
            avg_val_loss += val_loss
            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f',
                             i+1, average_loss / 50)
                average_loss = 0

        for robot_number in range(self.num_robots):
            # Keep track of tensorflow iterations for loading solver states.
            self.tf_iter[robot_number] += self._hyperparams['iterations']

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
        if self.policy[robot_number].scale != None:
            for n in range(N):
                obs[n, :, self.x_idx[robot_number]] = (obs[n, :, self.x_idx[robot_number]].T.dot(self.policy[robot_number].scale)
                                         + self.policy[robot_number].bias).T
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

    def save_shared_wts(self):
        var_dict = {}
        for var in self.shared_vars:
            var_dict[var.name] = var
        saver = tf.train.Saver(var_dict)
        save_path = saver.save(self.sess, "/tmp/model.ckpt")
        print("Shared weights saved in file: %s" % save_path)

    def restore_shared_wts(self):
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/model.ckpt")

    def save_all_wts(self,itr):
        var_list = [var for var in self.solver.trainable_variables]
        var_dict = {var.name: var for var in var_list}
        # saver = tf.train.Saver(var_dict)
        # save_path = saver.save(self.sess, self.checkpoint_prefix + "_itr"+str(itr)+'.ckpt')
        save_path = [self.policy[r].pickle_policy(deg_obs=len(self.x_idx[r])+len(self.img_idx[r]),
                                                  deg_action=self._dU[r], var_dict = var_dict,
                                                  checkpoint_path=self.checkpoint_prefix+'_rn_'+str(r), itr=itr)
                     for r in range(self.num_robots)]
        print "Model saved in files: ",  save_path

    def restore_all_wts(self, itr):
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_prefix + "_itr"+str(itr)+'.ckpt')

    def linearize(self, obs):
        """
        Linearize policy about observations
        Args:
            obs: Numpy array of observations that is T x dO
        """
        # TODO - modify this in case of image features being in the state.
        T = obs.shape[0]

        # Initialize
        pol_K = np.empty((T, self._dU, self._dO))
        pol_k = np.empty((T, self._dU))

        # Perform scaling
        x = obs.copy() # Store pre-scaled
        if self.policy.scale is not None:
            obs[:, self.x_idx] = \
                    obs[:, self.x_idx].dot(self.policy.scale) + \
                    self.policy.bias

        # Constant bias/gain matrices
        feed_dict = {self.obs_tensor: obs}
        pol_k = self.sess.run(self.act_op, feed_dict=feed_dict)
        for u in range(self._dU):
            pol_K[:, u, :] = self.sess.run(self.grads[u], feed_dict=feed_dict)

        # Correct bias
        for t in range(T):
            if self.policy.scale is not None:
                pol_K[t, :, :] = pol_K[t, :, :].dot(self.policy.scale)
            pol_k[t, :] -= pol_K[t, :, :].dot(x[t])

        return pol_K, pol_k

    def set_ent_reg(self, ent_reg, robot_number=0):
        """ Set the entropy regularization. """
        self.ent_reg[robot_number] = ent_reg

    def save_model(self, fname):
        LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname)

    def restore_model(self, fname):
        self.saver.restore(self.sess, fname)
        LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save_model(f.name) # TODO - is this implemented.
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
            os.remove(f.name+'.meta')
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': [pol.scale for pol in self.policy],
            'bias': [pol.bias for pol in self.policy],
            'tf_iter': self.tf_iter,
            'x_idx': self.policy.x_idx,
            'chol_pol_covar': self.policy.chol_pol_covar,
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.policy.x_idx = state['x_idx']
        self.policy.chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']

        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)

        # saver = tf.train.Saver()
        # check_file = self.checkpoint_file
        # saver.restore(self.sess, check_file)
    def update_ee(self, obs_full, tgt_mu_full, tgt_prc_full, tgt_wt_full,ee_full, itr_full, inner_itr):
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
        ee_reshaped = []
        itr_reshaped = []
        idx_reshaped = []
        batches_per_epoch_reshaped = []
        tgt_prc_orig_reshaped = []
        for robot_number in range(self.num_robots):
            obs = obs_full[robot_number]
            tgt_mu = tgt_mu_full[robot_number]
            tgt_prc = tgt_prc_full[robot_number]
            tgt_wt = tgt_wt_full[robot_number]
            itr = itr_full[robot_number]
            ee = ee_full[robot_number]
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
            ee = np.reshape(ee, (N*T,ee.shape[-1]))
            # Fold weights into tgt_prc.
            tgt_prc = tgt_wt * tgt_prc

            # TODO: Find entries with very low weights?

            # Normalize obs, but only compute normalzation at the beginning.
            if itr == 0 and inner_itr == 1:
                #TODO: may need to change this
                self.policy[robot_number].x_idx = self.x_idx[robot_number]
                self.policy[robot_number].scale = np.eye(np.diag(1.0 / (np.std(obs[:, self.x_idx[robot_number]], axis=0) + 1e-8)).shape[0])
                self.policy[robot_number].bias = np.zeros((-np.mean(obs[:, self.x_idx[robot_number]].dot(self.policy[robot_number].scale), axis=0)).shape)
                print("FIND")

            obs[:, self.x_idx[robot_number]] = obs[:, self.x_idx[robot_number]].dot(self.policy[robot_number].scale) + self.policy[robot_number].bias

            # Assuming that N*T >= self.batch_size.
            batches_per_epoch = np.floor(N*T / self.batch_size)
            idx = range(N*T)

            np.random.shuffle(idx)
            obs_reshaped.append(obs)
            tgt_mu_reshaped.append(tgt_mu)
            tgt_prc_reshaped.append(tgt_prc)
            tgt_wt_reshaped.append(tgt_wt)
            ee_reshaped.append(ee)
            N_reshaped.append(N)
            T_reshaped.append(T)
            itr_reshaped.append(itr)
            idx_reshaped.append(idx)
            batches_per_epoch_reshaped.append(batches_per_epoch)
            tgt_prc_orig_reshaped.append(tgt_prc_orig)

        average_loss = 0
        avg_taskloss = 0
        avg_val_loss =0
        all_val_losses = []
        continue_iters =True
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            feed_dict = {}
            robot_dict = {}
            if continue_iters:
                robot_dict[self.ls['keep_prob']] = 0.8
                for robot_number in range(self.num_robots):
                    start_idx = int(i * self.batch_size %
                                    (batches_per_epoch_reshaped[robot_number] * self.batch_size))
                    idx_i = idx_reshaped[robot_number][start_idx:start_idx+self.batch_size]
                    # if robot_number in self.val_agents:
                    #     val_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number][idx_i]
                    #     val_dict[self.action_tensors[robot_number]] = tgt_mu_reshaped[robot_number][idx_i]
                    #     val_dict[self.precision_tensors[robot_number]] = tgt_prc_reshaped[robot_number][idx_i]
                    # else:
                    feed_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number][idx_i]
                    feed_dict[self.action_tensors[robot_number]] = tgt_mu_reshaped[robot_number][idx_i]
                    feed_dict[self.precision_tensors[robot_number]] = tgt_prc_reshaped[robot_number][idx_i]

                        #feed_dict[self.ls['ee_input'][robot_number]] = ee_reshaped[robot_number][idx_i]
                    robot_dict.update(feed_dict)
                    #robot_dict[self.ls['task_output'][robot_number]] = ee_reshaped[robot_number][idx_i]

                #task_loss = self.task_solver(feed_dict, self.sess, device_string=self.device_string)
                train_loss = self.robot_solver(robot_dict, self.sess, device_string=self.device_string)
                #val_loss = self.sess.run(self.val_loss, val_dict)
                #avg_val_loss += val_loss
                average_loss += train_loss
                # avg_taskloss += task_loss
                if (i + 1) % 500 == 0:
                    LOGGER.debug('tensorflow iteration %d, average loss %f',
                                 i, average_loss / 500)
                    print 'robot loss is ', (average_loss/500)
                    #print 'task loss is ', (avg_taskloss/500
                    #print 'task loss is ', (avg_val_loss/500)
                    all_val_losses.append(average_loss)
                    average_loss = 0
                    avg_val_loss = 0
                    if len(all_val_losses) > 2:
                        print "checking past val losses", all_val_losses
                        if all_val_losses[-1] > all_val_losses[-2] and all_val_losses[-2] > all_val_losses[-3]:
                            print "Val loss is increasing, stop iters"
                            continue_iters = False
                # avg_taskloss = 0
        for robot_number in range(self.num_robots):
            # Keep track of tensorflow iterations for loading solver states.
            self.tf_iter[robot_number] += self._hyperparams['iterations']

            # Optimize variance.
            A = np.sum(tgt_prc_orig_reshaped[robot_number], 0) + 2 * N_reshaped[robot_number] * T_reshaped[robot_number] * \
                                          self.ent_reg[robot_number] * np.ones((self._dU[robot_number], self._dU[robot_number]))
            A = A / np.sum(tgt_wt_reshaped[robot_number])

            # TODO - Use dense covariance?
            self.var[robot_number] = 1 / np.diag(A)
        return self.policy
