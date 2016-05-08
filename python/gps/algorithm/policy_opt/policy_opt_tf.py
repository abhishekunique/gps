""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging

import numpy as np

import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
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
        self.dc_mode = self._hyperparams['dc_mode']
        for i, dU_ind in enumerate(dU):
            self.act_ops.append(None)
            self.loss_scalars.append(None)
            self.obs_tensors.append(None)
            self.precision_tensors.append(None)
            self.action_tensors.append(None)
            self.var.append(self._hyperparams['init_var'] * np.ones(dU_ind))
        self.init_network()
        # tv = tf.trainable_variables()
        # num_weights = 4
        # self.shared_vars_array = tv[:4]
        # self.other_vars_array = tv[4:]
        self.init_solver()
        self.sess = tf.Session()
        self.policy = []
        for dU_ind, ot, ap in zip(dU, self.obs_tensors, self.act_ops):
            self.policy.append(TfPolicy(dU_ind, ot, ap, np.zeros(dU_ind), self.sess, self.device_string))
        # List of indices for state (vector) data and image (tensor) data in observation.

        self.x_idx = []
        self.img_idx = []
        i = []
        for robot_number in range(self.num_robots):
            self.x_idx.append([])
            self.img_idx.append([])
            i.append(0)

        for robot_number, robot_params in enumerate(self._hyperparams['network_params']):
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
        allVars = tf.trainable_variables()
        self.shared_vars = []
        for var in allVars:
            if 'shared' in var.name:
                self.shared_vars.append(var)
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

        # if 'restore_all_wts' in self._hyperparams and self._hyperparams['restore_all_wts']:
        #     val_vars = np.load(self._hyperparams['restore_all_wts'])
        #     for i in range(num_weights):
        #         assign_op = tv[i].assign(val_vars[i])
        #         self.sess.run(assign_op)
        #     import IPython
        #     IPython.embed()

        # if self._hyperparams['restore_shared_wts']:
        #     self.restore_shared_wts()
        # merged = tf.merge_all_summaries()
        # writer = tf.train.SummaryWriter('/home/abhigupta/tensorboard_data', graph_def=self.sess.graph)

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        print "dc_mode", self.dc_mode
        if self.dc_mode:
            tf_maps, fc_vars, last_conv_vars, other = tf_map_generator(dim_input=self._dO, dim_output=self._dU, 
                                                                       batch_size=self.batch_size, 
                                                                       network_config=self._hyperparams['network_params'])
        else:
            tf_maps, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, 
                                                                batch_size=self.batch_size, 
                                                                network_config=self._hyperparams['network_params'])
            
        self.obs_tensors = []
        self.action_tensors = []
        self.precision_tensors = []
        self.act_ops = []
        self.loss_scalars = []
        self.fc_vars = fc_vars
        self.last_conv_vars = last_conv_vars
        self.feature_points= []
        if self.dc_mode:
            self.dc_vars = other['dc_vars']
            self.main_vars = other['main_vars']
            self.dc_loss= other['dc_loss']
            self.other = other
        else:
            self.main_vars = None

        for tf_map in tf_maps:
            self.obs_tensors.append(tf_map.get_input_tensor())
            self.action_tensors.append(tf_map.get_target_output_tensor())
            self.precision_tensors.append(tf_map.get_precision_tensor())
            self.act_ops.append(tf_map.get_output_op())
            self.loss_scalars.append(tf_map.get_loss_op())
            self.feature_points.append(tf_map.feature_points)
        self.combined_loss = tf.add_n(self.loss_scalars)
        # self.av = av
        # self.ls = ls

    def init_solver(self):
        """ Helper method to initialize the solver. """
       
        self.solver = TfSolver(loss_scalar=self.combined_loss,
                               solver_name=self._hyperparams['solver_type'],
                               base_lr=self._hyperparams['lr'],
                               lr_policy=self._hyperparams['lr_policy'],
                               momentum=self._hyperparams['momentum'],
                               weight_decay=self._hyperparams['weight_decay'],
                               fc_vars=self.fc_vars,
                               last_conv_vars=self.last_conv_vars,
                               vars_to_opt=self.main_vars)
        import IPython
        IPython.embed()
        if self.dc_mode:
            self.dc_solver = TfSolver(loss_scalar= self.dc_loss,
                                      solver_name=self._hyperparams['solver_type'],
                                      base_lr=self._hyperparams['lr'],
                                      lr_policy=self._hyperparams['lr_policy'],
                                      momentum=self._hyperparams['momentum'],
                                      weight_decay=self._hyperparams['weight_decay'],
                                      vars_to_opt=self.dc_vars)


    def update(self, obs_full, tgt_mu_full, tgt_prc_full, tgt_wt_full, itr_full, inner_itr):
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
            if itr == 0 and inner_itr == 1:
                #TODO: may need to change this
                self.policy[robot_number].x_idx = self.x_idx[robot_number]
                self.policy[robot_number].scale = np.diag(1.0 / np.std(obs[:, self.x_idx[robot_number]], axis=0))
                self.policy[robot_number].bias = -np.mean(obs[:, self.x_idx[robot_number]].dot(self.policy[robot_number].scale), axis=0)
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
        print "itr_full", itr_full
        print "inner_itr", inner_itr
        if itr_full[0] > 0:
            feed_dict = {}
            for robot_number in range(self.num_robots):
                feed_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number]

            num_values = obs_reshaped[0].shape[0]
            conv_values = self.solver.get_last_conv_values(self.sess, feed_dict, num_values, 100)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                feed_dict = {}
                for robot_number in range(self.num_robots):
                    start_idx = int(i * self.batch_size %
                                    (batches_per_epoch_reshaped[robot_number] * self.batch_size))
                    idx_i = idx_reshaped[robot_number][start_idx:start_idx+self.batch_size]
                    feed_dict[self.action_tensors[robot_number]] = tgt_mu_reshaped[robot_number][idx_i]
                    feed_dict[self.precision_tensors[robot_number]] = tgt_prc_reshaped[robot_number][idx_i]
                    feed_dict[self.last_conv_vars[robot_number*2]] = conv_values[robot_number*2][idx_i]
                    feed_dict[self.last_conv_vars[robot_number*2 + 1]] = conv_values[robot_number*2 + 1][idx_i]

                train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss
                if i % 100 == 0 and i != 0:
                    LOGGER.debug('tensorflow iteration %d, average loss %f',
                                 i, average_loss / 100)
                    print 'supervised fc_only tf loss is '
                    print (average_loss/100)
                    average_loss = 0
        average_loss = 0
        average_dc_loss =0
        for i in range(self._hyperparams['iterations']):
            # # Load in data for this batch.
            feed_dict = {}
            for robot_number in range(self.num_robots):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch_reshaped[robot_number] * self.batch_size))
                idx_i = idx_reshaped[robot_number][start_idx:start_idx+self.batch_size]
                feed_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number][idx_i]
                feed_dict[self.action_tensors[robot_number]] = tgt_mu_reshaped[robot_number][idx_i]
                feed_dict[self.precision_tensors[robot_number]] = tgt_prc_reshaped[robot_number][idx_i]
            # if i % 100 == 0:
            #     print(ffpvals)
            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if i % 100 == 0 and i != 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i, average_loss / 100)
                print 'supervised tf loss is '
                print (average_loss/100)
                average_loss = 0
            dc_dict = {}
            for robot_number in range(self.num_robots):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch_reshaped[robot_number] * self.batch_size))
                idx_i = idx_reshaped[robot_number][start_idx:start_idx+self.batch_size]
                dc_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number][idx_i]
                dc_dict[self.action_tensors[robot_number]] = tgt_mu_reshaped[robot_number][idx_i]
                dc_dict[self.precision_tensors[robot_number]] = tgt_prc_reshaped[robot_number][idx_i]
            # if i % 100 == 0:
            #     print(ffpvals)
            dc_loss = self.dc_solver(dc_dict, self.sess, device_string=self.device_string)

            average_dc_loss += dc_loss
            if i % 100 == 0 and i != 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i, average_loss / 100)
                print 'supervised dc loss is '
                print (average_dc_loss/100)
                average_loss = 0


        obs_plot = obs_reshaped[0][1]
        img_plot = self.single_img_with_fp(obs_plot, 0, 64,80,3)
        import matplotlib.image as mpimg
        import time
        mpimg.imsave('test_itr'+str(time.clock())+'.png', img_plot) 

        for robot_number in range(self.num_robots):
            # Keep track of tensorflow iterations for loading solver states.
            #TODO: Need to figure this out, not going to work
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
        try:
            for n in range(N):
                if self.policy[robot_number].scale is not None and self.policy[robot_number].bias is not None:
                    obs[n, :, self.x_idx[robot_number]] = (obs[n, :, self.x_idx[robot_number]].T.dot(self.policy[robot_number].scale)
                                             + self.policy[robot_number].bias).T
        except AttributeError:
            pass  # TODO: Should prob be called before update?

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                feed_dict = {self.obs_tensors[robot_number]: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(self.act_ops[robot_number], feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var[robot_number]), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var[robot_number]), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var[robot_number]), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def single_img_with_fp(self, obs, robot_number, img_height, img_width, num_channels ):
        fps = self.feature_points[robot_number]
        feed_dict = {}
        batch_obs = np.array([obs])
        feed_dict[self.obs_tensors[robot_number]] = batch_obs
        fps_vals= self.sess.run(fps, feed_dict)[0]
        fx_v = fps_vals[::2].astype(int)
        fy_v = fps_vals[1::2].astype(int)
        img = obs[self.img_idx[robot_number]]

        img = img.reshape((num_channels, img_width, img_height))
        img = np.transpose(img, [2,1,0])
        img = 255-img
        img[fx_v, fy_v, :] = 0
        img[fx_v, fy_v, 0] = 255
        return img


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


    def set_ent_reg(self, ent_reg, robot_number=0):
        """ Set the entropy regularization. """
        self.ent_reg[robot_number] = ent_reg

    # For pickling.
    def __getstate__(self):
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': [pol.scale for pol in self.policy],
            'bias': [pol.bias for pol in self.policy],
            'tf_iter': self.tf_iter,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.tf_iter = state['tf_iter']

        # saver = tf.train.Saver()
        # check_file = self.checkpoint_file
        # saver.restore(self.sess, check_file)
