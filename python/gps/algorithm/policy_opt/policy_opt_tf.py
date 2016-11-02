""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging

import numpy as np

import tensorflow as tf
import pickle
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.tf_utils import TfSolver
LOGGER = logging.getLogger(__name__)
import pickle

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
        self.ncond = self._hyperparams['ncond']
        for i, dU_ind in enumerate(dU):
            self.act_ops.append(None)
            self.loss_scalars.append(None)
            self.obs_tensors.append(None)
            self.precision_tensors.append(None)
            self.action_tensors.append(None)
            self.var.append(self._hyperparams['init_var'] * np.ones(dU_ind))
        self.act_ops_action = []
        self.loss_scalars_action = []
        self.obs_tensors_action = []
        self.precision_tensors_action = []
        self.action_tensors_action = []
        self.solver_action = None
        for i, dU_ind in enumerate(dU):
            self.act_ops_action.append(None)
            self.loss_scalars_action.append(None)
            self.obs_tensors_action.append(None)
            self.precision_tensors_action.append(None)
            self.action_tensors_action.append(None)

        if not self._hyperparams['run_feats']:
            self.init_network()
            self.init_solver()
        self.tf_vars = tf.trainable_variables()
        if self._hyperparams['run_feats']:
            self.init_feature_space()
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
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        if self._hyperparams['load_weights'] and self._hyperparams['run_feats']:
            import pickle
            val_vars = pickle.load(open(self._hyperparams['load_weights'], 'rb'))
            for k,v in self.var_list_feat.items():
                if k in val_vars:
                    print(k)
                    assign_op = v.assign(val_vars[k])
                    self.sess.run(assign_op)

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        dO = [len(self._hyperparams['r0_index_list']), len(self._hyperparams['r1_index_list'])]
        tf_maps, var_list, other = tf_map_generator(dim_input=dO, dim_output=self._dU, batch_size=self.batch_size,
                             network_config=self._hyperparams['network_params'], ncond=self.ncond)
        self.obs_tensors = []
        self.action_tensors = []
        self.precision_tensors = []
        self.act_ops = []
        self.loss_scalars = []
        self.feature_points= []
        self.individual_losses = []
        for tf_map in tf_maps:
            self.obs_tensors.append(tf_map.get_input_tensor())
            self.action_tensors.append(tf_map.get_target_output_tensor())
            self.precision_tensors.append(tf_map.get_precision_tensor())
            self.act_ops.append(tf_map.get_output_op())
            self.loss_scalars.append(tf_map.get_loss_op())
            self.feature_points.append(tf_map.feature_points)
            self.individual_losses.append(tf_map.individual_losses)
        self.combined_loss = tf.add_n(self.loss_scalars)
        self.var_list= var_list
        self.other = other

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver = TfSolver(loss_scalar=#tf.add_n(self.other['indiv_losses']) + 
                                            #tf.add_n(self.other['gen_loss']) 
                                            tf.add_n(self.other['ae_loss'])
                                            + self.other['contrast_loss'],#self.combined_loss,
                              solver_name=self._hyperparams['solver_type'],
                              base_lr=0.001,#self._hyperparams['lr'],
                              lr_policy=self._hyperparams['lr_policy'],
                              momentum=self._hyperparams['momentum'],
                              weight_decay=self._hyperparams['weight_decay'],
                              vars_to_opt=self.var_list.values())

        self.dc_solver = TfSolver(loss_scalar=tf.add_n(self.other['dc_loss']),
                      solver_name=self._hyperparams['solver_type'],
                      base_lr=0.001,#self._hyperparams['lr'],
                      lr_policy=self._hyperparams['lr_policy'],
                      momentum=self._hyperparams['momentum'],
                      weight_decay=self._hyperparams['weight_decay'],
                      vars_to_opt=self.other['dc_vars'])


    ### GAN structure for training ###
    def train_invariant_autoencoder(self, itr, obs_full, shaped_full, matched_full, weight_save_file):
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
        # traj_feats = self.run_features_forward(obs_full[0][0][0], 0)

        nconds = len(obs_full[0])
        import pickle
        val_vars = pickle.load(open(self._hyperparams['load_file'], 'rb'))
        self.var_list_feat = {}
        for v in tf.trainable_variables():
            self.var_list_feat[v.name] = v
        for k,v in self.var_list_feat.items():
            if k in val_vars:   
                print("COST LOAD")
                print(k)         
                assign_op = v.assign(val_vars[k])
                self.session.run(assign_op)

        # traj_feats = [self.run_features_forward(obs_full[0][c], 0) for c in range(nconds)]

        N_reshaped = [[] for c in range(nconds)]
        T_reshaped = [[] for c in range(nconds)]
        obs_reshaped = [[] for c in range(nconds)]
        shaped_cost_reshaped = [[] for c in range(nconds)]
        idx_reshaped = [[] for c in range(nconds)]
        batches_per_epoch_reshaped = [[] for c in range(nconds)]
        for robot_number in range(self.num_robots):
            for c in range(nconds):
                obs = obs_full[robot_number][c]
                shaped_cost = shaped_full[robot_number][c]
                N, T = obs.shape[:2]
                dU, dO = self._dU[robot_number], self._dO[robot_number]

                # Reshape inputs.
                obs = np.reshape(obs, (N*T, dO))
                print N, T, shaped_cost.shape, obs.shape
                shaped_cost = np.reshape(shaped_cost, (N*T,))
                # Assuming that N*T >= self.batch_size.
                batches_per_epoch = np.floor(N*T / self.batch_size)
                idx = range(N*T)

                np.random.shuffle(idx)
                obs_reshaped[c].append(obs)
                shaped_cost_reshaped[c].append(shaped_cost)
                N_reshaped[c].append(N)
                T_reshaped[c].append(T)
                idx_reshaped[c].append(idx)
                batches_per_epoch_reshaped[c].append(batches_per_epoch)
        # matched_reshaped = []

        # for robot_number in range(self.num_robots):
        #     dU, dO = self._dU[robot_number], self._dO[robot_number]
        #     matched_data = np.zeros((0, dO))
        #     for c in range(nconds):
        #         obs = matched_full[robot_number][c].get_X()
        #         N, T = obs.shape[:2]
        #         obs = np.reshape(obs, (N*T, dO))
        #         # import IPython
        #         # IPython.embed()
        #         matched_data = np.concatenate((matched_data, obs), axis = 0)
        #     matched_reshaped.append(matched_data)

        matched_reshaped = []
        for robot_number in range(self.num_robots):
            obs = matched_full[robot_number]
            N, T = obs.shape[:2]
            dO = [len(self._hyperparams['r0_index_list']), len(self._hyperparams['r1_index_list'])][robot_number]
            dU = self._dU[robot_number]
            obs = np.reshape(obs, (N*T, dO))
            matched_reshaped.append(obs)

        average_loss = 0
        average_dc_acc = np.zeros((nconds, 5))
        average_dc_loss = 0
        contrast_loss = 0
        ae_loss = 0
        should_disc = True
        maxitr = self._hyperparams['iterations']
        if itr == 0:
            maxitr = 30000
        else:
            maxitr = 1000
        maxitr = 0
        for i in range(maxitr):
            feed_dict = {}
            for robot_number in range(self.num_robots):
                idx_i = np.random.choice(len(matched_reshaped[robot_number]), self.batch_size, replace=False)
                feed_dict[self.other['contrast_input'][robot_number]] = matched_reshaped[robot_number][idx_i]
                # for c in range(nconds):
                #     start_idx = int(i * self.batch_size %
                #                     (batches_per_epoch_reshaped[c][robot_number] * self.batch_size))
                #     idx_i = idx_reshaped[c][robot_number][start_idx:start_idx+self.batch_size]
                #     feed_dict[self.other['nn_inputs'][robot_number][c][0]] = obs_reshaped[c][robot_number][idx_i]
                #     if robot_number == 0:
                #         feed_dict[self.other['nn_inputs'][robot_number][c][1]] = shaped_cost_reshaped[c][robot_number][idx_i]
            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)

            # if np.isnan(train_loss):
            #     import IPython
            #     IPython.embed()
            # contrast_loss += self.sess.run(self.other['contrast_loss'], feed_dict)
            # ae_loss += self.sess.run(tf.add_n(self.other['ae_loss']), feed_dict)
            average_loss += train_loss
            if i % 100 == 0 and i != 0:
                # LOGGER.debug('tensorflow iteration %d, average loss %f',
                #              i, average_loss / 100)
                print i, maxitr
                print 'supervised tf loss is '
                print (average_loss/100)
                # print (contrast_loss/100.0)
                # print (ae_loss/100.0)
                feed_dict = {}
                for robot_number in range(self.num_robots):
                    feed_dict[self.other['contrast_input'][robot_number]] = matched_reshaped[robot_number]
                print self.solver(feed_dict, self.sess, device_string=self.device_string)
                average_loss = 0
                contrast_loss = 0
                ae_loss = 0

                # r = self.reward_forward(obs_reshaped, nconds)
                # for rb in range(self.num_robots):
                #     for cc in range(nconds):
                #         pred = np.reshape(r[cc][rb], (-1))
                #         # print pred.shape, shaped_cost_reshaped[cc][rb].shape, (pred - shaped_cost_reshaped[cc][rb]).shape
                #         norm = np.linalg.norm(pred - shaped_cost_reshaped[cc][rb])/np.sqrt(pred.shape[0])
                #         print "r" + str(rb) + " c" + str(cc) + " " + str(norm)

                #import IPython
                #IPython.embed()
                # print np.linalg.norm(self.reward_forward(obs_full[0][0], 0)  -
                #     shaped_cost_reshaped[0][0])/np.sqrt(shaped_cost_reshaped[0][0].shape[0])
            # dc_feed_dict = {}
            # # for robot_number in range(self.num_robots):
            # #     start_idx = int(i * self.batch_size %
            # #                    (batches_per_epoch_reshaped[robot_number] * self.batch_size))
            # #     idx_i = idx_reshaped[robot_number][start_idx:start_idx+self.batch_size]
            # #     dc_feed_dict[self.obs_tensors[robot_number]] = obs_reshaped[robot_number][idx_i]
            # #     if robot_number == 0:
            # #         dc_feed_dict[self.action_tensors[robot_number]] = shaped_cost_reshaped[robot_number][idx_i]
            # # # if should_disc:
            # for j in range(1):
            #     for robot_number in range(self.num_robots):
            #         for c in range(nconds):
            #             start_idx = int((i+j*31) * self.batch_size %
            #                             (batches_per_epoch_reshaped[c][robot_number] * self.batch_size))
            #             idx_i = idx_reshaped[c][robot_number][start_idx:start_idx+self.batch_size]
            #             feed_dict[self.other['nn_inputs'][robot_number][c][0]] = obs_reshaped[c][robot_number][idx_i]
            #             if robot_number == 0:
            #                 feed_dict[self.other['nn_inputs'][robot_number][c][1]] = shaped_cost_reshaped[c][robot_number][idx_i]
            #     # dc_loss = self.dc_solver(feed_dict, self.sess, device_string=self.device_string)
            #     # average_dc_loss += dc_loss
            # prediction = self.sess.run(self.other['dc_output'], feed_dict)

            # for c in range(nconds):
            #     p0 = prediction[c][0][:, 0]
            #     p1 = prediction[c][1][:, 1]
            #     tot = p0.shape[0] + p1.shape[0]
            #     correct = np.sum(p0 > 0.5) + np.sum(p1 > 0.5)
            #     average_dc_acc[c][0] += correct / float(tot)
            #     average_dc_acc[c][1] += np.mean(p0)
            #     average_dc_acc[c][2] += np.std(p0)
            #     average_dc_acc[c][3] += np.mean(p1)
            #     average_dc_acc[c][4] += np.std(p1)

            # if i % 100 == 0 and i != 0:
            #     np.set_printoptions(suppress=True)
            #     LOGGER.debug('tensorflow iteration %d, average loss %f',
            #                  i, average_dc_loss / 100)
            #     print 'supervised dc loss is '
            #     print (average_dc_loss/100)
            #     print (average_dc_acc/100)
            #     # should_disc = average_dc_acc < 80
            #     average_dc_loss = 0
            #     average_dc_acc = np.zeros((nconds, 5))
            #     # if i == 1000:
            #     #     import IPython
            #     #     IPython.embed()


        var_dict = {}
        for k, v in self.var_list.items():
            var_dict[k] = self.sess.run(v)
        pickle.dump(var_dict, open(weight_save_file, "wb"))

        # obs = obs_full[0]
        # N, T = obs.shape[:2]
        # dO = obs.shape[2]
        # obs = np.reshape(obs, (N*T, dO))
        # y = shaped_cost_reshaped[0]
        # A = obs
        # x = np.linalg.lstsq(A, y)
        #y_pred = A.dot(x)
        # traj_feats = None
        #need to take mean here
        # np.save("fps_r0.npy", traj_feats)
        print("done training invariant autoencoder and saving weights")
        traj_feats = np.array([self.run_features_forward(obs_full[0][c], 0) for c in range(nconds)])

        # import IPython
        # IPython.embed()
        traj_feats_one = np.array([self.run_features_forward(obs_full[0][c], 0) for c in range(nconds)])
        mtraj_feats = np.mean(traj_feats, axis=1)
        print "err", np.linalg.norm(traj_feats - mtraj_feats[:, None, :, :])
        return traj_feats, var_dict

    def reward_forward(self, obs_reshaped, nconds):
        feed_dict = {}
        for robot_number in range(self.num_robots):
                for c in range(nconds):
                    feed_dict[self.other['nn_inputs'][robot_number][c][0]] = obs_reshaped[c][robot_number]
        output = self.sess.run(self.other['nn_output'], feed_dict=feed_dict)
        return output

    def run_features_forward(self, obs, robot_number):
        feed_dict = {}
        N, T = obs.shape[:2]
        dO = obs.shape[2]
        #dO = [len(self._hyperparams['r0_index_list']), len(self._hyperparams['r1_index_list'])][robot_number]
        dU = self._dU[robot_number]
        obs = np.reshape(obs, (N*T, dO))
        obs = obs[:, [self._hyperparams['r0_index_list'], self._hyperparams['r1_index_list']][robot_number]]
        feed_dict[self.other['contrast_input'][robot_number]] = obs
        output = self.sess.run(self.other['ae_feats'][robot_number], feed_dict=feed_dict)
        output = np.reshape(output, (N, T, -1))
        return output

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

            average_loss += train_loss
            if i % 100 == 0 and i != 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i, average_loss / 100)
                print 'supervised tf loss is '
                print (average_loss/100)
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
