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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors
from pykcca.kcca import KCCA
from pykcca.kernels import LinearKernel
from kernel_cca import KernelCCA

class MLPlotter:
    """
    Plot/save machine learning data
    """
    def __init__(self, title):
        self.error_generator = []
        self.error_discriminator = []
        self.error_generator_only = []
        ### setup plot figure
        self.f, self.axes = plt.subplots(1, 3, figsize=(15,7))
        mng = plt.get_current_fig_manager()
        plt.suptitle(title)
        plt.show(block=False)

    def plot(self):
        f, axes = self.f, self.axes

        for ax in axes:
                ax.clear()

        axes[0].plot(*zip(*self.error_generator), color='k', linestyle='-', label='Train')
        axes[1].plot(*zip(*self.error_discriminator), color='r', linestyle='--', label='Val')
        axes[2].plot(*zip(*self.error_generator_only), color='r', linestyle='--', label='Val')

        axes[0].set_title('Error')
        axes[0].set_ylabel('Percentage')
        axes[0].set_xlabel('Epoch')

        axes[1].set_title('Error')
        axes[1].set_ylabel('Percentage')
        axes[1].set_xlabel('Epoch')

        axes[2].set_title('Error')
        axes[2].set_ylabel('Percentage')
        axes[2].set_xlabel('Epoch')

        f.canvas.draw()

        plt.legend()
        plt.pause(0.01)

    def add_generator(self, itr, err):
        self.error_generator.append((itr, err))

    def add_discriminator(self, itr, err):
        self.error_discriminator.append((itr, err))

    def add_generator_only(self, itr, err):
        self.error_generator_only.append((itr, err))

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'plotter.pkl'), 'w') as f:
            pickle.dump({'error_generator':self.error_generator,
                         'error_discriminator':self.error_discriminator,
                         'error_generator_only':self.error_generator_only},
                        f)

        self.f.savefig(os.path.join(save_dir, 'training.png'))

    def close(self):
        plt.close(self.f)

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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

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
        if 'r0_index_list' in self._hyperparams:
            dO = [len(self._hyperparams['r0_index_list']), len(self._hyperparams['r1_index_list'])]
        else:
            dO = self._dO
        tf_maps, other = tf_map_generator(dim_input_state=dO, dim_input_action=self._dU, batch_size=self.batch_size,
                             network_config=self._hyperparams['network_params'])
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
        # self.combined_loss = tf.add_n(self.loss_scalars)
        self.other= other

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver =TfSolver(loss_scalar=tf.add_n(self.other['all_losses']),
                              solver_name=self._hyperparams['solver_type'],
                              base_lr=self._hyperparams['lr'],
                              lr_policy=self._hyperparams['lr_policy'],
                              momentum=self._hyperparams['momentum'],
                              weight_decay=self._hyperparams['weight_decay'],
                              vars_to_opt=self.other['all_variables'].values())

        # self.dc_solver =TfSolver(loss_scalar=tf.add_n(self.other['dc_loss']),
        #                       solver_name=self._hyperparams['solver_type'],
        #                       base_lr=0.0005, #self._hyperparams['lr'],
        #                       lr_policy=self._hyperparams['lr_policy'],
        #                       momentum=self._hyperparams['momentum'],
        #                       weight_decay=self._hyperparams['weight_decay'],
        #                       vars_to_opt=self.other['dc_variables'].values())

    def train_invariant_autoencoder(self, obs_full, next_obs_full, action_full, obs_extended_full):
        import matplotlib.pyplot as plt
        num_conds, num_samples, T_extended, _ = obs_extended_full[0].shape

        
        # for cond in range(num_conds):
        #     for s_no in range(num_samples):
        #         xs = []
        #         ys = []
        #         for robot_number in range(self.num_robots):
        #             color = ['r', 'b'][robot_number]
        #             x = obs_extended_full[robot_number][cond, s_no, :, 6+2*robot_number]
        #             y = obs_extended_full[robot_number][cond, s_no, :, 8+2*robot_number]
        #             plt.scatter(x, y, c=color)
        #             xs.append(x)
        #             ys.append(y)
        #         plt.plot([xs[0], xs[1]], [ys[0], ys[1]])
        # plt.show()
        # import IPython
        # IPython.embed()

        obs_reshaped = []
        next_obs_reshaped = []
        action_reshaped = []
        #TODO: [SCALE OBSERVATIONS BACK DOWN TO REASONABLE RANGE]
        for robot_number in range(self.num_robots):

            obs = obs_full[robot_number]
            N, T = obs.shape[:2]

            dO = obs.shape[2]
            dU = self._dU[robot_number]

            obs = np.reshape(obs, (N*T, dO))

            next_obs = next_obs_full[robot_number]
            next_obs = np.reshape(next_obs, (N*T, dO))
            

            action = action_full[robot_number]
            action = np.reshape(action, (N*T, dU))

            obs_reshaped.append(obs)
            next_obs_reshaped.append(next_obs)
            action_reshaped.append(action)

        idx = range(N*T)
        np.random.shuffle(idx)
        batches_per_epoch = np.floor(N*T / self.batch_size)
        average_loss = 0
        all_losses = np.zeros((len(self.other['all_losses']),))
        for i in range(self._hyperparams['iterations']):
            feed_dict = {}
            start_idx = int(i * self.batch_size % (batches_per_epoch*self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            for robot_number in range(self.num_robots):
                feed_dict[self.other['state_inputs'][robot_number]] = obs_reshaped[robot_number][idx_i]
                feed_dict[self.other['next_state_inputs'][robot_number]] = next_obs_reshaped[robot_number][idx_i]
                feed_dict[self.other['action_inputs'][robot_number]] = action_reshaped[robot_number][idx_i]
            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)
            all_losses += self.sess.run(self.other['all_losses'], feed_dict)

            average_loss += train_loss

            if i % 1000 == 0 and i != 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i, average_loss / 100)
                print 'supervised tf loss is '
                print (average_loss/1000)
                print (all_losses/1000)
                print("--------------------------")
                average_loss = 0
                all_losses = np.zeros((len(self.other['all_losses']),))
        import IPython
        IPython.embed()
        var_dict = {}
        for k, v in self.other['all_variables'].items():
            var_dict[k] = self.sess.run(v)
        pickle.dump(var_dict, open("subspace_state.pkl", "wb"))



        # num_conds, num_samples, T_extended, dO = obs_extended_full[0].shape
        # cond_feats = np.zeros((num_conds, num_samples, T_extended, 30))
        # for cond in range(num_conds):
        #     for sample_num in range(num_samples):
        #         feed_dict = {self.other['state_inputs'][0]: obs_extended_full[0][cond][sample_num]}
        #         cond_feats[cond, sample_num] = self.sess.run(self.other['state_features_list'][0], feed_dict=feed_dict)
        # np.save("3link_feats.npy", np.asarray(cond_feats))



        cond_feats = np.zeros((num_conds, num_samples, T_extended, 30))
        cond_feats_other = np.zeros((num_conds, num_samples, T_extended, 30))
        l2_loss = 0
        for cond in range(num_conds):
            for sample_num in range(num_samples):
                feed_dict = {self.other['state_inputs'][0]: obs_extended_full[0][cond][sample_num], 
                            self.other['state_inputs'][1]: obs_extended_full[1][cond][sample_num]}
                cond_feats[cond, sample_num] = self.sess.run(self.other['state_features_list'][0], feed_dict=feed_dict)
                cond_feats_other[cond, sample_num] = self.sess.run(self.other['state_features_list'][1], feed_dict=feed_dict)
                l2_loss = np.sum(np.linalg.norm(cond_feats[cond, sample_num] - cond_feats_other[cond, sample_num]))
        print(l2_loss)
        print("RAN THROUGH FEATURES")
        import IPython
        IPython.embed()
        cond_feats = np.reshape(cond_feats, (num_conds*num_samples*T_extended,30))
        cond_feats_other = np.reshape(cond_feats_other, (num_conds*num_samples*T_extended,30))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cond_feats)
        distances, indices = nbrs.kneighbors(cond_feats_other)
        indices = np.reshape(indices, (num_conds, num_samples, T_extended))
        dO_robot0 = obs_extended_full[0].shape[-1]
        obs_full_reshaped = np.reshape(obs_extended_full[0], (num_conds*num_samples*T_extended,dO_robot0))
        print("CHECK NN")
        import IPython
        IPython.embed()
        # for cond in range(num_conds):
        #     for s_no in range(num_samples):
        #         color = ['r', 'b'][robot_number]
        #         for t in range(T_extended):
        #             x = obs_extended_full[1][cond, s_no, t, 8]
        #             y = obs_extended_full[1][cond, s_no, t, 10]
        #             nnbr_currpoint = indices[cond, s_no, t]
        #             x_nbr = obs_full_reshaped[nnbr_currpoint][6]
        #             y_nbr = obs_full_reshaped[nnbr_currpoint][8]
        #             print("X: " + str([x,x_nbr]))
        #             print("Y: " + str([y,y_nbr]))
        #             lines = plt.plot([x,x_nbr], [y,y_nbr])
        # plt.show()
        # import IPython
        # IPython.embed()
        np.save("3link_feats.npy", np.asarray(cond_feats))

        print("done training invariant autoencoder and saving weights")

    def train_invariant_dc(self, obs_full, next_obs_full, action_full, obs_extended_full):
        num_conds, num_samples, T_extended, _ = obs_extended_full[0].shape

        
        for cond in range(num_conds):
            for s_no in range(num_samples):
                for robot_number in range(self.num_robots):
                    color = ['r', 'b'][robot_number]
                    x = obs_extended_full[robot_number][cond, s_no, :, 6+2*robot_number]
                    y = obs_extended_full[robot_number][cond, s_no, :, 8+2*robot_number]
                    plt.scatter(x, y, c=color)
        import IPython
        IPython.embed()
        obs_reshaped = []
        next_obs_reshaped = []
        action_reshaped = []
        #TODO: [SCALE OBSERVATIONS BACK DOWN TO REASONABLE RANGE]
        for robot_number in range(self.num_robots):

            obs = obs_full[robot_number]
            N, T = obs.shape[:2]

            dO = obs.shape[2]
            dU = self._dU[robot_number]

            obs = np.reshape(obs, (N*T, dO))

            next_obs = next_obs_full[robot_number]
            next_obs = np.reshape(next_obs, (N*T, dO))
            

            action = action_full[robot_number]
            action = np.reshape(action, (N*T, dU))

            obs_reshaped.append(obs)
            next_obs_reshaped.append(next_obs)
            action_reshaped.append(action)

        idx = range(N*T)
        np.random.shuffle(idx)
        batches_per_epoch = np.floor(N*T / self.batch_size)




        dc_loss = 0
        pred_error = 0
        average_loss = 0
        all_losses = np.zeros((len(self.other['all_losses']),))
        gen_loss = np.zeros((len(self.other['gen_loss']),))
        all_losses_dc = np.zeros((len(self.other['dc_loss']),))
        mlplt = MLPlotter("Learning curves")
        robot0_classification = np.zeros((self.batch_size,))
        robot1_classification = np.ones((self.batch_size,))
        for i in range(self._hyperparams['iterations']):
            feed_dict = {}
            start_idx = int(i * self.batch_size % (batches_per_epoch*self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            for robot_number in range(self.num_robots):
                feed_dict[self.other['state_inputs'][robot_number]] = obs_reshaped[robot_number][idx_i]
                feed_dict[self.other['next_state_inputs'][robot_number]] = next_obs_reshaped[robot_number][idx_i]
                feed_dict[self.other['action_inputs'][robot_number]] = action_reshaped[robot_number][idx_i]
            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)
            all_losses += self.sess.run(self.other['all_losses'], feed_dict)
            gen_loss += self.sess.run(self.other['gen_loss'], feed_dict)
            predictions_full = self.sess.run(self.other['predictions_full'], feed_dict)
            for robot_number in range(self.num_robots):
                for pred in predictions_full[robot_number]:
                    pred_error += np.sum(pred != [robot0_classification, robot1_classification][robot_number])
            average_loss += train_loss
            if i % 200 == 0 and i != 0:

                print 'supervised tensorflow loss is '
                print (average_loss/200)
                print (all_losses/200)
                print (gen_loss/200)

                mlplt.add_generator(i, average_loss / 200)
                mlplt.add_generator_only(i, np.sum(all_losses) / 200)
                print("PREDICTION ERROR:")
                print(pred_error/(2.0*self.batch_size*200))
                mlplt.add_discriminator(i, pred_error/(2.0*self.batch_size*200))
                pred_error = 0
                print("--------------------------")
                average_loss = 0
                all_losses = np.zeros((len(self.other['all_losses']),))
                gen_loss = np.zeros((len(self.other['gen_loss']),))

            # if i > 1000:
            dc_feed_dict = {}
            start_idx = int(i * self.batch_size % (batches_per_epoch*self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            for robot_number in range(self.num_robots):
                dc_feed_dict[self.other['state_inputs'][robot_number]] = obs_reshaped[robot_number][idx_i]
                dc_feed_dict[self.other['next_state_inputs'][robot_number]] = next_obs_reshaped[robot_number][idx_i]
                dc_feed_dict[self.other['action_inputs'][robot_number]] = action_reshaped[robot_number][idx_i]
            dc_loss += self.dc_solver(dc_feed_dict, self.sess, device_string=self.device_string)
            all_losses_dc += self.sess.run(self.other['dc_loss'], dc_feed_dict)

            if i % 200 == 0 and i != 0:
                print 'supervised dc loss is '
                print (dc_loss/200)
                print (all_losses_dc/200)
                print("--------------------------")
                dc_loss = 0
                all_losses_dc = np.zeros((len(self.other['dc_loss']),))
        mlplt.plot()
        raw_input()
        mlplt.close()
        import IPython
        IPython.embed()
        var_dict = {}
        for k, v in self.other['all_variables'].items():
            var_dict[k] = self.sess.run(v)
        pickle.dump(var_dict, open("subspace_state.pkl", "wb"))



        
        cond_feats = np.zeros((num_conds, num_samples, T_extended, 30))
        cond_feats_other = np.zeros((num_conds, num_samples, T_extended, 30))
        l2_loss = 0
        for cond in range(num_conds):
            for sample_num in range(num_samples):
                feed_dict = {self.other['state_inputs'][0]: obs_extended_full[0][cond][sample_num], 
                            self.other['state_inputs'][1]: obs_extended_full[1][cond][sample_num]}
                cond_feats[cond, sample_num] = self.sess.run(self.other['state_features_list'][0], feed_dict=feed_dict)
                cond_feats_other[cond, sample_num] = self.sess.run(self.other['state_features_list'][1], feed_dict=feed_dict)
                l2_loss = np.sum(np.linalg.norm(cond_feats[cond, sample_num] - cond_feats_other[cond, sample_num]))
        print(l2_loss)
        print("RAN THROUGH FEATURES")
        import IPython
        IPython.embed()
        cond_feats = np.reshape(cond_feats, (num_conds*num_samples*T_extended,30))
        cond_feats_other = np.reshape(cond_feats_other, (num_conds*num_samples*T_extended,30))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cond_feats)
        distances, indices = nbrs.kneighbors(cond_feats_other)
        indices = np.reshape(indices, (num_conds, num_samples, T_extended))
        dO_robot0 = obs_extended_full[0].shape[-1]
        obs_full_reshaped = np.reshape(obs_extended_full[0], (num_conds*num_samples*T_extended,dO_robot0))
        print("CHECK NN")
        import IPython
        IPython.embed()
        for cond in range(num_conds):
            for s_no in range(num_samples):
                color = ['r', 'b'][robot_number]
                for t in range(T_extended):
                    x = obs_extended_full[1][cond, s_no, t, 8]
                    y = obs_extended_full[1][cond, s_no, t, 10]
                    nnbr_currpoint = indices[cond, s_no, t]
                    x_nbr = obs_full_reshaped[nnbr_currpoint][6]
                    y_nbr = obs_full_reshaped[nnbr_currpoint][8]
                    print("X: " + str([x,x_nbr]))
                    print("Y: " + str([y,y_nbr]))
                    lines = plt.plot([x,x_nbr], [y,y_nbr])
        plt.show()
        import IPython
        IPython.embed()
        cond_feats = np.reshape(cond_feats, (num_conds, num_samples, T_extended, 30))
        np.save("3link_feats.npy", np.asarray(cond_feats))
        print("done training invariant DC and saving weights")

    def run_features_forward(self, obs, robot_number):
        feed_dict = {}
        N, T = obs.shape[:2]
        dO = [len(self._hyperparams['r0_index_list']), len(self._hyperparams['r1_index_list'])][robot_number]
        dU = self._dU[robot_number]
        obs = np.reshape(obs, (N*T, dO))
        feed_dict[self.obs_tensors_feat[robot_number]] = obs
        output = self.sess.run(self.feature_points_feat[robot_number], feed_dict=feed_dict)
        output = np.reshape(output, (N, T, 60))
        return output

    def cca(self,obs_full):
        from sklearn.cross_decomposition import CCA
        num_components = 6
        self.fitted_cca = KernelCCA(n_components=num_components, kernel="linear",
            kernel_params={"c": 1, "deg":2, "sigma":6}, eigen_solver='auto',
                 center=True, pgso=True, eta=1e-4)#CCA(num_components)
        Y, X = obs_full
        N = X.shape[0]
        T = X.shape[1]
        X = np.reshape(X, [N*T, -1])
        Y = np.reshape(Y, [N*T, -1])
        self.fitted_cca.fit(X,Y)

        # print "fitting kcca"
        # self.kcca = KernelCCA(n_components=num_components, kernel="linear", gamma=None,
        #          degree=3, coef0=1, kernel_params=None, eigen_solver='auto',
        #          center=True, pgso=True, eta=0, kapa=0.1, nor=2,
        #          max_iter=500, tol=1e-6, copy=True)
        # self.kcca.fit(X, Y)
        # import IPython
        # IPython.embed()
        # kernel = LinearKernel()
        # self.kcca = KCCA(kernel, kernel,
        #             regularization=0,#1e-5,
        #             decomp='icd',
        #             lrank=100,
        #             method='simplified_hardoon_method',
        #             scaler1=lambda x:x,
        #             scaler2=lambda x:x,
        #             ).fit(X,Y)



        return X,Y

    def run_cca(self,obs_full):
        from sklearn.cross_decomposition import CCA
        Y, X = obs_full
        N = X.shape[0]
        T = X.shape[1]
        X = np.reshape(X, [N*T, -1])
        Y = np.reshape(Y, [N*T, -1])
        r1, r0 = self.fitted_cca.transform(X,Y)
        # y1, y2 = self.kcca.transform(X, Y)
        # import IPython
        # IPython.embed()
        return r0

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
