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
import time
LOGGER = logging.getLogger(__name__)
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors
class MLPlotter:
    """
    Plot/save machine learning data
    """
    def __init__(self, title):
        self.error_generator = []
        self.error_discriminator = []
        self.error_generator_only = []
        self.grad1 = []
        self.grad2 = []
        self.grad3 = []
        self.grad4 = []
        self.grad5 = []
        self.grad6 = []

        ### setup plot figure
        self.f, self.axes = plt.subplots(3, 3, figsize=(15,7))
        mng = plt.get_current_fig_manager()
        plt.suptitle(title)
        plt.show(block=False)

    def plot(self):
        f, axes = self.f, self.axes

        for ax in axes:
            for a in ax:
                a.clear()

        axes[0][0].plot(*zip(*self.error_generator), color='k', linestyle='-', label='Train')
        axes[0][1].plot(*zip(*self.error_discriminator), color='r', linestyle='--', label='Val')
        axes[0][2].plot(*zip(*self.error_generator_only), color='r', linestyle='--', label='Val')

        axes[0][0].set_title('Error Generator')
        axes[0][0].set_ylabel('Percentage')
        axes[0][0].set_xlabel('Epoch')

        axes[0][1].set_title('Error Discriminator')
        axes[0][1].set_ylabel('Percentage')
        axes[0][1].set_xlabel('Epoch')

        axes[0][2].set_title('Error Generator Only')
        axes[0][2].set_ylabel('Percentage')
        axes[0][2].set_xlabel('Epoch')

        axes[1][0].plot(*zip(*self.grad1), color='k', linestyle='-')
        axes[1][1].plot(*zip(*self.grad2), color='k', linestyle='-')
        axes[1][2].plot(*zip(*self.grad3), color='k', linestyle='-')
        axes[2][0].plot(*zip(*self.grad4), color='k', linestyle='-')
        axes[2][1].plot(*zip(*self.grad5), color='k', linestyle='-')
        axes[2][2].plot(*zip(*self.grad6), color='k', linestyle='-')

        f.canvas.draw()

        plt.legend()
        plt.pause(0.01)

    def add_generator(self, itr, err):
        self.error_generator.append((itr, err))

    def add_discriminator(self, itr, err):
        self.error_discriminator.append((itr, err))

    def add_generator_only(self, itr, err):
        self.error_generator_only.append((itr, err))
    def add_grad(self, itr, err):
        self.grad1.append((itr,err[0]))
        self.grad2.append((itr,err[1]))
        self.grad3.append((itr,err[2]))
        self.grad4.append((itr,err[3]))
        self.grad5.append((itr,err[4]))
        self.grad6.append((itr,err[5]))

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

        if not False: #self._hyperparams['run_feats']:
            self.init_network()
            self.init_solver()
        self.tf_vars = tf.trainable_variables()
        # if self._hyperparams['run_feats']:
        #     self.init_feature_space()
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
            all_vars = self.other['all_variables'].copy()
            # all_vars.update(self.other['dc_variables'])
            for k,v in all_vars.items():
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
        #self._hyperparams['lr'] = self.other['hyperparams'][3]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver =TfSolver(loss_scalar=tf.add_n(self.other['all_losses']),#tf.add_n(self.other['gen_loss']),
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

    def train_invariant_autoencoder(self, obs_full, next_obs_full, action_full, obs_extended_full, ee_full):
        import matplotlib.pyplot as plt
        print "in train"
        num_conds, num_samples, T_extended, _ = obs_extended_full[0].shape

        
        # for cond in range(num_conds):
        #     for s_no in range(num_samples):
        #         xs = []
        #         ys = []
        #         for robot_number in range(self.num_robots):
        #             color = ['r', 'b'][robot_number]
        #             x = ee_full[robot_number][cond*4+ s_no, :, 0]
        #             y = ee_full[robot_number][cond*4+ s_no, :, 2]
        #             plt.scatter(x, y, c=color)
        #             xs.append(x)
        #             ys.append(y)
        #         #plt.plot(xs[0], ys[0], xs[1], ys[1])
        # plt.show()
        # import IPython
        # IPython.embed()

        obs_reshaped = []
        next_obs_reshaped = []
        action_reshaped = []
        ee_reshaped = []
        #TODO: [SCALE OBSERVATIONS BACK DOWN TO REASONABLE RANGE]
        for robot_number in range(self.num_robots):

            obs = obs_full[robot_number]
            N, T = obs.shape[:2]

            dO = obs.shape[2]
            dU = self._dU[robot_number]

            obs = np.reshape(obs, (N*T, dO))
            ee = ee_full[robot_number]
            ee = np.reshape(ee, (N*T, ee.shape[-1]))
            ee_reshaped.append(ee)
            # next_obs = next_obs_full[robot_number]
            # next_obs = np.reshape(next_obs, (N*T, dO))
            

            action = action_full[robot_number]
            action = np.reshape(action, (N*T, dU))

            obs_reshaped.append(obs)
            #next_obs_reshaped.append(next_obs)
            action_reshaped.append(action)
        mlplt = MLPlotter("Learning curves")
        idx = range(N*T)
        np.random.shuffle(idx)
        batches_per_epoch = np.floor(N*T / self.batch_size)
        average_loss = 0
        contrastive =0
        checkitr = 200
        all_losses = np.zeros((len(self.other['gen_loss']),))
        print "starting tf iters"
        grads = np.zeros(6)
        # import IPython
        # IPython.embed()
        time0 = time.clock()
        # import IPython
        # IPython.embed()
        if self._hyperparams['strike']:
            obs_full_reshaped = [np.reshape(obs_full[i], (num_conds, num_samples, T,-1)) for i in range(self.num_robots)]
            cond_feats = np.zeros((num_conds, num_samples, T, 32))
            cond_feats_other = np.zeros((num_conds, num_samples, T, 32))
            l2_loss = 0
            #import IPython
            #IPython.embed()
            for cond in range(num_conds):
                for sample_num in range(num_samples):
                    feed_dict = {self.other['state_inputs'][0]: obs_full_reshaped[0][cond][sample_num], 
                                 self.other['state_inputs'][1]: obs_full_reshaped[1][cond][sample_num]}
                    cond_feats[cond, sample_num] = self.sess.run(self.other['state_features_list'][0], feed_dict=feed_dict)
                    cond_feats_other[cond, sample_num] = self.sess.run(self.other['state_features_list'][1], feed_dict=feed_dict)
                    l2_loss = np.sum(np.linalg.norm(cond_feats[cond, sample_num] - cond_feats_other[cond, sample_num]))
            print(l2_loss)
            cond_feats = np.reshape(cond_feats, (num_conds*num_samples*T,32))

            #plt.imshow(img)
            cond_feats = np.reshape(cond_feats, (num_conds, num_samples, T, 32))
            np.save("3link_feats.npy", cond_feats)
            import IPython
            IPython.embed()
        for i in range(self._hyperparams['iterations']):
            feed_dict = {}
            start_idx = int(i * self.batch_size % (batches_per_epoch*self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            for robot_number in range(self.num_robots):
                feed_dict[self.other['state_inputs'][robot_number]] = obs_reshaped[robot_number][idx_i]
                # feed_dict[self.other['ee_inputs'][robot_number]] = ee_reshaped[robot_number][idx_i]
                # feed_dict[self.other['next_state_inputs'][robot_number]] = next_obs_reshaped[robot_number][idx_i]
                # feed_dict[self.other['action_inputs'][robot_number]] = action_reshaped[robot_number][idx_i]
            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)
            #a, c, grad = self.sess.run([self.other['gen_loss'], self.other['contrastive'], self.other['gradients']], feed_dict)
            a, c= self.sess.run([self.other['gen_loss'], self.other['contrastive']], feed_dict)
            all_losses+= a
            contrastive+= c
            #all_losses += self.sess.run(self.other['gen_loss'], feed_dict)
            #contrastive += self.sess.run(self.other['contrastive'], feed_dict)
            #grad = self.sess.run(self.other['gradients'], feed_dict)
            # grads[0] +=np.mean(np.square(grad[0]))
            # grads[1] +=np.mean(np.square(grad[1]))
            # grads[2] += np.mean(np.square(grad[2]))
            # grads[3] +=np.mean(np.square(grad[3]))
            # grads[4] +=np.mean(np.square(grad[4]))
            # grads[5] += np.mean(np.square(grad[5]))
            average_loss += train_loss/200

            if i % checkitr == 0 and i != 0:
                time1 = time.clock()
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i, average_loss / checkitr)
                print 'iter', i, 'supervised tf loss is '
                print (average_loss)
                print (all_losses/checkitr)

                print contrastive/checkitr
                print "time for", checkitr, "iters", time1-time0
                time0 = time1
                print("--------------------------")
                mlplt.add_generator_only(i, np.sum(all_losses) / 200)
                mlplt.add_discriminator(i, contrastive/200)
                mlplt.add_generator(i, average_loss)
                #mlplt.add_grad(i, grads/200)
                average_loss = 0
                contrastive =0
                all_losses = np.zeros((len(self.other['gen_loss']),))
                mlplt.plot()

        #import IPython
        #IPython.embed()
        mlplt.plot()
        var_dict = {}
        for k, v in self.other['all_variables'].items():
            var_dict[k] = self.sess.run(v)
        pickle.dump(var_dict, open("img_weights.pkl", "wb"))



        # num_conds, num_samples, T_extended, dO = obs_extended_full[0].shape
        # cond_feats = np.zeros((num_conds, num_samples, T_extended, 30))
        # for cond in range(num_conds):
        #     for sample_num in range(num_samples):
        #         feed_dict = {self.other['state_inputs'][0]: obs_extended_full[0][cond][sample_num]}
        #         cond_feats[cond, sample_num] = self.sess.run(self.other['state_features_list'][0], feed_dict=feed_dict)
        # np.save("3link_feats.npy", np.asarray(cond_feats))


        obs_full_reshaped = [np.reshape(obs_full[i], (num_conds, num_samples, T,-1)) for i in range(self.num_robots)]
        cond_feats = np.zeros((num_conds, num_samples, T, 32))
        cond_feats_other = np.zeros((num_conds, num_samples, T, 32))
        l2_loss = 0
        #import IPython
        #IPython.embed()
        for cond in range(num_conds):
            for sample_num in range(num_samples):
                feed_dict = {self.other['state_inputs'][0]: obs_full_reshaped[0][cond][sample_num], 
                            self.other['state_inputs'][1]: obs_full_reshaped[1][cond][sample_num]}
                cond_feats[cond, sample_num] = self.sess.run(self.other['state_features_list'][0], feed_dict=feed_dict)
                cond_feats_other[cond, sample_num] = self.sess.run(self.other['state_features_list'][1], feed_dict=feed_dict)
                l2_loss = np.sum(np.linalg.norm(cond_feats[cond, sample_num] - cond_feats_other[cond, sample_num]))
        print(l2_loss)
        print("RAN THROUGH FEATURES")
        import IPython
        IPython.embed()
        cond_feats = np.reshape(cond_feats, (num_conds*num_samples*T,32))
        cond_feats_other = np.reshape(cond_feats_other, (num_conds*num_samples*T,32))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cond_feats)
        distances, indices = nbrs.kneighbors(cond_feats_other)
        indices = np.reshape(indices, (num_conds, num_samples, T))
        dO_robot0 = obs_full_reshaped[0].shape[-1]
        #obs_full_reshaped = np.reshape(obs_full[0], (num_conds*num_samples*T,dO_robot0))
        #ee_full_reshaped = np.reshape(ee_full
        print("CHECK NN")
        #import IPython
        #IPython.embed()
        leng = 0
        for cond in range(num_conds):
            for s_no in range(num_samples):
                color = ['r', 'b'][robot_number]
                for t in range(T):
                    x = ee_full[1][cond*num_samples+ s_no, t, 0]
                    y = ee_full[1][cond*num_samples+ s_no, t, 2]
                    # x = obs_extended_full[1][cond, s_no, t, 8]
                    # y = obs_extended_full[1][cond, s_no, t, 10]
                    nnbr_currpoint = indices[cond, s_no, t]
                    x_nbr = ee_reshaped[0][nnbr_currpoint][0]
                    y_nbr = ee_reshaped[0][nnbr_currpoint][2]
                    print("X: " + str([x,x_nbr]))
                    print("Y: " + str([y,y_nbr]))
                    lines = plt.plot([x,x_nbr], [y,y_nbr])
                    leng += np.linalg.norm([x-x_nbr, y-y_nbr])
        # plt.show()
        print "leng", leng   
        #import IPython
        #IPython.embed()
        

        #plt.imshow(img)
        cond_feats = np.reshape(cond_feats, (num_conds, num_samples, T, 32))
        np.save("3link_feats.npy", cond_feats)
        import IPython
        IPython.embed()
        r = 0
        t = 40
        s = 1
        orig = obs_full[r][1][t]
        ee = ee_full[r][1][t]
        img = np.transpose(obs_full[r][1][t].reshape(3,80,64), [1,2,0])
        feed_dict[self.other['state_inputs'][r]] = obs_full[r][s]
        images = np.transpose(obs_full[r][s].reshape(T,3,80,64), [0,2,3,1]).astype(np.uint8)
        #feed_dict= {}
        #feed_dict[self.other['state_inputs'][r]] = [orig.reshape(3*80*64)]

        output= self.sess.run(self.other['output'][r], feed_dict)
        fp = self.sess.run(self.other['state_features_list'][r], feed_dict)
        #target: 0, 1, 3 4 5 11 14 // not 4,  6 
        # ee: 8? 9? 12 15 // 2 8 not 12, 15
        # block: 9
        # robot 0: 8,12 is not ee, 9 is block, 15 is ee. 0,1,3,5 at least are target
        image = img.astype(np.uint8)
        # ee2 = self.sess.run(self.other['ee_output'][r], feed_dict)[0]
        # image[:,:,0] += 104
        # image[:,:,1]+= 117
        # image[:,:,2]+= 123
        #for i in range(16):
        x = int(fp[i*2]+40)
        y = int(fp[i*2+1]+32)
        print x,y
        image[x,y,:] = 255
        image[max(0, x-1),y,:] = 100
        image[min(79, x+1),y,:] = 100
        image[x,max(y,0),:] = 100
        image[x,min(y+1,63),:] = 100
        out = plt.imshow(image)
        plt.show(out)
        def add_feat(images, fp, i):
            T,_,_,_ = images.shape
            for t in range(T):
                #x = int(fp[t,i*2]*40+40)
                #y = int(fp[t,i*2+1]*32+32)
                x = int(fp[t,i*2]*40+40)
                y = int(fp[t,i*2+1]*32+32)
                print x,y
                images[t,x,y,:] = [250,250,102]#255
                images[t,max(0, x-1),y,:] = [250,250,102]
                images[t,min(79, x+1),y,:] = [250,250,102]
                images[t,x,max(y,0),:] = [250,250,102]
                images[t,x,min(y+1,63),:] = [250,250,102]
            return images

        def ani_frame(images, name):
            import moviepy.editor as mpy

            def make_frame(n):
                tmp = images[n,:,:,:]
                return tmp
            #clip = mpy.VideoClip(make_frame, duration=5)
            clip = mpy.ImageSequenceClip([images[i] for i in range(100)], fps=20)
            clip.write_gif("/home/coline/Desktop/test"+name+".gif",fps=20)
            return clip
        for r in [0,1]:
            for s in [2,9]:
                images_fp = np.transpose(obs_full[r][s].reshape(T,3,80,64), [0,2,3,1]).astype(np.uint8)
                for feat in range(16):
                    feed_dict[self.other['state_inputs'][r]] = obs_full[r][s]
                    fp = self.sess.run(self.other['state_features_list'][r], feed_dict)
                    images_fp = add_feat(images_fp, fp, feat)
                ani_frame(images_fp, "strikeblue"+str(feat)+"_r"+str(r)+"_s"+str(s))
            
        data = np.zeros((14,100, 32))
        for s in range(14):
            r = 0
            images_fp = np.transpose(obs_full[r][s].reshape(T,3,80,64), [0,2,3,1]).astype(np.uint8)
            feed_dict[self.other['state_inputs'][r]] = obs_full[r][s]
            fp = self.sess.run(self.other['state_features_list'][r], feed_dict)
            data[s] = fp
            for feat in range(16):
                images_fp = add_feat(images_fp, fp, feat)
            ani_frame(images_fp, "allfeats_test"+str(feat)+"_r"+str(r)+"_s"+str(s))
            
        for r in range(2):
            for s in [1,10]:
                feed_dict[self.other['state_inputs'][r]] = obs_full[r][s]
                output= self.sess.run(self.other['output'][r], feed_dict)
                ani_frame(output, "strike"+str(r)+"_"+str(s))

        print("done training invariant autoencoder and saving weights")

    def train_invariant_dc(self, obs_full, next_obs_full, action_full, obs_extended_full):
        num_conds, num_samples, T_extended, _ = obs_extended_full[0].shape
        feat_size, act_size, dc_w, lr = self.other['hyperparams']
        #filename = 'test_f'+str(feat_size)+'_a'+str(act_size)+'_w'+str(dc_w)
        filename = 'lr_'+str(lr)
        for cond in range(num_conds):
            for s_no in range(num_samples):
                for robot_number in range(self.num_robots):
                    color = ['r', 'b'][robot_number]
                    x = obs_extended_full[robot_number][cond, s_no, :, 6+2*robot_number]
                    y = obs_extended_full[robot_number][cond, s_no, :, 8+2*robot_number]
                    plt.scatter(x, y, c=color)
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




        dc_loss = 0
        pred_error = 0
        average_loss = 0
        all_losses = np.zeros((len(self.other['all_losses']),))
        gen_loss = np.zeros((len(self.other['gen_loss']),))
        all_losses_dc = np.zeros((len(self.other['dc_loss']),))
        contrastive = 0
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
            #all_losses += self.sess.run(self.other['all_losses'], feed_dict)
            gen_loss += self.sess.run(self.other['gen_loss'], feed_dict)
            contrastive += self.sess.run(self.other['contrastive'], feed_dict)
            #preds = self.other['predictions_full'][0]+ self.other['predictions_full'][1]
            #preds= self.sess.run(preds, feed_dict)
            #predictions_full = [[preds[j] for j in range(len(self.other['predictions_full'][0]))],
            #                    [preds[j+len(self.other['predictions_full'][0])] for j in range(len(self.other['predictions_full'][1]))]]
            #for robot_number in range(self.num_robots):
            #    for pred in predictions_full[robot_number]:
            #        pred_error += np.sum(pred != [robot0_classification, robot1_classification][robot_number])
            average_loss += train_loss
            if i % 200 == 0:# and i != 0:
                print "hi", i
                print 'supervised tensorflow loss is '
                print (average_loss/200)
                #print (all_losses/200)
                print "autoencoder", (gen_loss/200)
                print "contrastive", contrastive/200
                # mlplt.add_generator(i, average_loss / 200)
                mlplt.add_generator_only(i, np.sum(gen_loss) / 200)
                #print("PREDICTION ERROR:")
                #print(pred_error/(2.0*self.batch_size*200))
                mlplt.add_discriminator(i, contrastive/200)
                #pred_error = 0
                print("--------------------------")
                average_loss = 0
                all_losses = np.zeros((len(self.other['all_losses']),))
                gen_loss = np.zeros((len(self.other['gen_loss']),))
                contrastive = 0
            # if i > 1000:
            # dc_feed_dict = {}
            # start_idx = int(i * self.batch_size % (batches_per_epoch*self.batch_size))
            # idx_i = idx[start_idx:start_idx+self.batch_size]
            # for robot_number in range(self.num_robots):
            #     dc_feed_dict[self.other['state_inputs'][robot_number]] = obs_reshaped[robot_number][idx_i]
            #     dc_feed_dict[self.other['next_state_inputs'][robot_number]] = next_obs_reshaped[robot_number][idx_i]
            #     dc_feed_dict[self.other['action_inputs'][robot_number]] = action_reshaped[robot_number][idx_i]
            # dc_loss += self.dc_solver(dc_feed_dict, self.sess, device_string=self.device_string)
            # all_losses_dc += self.sess.run(self.other['dc_loss'], dc_feed_dict)

            # if i % 200 == 0 and i != 0:
            #     print 'supervised dc loss is '
            #     print (dc_loss/200)
            #     print (all_losses_dc/200)
            #     print("--------------------------")
            #     dc_loss = 0
            #     all_losses_dc = np.zeros((len(self.other['dc_loss']),))
        import IPython
        IPython.embed()
        mlplt.plot()
        
        print "DONE WITH TF ITERS"

        # import IPython
        # IPython.embed()
        #mlplt.close()
        var_dict = {}
        for k, v in self.other['all_variables'].items():
            var_dict[k] = self.sess.run(v)
        



        
        cond_feats = np.zeros((num_conds, num_samples, T_extended, feat_size))
        cond_feats_other = np.zeros((num_conds, num_samples, T_extended, feat_size))
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
        # import IPython
        # IPython.embed()
        cond_feats = np.reshape(cond_feats, (num_conds*num_samples*T_extended,feat_size))
        cond_feats_other = np.reshape(cond_feats_other, (num_conds*num_samples*T_extended,feat_size))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cond_feats)
        distances, indices = nbrs.kneighbors(cond_feats_other)
        indices = np.reshape(indices, (num_conds, num_samples, T_extended))
        dO_robot0 = obs_extended_full[0].shape[-1]
        obs_full_reshaped = np.reshape(obs_extended_full[0], (num_conds*num_samples*T_extended,dO_robot0))
        print("CHECK NN")
        # import IPython
        # IPython.embed()
        line_length = 0
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
                    line_length += np.linalg.norm(np.array([x-x_nbr, y-y_nbr]))
        # plt.savefig(filename+"_"+str(line_length)+"_fig.png")
        # pickle.dump(var_dict, open(filename+"_"+str(line_length)+"_subspace_state.pkl", "wb"))
        plt.show()
        import IPython
        IPython.embed()
        cond_feats = np.reshape(cond_feats, (num_conds, num_samples, T_extended, feat_size))
        # np.save(filename+"_"+str(line_length)+"_3link_feats.npy", np.asarray(cond_feats))
        np.save('3link_feats', cond_feats)
        # np.save(filename+"_length_"+str(line_length)+".npy", line_length)
        print("done training invariant DC and saving weights")
        return

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
    def run_one_image_forward(self, img, robot_number):
        feed_dict = {}
        # dO = [len(self._hyperparams['r0_index_list']), len(self._hyperparams['r1_index_list'])][robot_number]
        # dU = self._dU[robot_number]
        feed_dict[self.other['state_inputs'][robot_number]] = [img]
        output = self.sess.run(self.other['state_features_list'][robot_number], feed_dict=feed_dict)[0]
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
