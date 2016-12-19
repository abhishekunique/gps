""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np
import time
import mjcpy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_MUJOCO
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE

from gps.sample.sample import Sample
import tensorflow as tf

def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        if 'tendon' in hyperparams:
            self.tendon_object = hyperparams['tendon']
        else:
            self.tendon_object = False

        Agent.__init__(self, config)
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])
        self.init_nn()

    def init_nn(self):
        self.sess = tf.Session()
        num_hidden = 4
        layer_size = 60
        dim_hidden = [layer_size]*num_hidden
        dim_hidden_action = [layer_size]*num_hidden

        #defining input placeholders
        state_input_target = tf.placeholder("float", [None,14], name='nn_input_state1')
        action_input_source = tf.placeholder("float", [None,3], name='nn_input_action0')
        self.state_input_target = state_input_target
        self.action_input_source = action_input_source
        #appending into lists
        robot_number = 1
        w0_state = init_weights((14, dim_hidden[0]), name='w0_state' + str(robot_number))
        b0_state = init_bias((dim_hidden[0],), name='b0_state'+str(robot_number))
        w1_state = init_weights((dim_hidden[0], dim_hidden[1]), name='w1_state' + str(robot_number))
        b1_state = init_bias((dim_hidden[1],), name='b1_state' + str(robot_number))
        w2_state = init_weights((dim_hidden[1], dim_hidden[2]), name='w2_state' + str(robot_number))
        b2_state = init_bias((dim_hidden[2],), name='b2_state' + str(robot_number))
        w3_state_ae = init_weights((dim_hidden[2], dim_hidden[3]), name='w3_state_ae' + str(robot_number))
        b3_state_ae = init_bias((dim_hidden[3],), name='b3_state_ae' + str(robot_number))
        w4_state_ae = init_weights((dim_hidden[3], 12), name='w4_state_ae' + str(robot_number))
        b4_state_ae = init_bias((12,), name='b4_state_ae' + str(robot_number))

        ### STATE EMBEDDING ###
        layer0_state = tf.nn.relu(tf.matmul(state_input_target, w0_state) + b0_state)
        layer1_state = tf.nn.relu(tf.matmul(layer0_state, w1_state) + b1_state)
        layer2_state = tf.nn.relu(tf.matmul(layer1_state, w2_state) + b2_state)
        #autoencoding output#
        layer3_state_ae = tf.nn.relu(tf.matmul(layer2_state, w3_state_ae) + b3_state_ae)
        output_state_ae = tf.matmul(layer3_state_ae, w4_state_ae) + b4_state_ae
        self.source_traj = output_state_ae
        ### END STATE EMBEDDING ###

         #DEFINING ACTION VARIABLES

        w0_action = init_weights((3, dim_hidden_action[0]), name='w0_action' + str(robot_number))
        b0_action = init_bias((dim_hidden_action[0],), name='b0_action'+str(robot_number))
        w1_action = init_weights((dim_hidden_action[0], dim_hidden_action[1]), name='w1_action' + str(robot_number))
        b1_action = init_bias((dim_hidden_action[1],), name='b1_action' + str(robot_number))
        w2_action = init_weights((dim_hidden_action[1], dim_hidden_action[2]), name='w2_action' + str(robot_number))
        b2_action = init_bias((dim_hidden_action[2],), name='b2_action' + str(robot_number))

        w3_action = init_weights((dim_hidden_action[2], dim_hidden_action[3]), name='w3_action' + str(robot_number))
        b3_action = init_bias((dim_hidden_action[3],), name='b3_action' + str(robot_number))
        w4_action = init_weights((dim_hidden_action[3], 4), name='w4_action' + str(robot_number))
        b4_action = init_bias((4,), name='b4_action' + str(robot_number))

        ### ACTION EMBEDDING ###
        layer0_action = tf.nn.relu(tf.matmul(action_input_source, w0_action) + b0_action)
        layer1_action = tf.nn.relu(tf.matmul(layer0_action, w1_action) + b1_action)
        layer2_action = tf.nn.relu(tf.matmul(layer1_action, w2_action) + b2_action)
        layer3_action = tf.nn.relu(tf.matmul(layer2_action, w3_action) + b3_action)
        output_action = tf.matmul(layer3_action, w4_action) + b4_action
        self.target_act = output_action
        ### END ACTION EMBEDDING ###
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        import pickle
        val_vars = pickle.load(open('subspace_state.pkl', 'rb'))
        for v in tf.trainable_variables():
            k = v.name[:-2]
            for vvkey in val_vars.keys():
                if k in vvkey:
                    print("LOADED")   
                    print(k)        
                    assign_op = v.assign(val_vars[vvkey])
                    self.sess.run(assign_op)


    def sample_pol(self, policy, condition, verbose=True, save=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
        """
        # Create new sample, populate first time step.
        print("TAKING SAMPLE")
        new_sample = self._init_sample(condition)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        noise = generate_noise(self.T, 3, self._hyperparams)
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        self._world[condition].set_model(self._model[condition])
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            X_t_sliced = np.asarray([np.concatenate([X_t[:4], X_t[5:9], X_t[10:13], X_t[19:22]])])
            X_t_in = self.sess.run(self.source_traj, feed_dict = {self.state_input_target: X_t_sliced})
            X_t_replaced = np.zeros((26,))
            X_t_replaced[:3] = X_t_in[0, :3]
            X_t_replaced[4:7] = X_t_in[0, 3:6]
            X_t_replaced[8:11] = X_t_in[0, 6:9]
            X_t_replaced[17:20] = X_t_in[0, 9:12]

            X_t_replaced[3:4] = X_t[4:5]
            X_t_replaced[7:8] = X_t[9:10]
            X_t_replaced[11:17] = X_t[13:19]
            X_t_replaced[20:26] = X_t[22:28]
            mj_U_out = policy.act(X_t_replaced, None, t, noise[t, :])
            mj_U_out = np.asarray([mj_U_out])
            mj_U = self.sess.run(self.target_act, feed_dict = {self.action_input_source: mj_U_out})[0]
            mj_U = np.asarray(mj_U, dtype=np.float64)
            print(mj_U)
            U[t, :] = mj_U
            if verbose:
                self._world[condition].plot(mj_X)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)
                #TODO: Some hidden state stuff will go here.
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._world = []
        self._model = []
        # import IPython
        # IPython.embed()
        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        # if not isinstance(filename, list):
        #     self._world = mjcpy.MJCWorld(filename)
        #     self._model = self._world.get_model()
        #     self._world = [self._world
        #                    for _ in range(self._hyperparams['conditions'])]
        #     self._model = [copy.deepcopy(self._model)
        #                    for _ in range(self._hyperparams['conditions'])]
        # else:
        for i in range(self._hyperparams['conditions']):
            self._world.append(mjcpy.MJCWorld(self._hyperparams['filename'][i]))
            self._model.append(self._world[i].get_model())

        for i in range(self._hyperparams['conditions']):
            for j in range(len(self._hyperparams['pos_body_idx'][i])):
                idx = self._hyperparams['pos_body_idx'][i][j]
                self._model[i]['body_pos'][idx, :] += \
                        self._hyperparams['pos_body_offset'][i][j]
            self._world[i].set_model(self._model[i])
            x0 = self._hyperparams['x0'][i]
            idx = len(x0) // 2
            data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
            self._world[i].set_data(data)
            self._world[i].kinematics()

        self._joint_idx = list(range(self._model[0]['nq']))
        self._vel_idx = [i + self._model[0]['nq'] for i in self._joint_idx]

        # Initialize x0.
        self.x0 = []
        self.eepts0 = [None]*self._hyperparams['conditions']
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                if self.tendon_object == False:
                    self.eepts0[i] = self._world[i].get_data()['site_xpos'].flatten()
                else:
                    self.eepts0[i] = self._world[i].get_data()['site_xpos'][self.tendon_object].flatten()
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], self.eepts0[i], np.zeros_like(self.eepts0[i])])
                )
            else:
                self.x0.append(self._hyperparams['x0'][i])

        cam_pos = self._hyperparams['camera_pos']
        for i in range(self._hyperparams['conditions']):
            self._world[i].init_viewer(AGENT_MUJOCO['image_width'],
                                       AGENT_MUJOCO['image_height'],
                                       cam_pos[0], cam_pos[1], cam_pos[2],
                                       cam_pos[3], cam_pos[4], cam_pos[5])

    # def demonstrate_reward_shaping(self, condition):
    #     """
    #     Run training by iteratively sampling and taking an iteration.
    #     Args:
    #         itr_load: If specified, loads algorithm state from that
    #             iteration, and resumes training at the next iteration.
    #     Returns: None
    #     """
    #     marker_idx = [-3, -2, -1]
    #     mj_X = self._hyperparams['x0'][condition]
    #     self._world[condition].plot(mj_X)
    #     init_key=None
    #     waypoint_list = []
    #     self._world[condition].plot(mj_X)
    #     temp_data = self._world[condition].get_data()
    #     init_pos = temp_data['site_xpos'][-1]
    #     while (init_key!="q"):
    #         offset = np.zeros((3,))
    #         init_key = raw_input("w/a/s/d for up, left, down, right. r for record point. q for end.")
    #         if init_key == "w":
    #             offset = np.array([0.1, 0, 0])
    #         elif init_key == "a":
    #             offset = np.array([0, 0, -0.1])
    #         elif init_key == "s":
    #             offset = np.array([-0.1, 0, 0])
    #         elif init_key == "d":
    #             offset = np.array([0, 0, 0.1])
    #         init_pos += offset
    #         if init_key == "r":
    #             print("RECORDED")
    #             waypoint_list.append(copy.copy(init_pos))
    #         self._model[condition]['body_pos'][-2, :] += offset
    #         self._world[condition].set_model(self._model[condition])
    #         self._world[condition].plot(mj_X)
    #         time.sleep(0.01)
        
    #     num_waypoints = len(waypoint_list)
    #     waypoint_dist = self.T/num_waypoints
    #     target = np.zeros((self.T, 9))
    #     curr_start = 0
    #     for waypoint in waypoint_list:
    #         target[curr_start:curr_start + waypoint_dist, 0:3] = waypoint
    #         curr_start = curr_start + waypoint_dist
    #     print(waypoint_list)
    #     return target

    def sample(self, policy, condition, verbose=True, save=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
        """
        # Create new sample, populate first time step.
        new_sample = self._init_sample(condition)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        noise = generate_noise(self.T, self.dU, self._hyperparams)
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        self._world[condition].set_model(self._model[condition])
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            print(mj_U.shape)
            U[t, :] = mj_U
            if verbose:
                self._world[condition].plot(mj_X)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)
                #TODO: Some hidden state stuff will go here.
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, condition):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
        """
        sample = Sample(self)
        sample.set(JOINT_ANGLES,
                   self._hyperparams['x0'][condition][self._joint_idx], t=0)
        sample.set(JOINT_VELOCITIES,
                   self._hyperparams['x0'][condition][self._vel_idx], t=0)
        self._data = self._world[condition].get_data()
        eepts = self.eepts0[condition] #self._data['site_xpos'].flatten()
        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)
        if self.tendon_object == False:
            jac = np.zeros([eepts.shape[0], self._model[condition]['nq']])
            for site in range(eepts.shape[0] // 3):
                idx = site * 3
                jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)
        else:
            jac = np.zeros([eepts.shape[0], self._model[condition]['nq']])
            for idx_num,site in enumerate(self.tendon_object):
                idx = idx_num*3
                jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)

        # save initial image to meta data
        self._world[condition].plot(self._hyperparams['x0'][condition])
        img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                      self._hyperparams['image_height'])
        # mjcpy image shape is [height, width, channels],
        # dim-shuffle it for later conv-net processing,
        # and flatten for storage
        img_data = np.transpose(img["img"], (2, 1, 0)).flatten()
        # if initial image is an observation, replicate it for each time step
        if CONTEXT_IMAGE in self.obs_data_types:
            sample.set(CONTEXT_IMAGE, np.tile(img_data, (self.T, 1)), t=None)
        else:
            sample.set(CONTEXT_IMAGE, img_data, t=None)
        sample.set(CONTEXT_IMAGE_SIZE, np.array([self._hyperparams['image_channels'],
                                                self._hyperparams['image_width'],
                                                self._hyperparams['image_height']]), t=None)
        # only save subsequent images if image is part of observation
        if RGB_IMAGE in self.obs_data_types:
            sample.set(RGB_IMAGE, img_data, t=0)
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
        return sample

    def _set_sample(self, sample, mj_X, t, condition):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
        """
        sample.set(JOINT_ANGLES, np.array(mj_X[self._joint_idx]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(mj_X[self._vel_idx]), t=t+1)
        if self.tendon_object == False:
            curr_eepts = self._data['site_xpos'].flatten()
        else:
            curr_eepts = self._data['site_xpos'][self.tendon_object].flatten()
        sample.set(END_EFFECTOR_POINTS, curr_eepts, t=t+1)
        prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
        eept_vels = (curr_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)
        if self.tendon_object == False:
            jac = np.zeros([curr_eepts.shape[0], self._model[condition]['nq']])
            for site in range(curr_eepts.shape[0] // 3):
                idx = site * 3
                jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)
        else:
            jac = np.zeros([curr_eepts.shape[0], self._model[condition]['nq']])
            for idx_num,site in enumerate(self.tendon_object):
                idx = idx_num*3
                jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t+1)
        if RGB_IMAGE in self.obs_data_types:
            img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                          self._hyperparams['image_height'])
            sample.set(RGB_IMAGE, np.transpose(img["img"], (2, 1, 0)).flatten(), t=t+1)

    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        img = obs[imstart:imend]
        img = img.reshape((image_channels, image_width, image_height))
        img = np.transpose(img, [1, 2, 0])
        return img
