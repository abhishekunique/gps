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
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, TENDON_LENGTHS, TENDON_VELOCITIES

from gps.sample.sample import Sample


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

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        if not isinstance(filename, list):
            self._world = mjcpy.MJCWorld(filename)
            self._model = self._world.get_model()
            self._world = [self._world
                           for _ in range(self._hyperparams['conditions'])]
            self._model = [copy.deepcopy(self._model)
                           for _ in range(self._hyperparams['conditions'])]
        else:
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
        # if self.tendon_object != False:
        self._tenlength_idx = [self._model[0]['nq'] + self._model[0]['nv'] + i for i in range(self._model[0]['ntendon'])]
        self._tenvelocity_idx = [i + self._model[0]['nq'] + self._model[0]['nv'] + self._model[0]['ntendon'] for i in range(self._model[0]['ntendon'])]

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
        x0 = self._hyperparams['x0'][condition]
        idx = len(x0) // 2
        data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
        self._world[condition].set_data(data)
        self._world[condition].kinematics()
        self._world[condition].set_model(self._model[condition])
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            if verbose:
                self._world[condition].plot(mj_X)
                time.sleep(0.01)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    # import IPython
                    # IPython.embed()
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)
                    # print(mj_X)

                #TODO: Some hidden state stuff will go here.
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition)
            # raw_input('next')
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
        sample.set(JOINT_ANGLES, self._hyperparams['x0'][condition][self._joint_idx], t=0)
        sample.set(JOINT_VELOCITIES, self._hyperparams['x0'][condition][self._vel_idx], t=0)
        sample.set(TENDON_LENGTHS, self._hyperparams['x0'][condition][self._tenlength_idx], t=0)
        sample.set(TENDON_VELOCITIES, self._hyperparams['x0'][condition][self._tenvelocity_idx], t=0)
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
        # img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
        #                                               self._hyperparams['image_height'])
        # # mjcpy image shape is [height, width, channels],
        # # dim-shuffle it for later conv-net processing,
        # # and flatten for storage
        # img_data = np.transpose(img["img"], (2, 1, 0)).flatten()
        # # if initial image is an observation, replicate it for each time step
        # if CONTEXT_IMAGE in self.obs_data_types:
        #     sample.set(CONTEXT_IMAGE, np.tile(img_data, (self.T, 1)), t=None)
        # else:
        #     sample.set(CONTEXT_IMAGE, img_data, t=None)
        # sample.set(CONTEXT_IMAGE_SIZE, np.array([self._hyperparams['image_channels'],
        #                                         self._hyperparams['image_width'],
        #                                         self._hyperparams['image_height']]), t=None)
        # # only save subsequent images if image is part of observation
        # if RGB_IMAGE in self.obs_data_types:
        #     sample.set(RGB_IMAGE, img_data, t=0)
        #     sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
        #                                 self._hyperparams['image_width'],
        #                                 self._hyperparams['image_height']], t=None)
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
        # import IPython
        # IPython.embed()
        sample.set(TENDON_LENGTHS, np.array(mj_X[self._tenlength_idx]), t=t+1)
        sample.set(TENDON_VELOCITIES, np.array(mj_X[self._tenvelocity_idx]), t=t+1)
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
        # if RGB_IMAGE in self.obs_data_types:
        #     img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
        #                                                   self._hyperparams['image_height'])
        #     sample.set(RGB_IMAGE, np.transpose(img["img"], (2, 1, 0)).flatten(), t=t+1)

    # def _get_image_from_obs(self, obs):
    #     imstart = 0
    #     imend = 0
    #     image_channels = self._hyperparams['image_channels']
    #     image_width = self._hyperparams['image_width']
    #     image_height = self._hyperparams['image_height']
    #     for sensor in self._hyperparams['obs_include']:
    #         # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
    #         if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
    #             imend = imstart + self._hyperparams['sensor_dims'][sensor]
    #             break
    #         else:
    #             imstart += self._hyperparams['sensor_dims'][sensor]
    #     img = obs[imstart:imend]
    #     img = img.reshape((image_channels, image_width, image_height))
    #     img = np.transpose(img, [1, 2, 0])
    #     return img
