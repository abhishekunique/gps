""" This file defines an agent for the MuJoCo simulator environment. """
import numpy as np

from gps.agent.agent_utils import setup

from gps.sample.sample import Sample
from gps.agent.agent import Agent
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINTS, ACTION

class AgentRecorded(Agent):

    nan_flag = False

    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        self.setup_x0()
        self._setup_conditions()

    def setup_x0(self):
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            eepts = np.zeros(self._hyperparams['sensor_dims'][END_EFFECTOR_POINTS])
            self.x0.append(
                np.concatenate([self._hyperparams['x0'], eepts, eepts])
            )
    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def sample(self, policy, condition, verbose=True, save=True, noisy=True, index=None):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        assert index is not None
        assert 0 <= index < self._hyperparams['number_samples']
        path_stem = self._hyperparams['real_obs_path']
        full_index = condition * self._hyperparams['number_samples'] + index
        x_vals = np.load("{path_stem}_X.npy".format(path_stem=path_stem))[full_index]
        obs_vals = np.load("{path_stem}_obs.npy".format(path_stem=path_stem))[full_index]
        act_vals = np.load("{path_stem}_act.npy".format(path_stem=path_stem))[full_index]
        # Create new sample, populate first time step.
        new_sample = Sample(self)
        new_sample.set(ACTION, act_vals)
        new_sample._X = x_vals
        new_sample._obs = obs_vals
        assert new_sample.get_obs().size != 0
        if save:
            self._samples[condition].append(new_sample)
        return new_sample
