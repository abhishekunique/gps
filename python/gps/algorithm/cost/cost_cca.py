""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier
import tensorflow as tf
import pickle
def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

class CostCCA(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        #may need to change
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)
        self.init_feature_space()

    def init_feature_space(self):
        """ Helper method to initialize the tf networks used """
        with open('multiproxy_cca.pkl', 'rb') as f:
            self.cca = pickle.load(f)
            self.x_weights = self.cca.X_.T.dot(self.cca.alphas_)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        tgt = self._hyperparams['target_feats']
        x = sample.get_obs()
        # x = np.concatenate([x[:, 0:4], x[:, 5:9], x[:, 10:13], x[:, 19:22]], axis=1)
        x = np.concatenate([x[:, 0:4], x[:, 5:9]], axis=1)
        feat_forward = self.cca.transform(x)
        num_feats = feat_forward.shape[1]
        num_inputs = x.shape[1]
        
        size_ls = 28
        l = np.zeros((T,))
        ls = np.zeros((T,size_ls))
        lss = np.zeros((T, size_ls, size_ls))
        for t in range(T):
            l[t] = (feat_forward[t] - tgt[t]).dot(np.eye(6)/(2.0)).dot(feat_forward[t] - tgt[t])
            grad_mult = (feat_forward[t] - tgt[t]).dot(self.x_weights.T)

            ls[t, 0:4] = grad_mult[0:4]
            ls[t, 5:9] = grad_mult[4:8]
            # ls[t, 10:13] = grad_mult[8:11]
            # ls[t, 19:22] = grad_mult[11:14]
            # hess_mult = gradients_all[t].T.dot(gradients_all[t])

            # lss[t,0:4,0:4] = hess_mult[0:4, 0:4]
            # lss[t,5:9,0:4] = hess_mult[4:8, 0:4]
            # # lss[t,10:13,0:4] = hess_mult[8:11, 0:4]
            # # lss[t,19:22,0:4] = hess_mult[11:14, 0:4]

            # lss[t,0:4,5:9] = hess_mult[0:4, 4:8]
            # lss[t,5:9,5:9] = hess_mult[4:8, 4:8]
            # # lss[t,10:13,5:9] = hess_mult[8:11, 4:8]
            # # lss[t,19:22,5:9] = hess_mult[11:14, 4:8]

            # # lss[t,0:4,10:13] = hess_mult[0:4, 8:11]
            # # lss[t,5:9,10:13] = hess_mult[4:8, 8:11]
            # lss[t,10:13,10:13] = hess_mult[8:11, 8:11]
            # lss[t,19:22,10:13] = hess_mult[11:14, 8:11]

            # lss[t,0:4,19:22] = hess_mult[0:4, 11:14]
            # lss[t,5:9,19:22] = hess_mult[4:8, 11:14]
            # lss[t,10:13,19:22] = hess_mult[8:11, 11:14]
            # lss[t,19:22,19:22] = hess_mult[11:14, 11:14]

        final_l += l
        final_lx += ls
        final_lxx += lss
        # print "cca", np.sum(final_l), final_lx, final_l.shape, final_lx.shape
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux


    @classmethod
    def tf_loss(cls, hyperparams, T, x, u_input, jx_input, ee_input):
        x = tf.concat(1, [x[:, 0:4], x[:, 5:9]])
        with open('multiproxy_cca.pkl', 'rb') as f:
            cca = pickle.load(f)
        tgt = hyperparams['target_feats']
        return tf.reduce_sum((tgt - cca.transform_tf(x)) ** 2, 1) / 2, False
