""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier
import tensorflow as tf
def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

class CostDevRsAction(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        #may need to change
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)
        self.init_feature_space()

    def init_feature_space(self):
        """ Helper method to initialize the tf networks used """
        import pickle
        val_vars = pickle.load(open(self._hyperparams['load_file'], 'rb'))
        g = tf.Graph()
        self.graph = g
        n_layers = 4
        layer_size = 20
        dim_hidden = (n_layers - 1)*[layer_size]
        feature_layers = []
        dim_input = 4
        num_feats = 20
        with g.as_default():
            nn_input = tf.placeholder("float", [None, dim_input], name='action_nn_input101')
            w_input = init_weights((dim_input ,dim_hidden[0]), name='aw_input101')
            b_input = init_bias((dim_hidden[0],), name='ab_input101')
            w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='aw1_101' )
            b1 = init_bias((dim_hidden[1],), name='ab1_101')
            w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='aw2_101')
            b2 = init_bias((dim_hidden[2],), name='ab2_101')
            w_output = init_weights((dim_hidden[2], dim_input), name='aw_output101')
            b_output = init_bias((dim_input,), name = 'ab_output101')
            layer0 = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
            layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
            feature_layers = layer1
            layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
            output = tf.matmul(layer2, w_output) + b_output
            gradients = tf.gradients(layer1, nn_input)
            init_op = tf.initialize_local_variables()
            self.feature_layers = feature_layers
            self.gradients = gradients
            self.input = nn_input
            col_sum = tf.reduce_sum(self.feature_layers, 0)
            split_feats = tf.split(0, num_feats, col_sum)
            grad_ops = []
            for j in range(num_feats):
                grad_ops += tf.gradients(split_feats[j], self.input)
            self.grad_ops = grad_ops
        self.session = tf.Session(graph=g)
        self.session.run(init_op)
        with g.as_default():
            self.var_list_feat = {}
            for v in tf.trainable_variables():
                self.var_list_feat[v.name] = v
            for k,v in self.var_list_feat.items():
                if k in val_vars:   
                    print("COST LOAD")
                    print(k)         
                    assign_op = v.assign(val_vars[k])
                    self.session.run(assign_op)
        raw_input('action loaded')

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
        x = sample.get_U()
        feed_dict = {self.input: x}
        feat_forward = self.session.run(self.feature_layers, feed_dict=feed_dict)
        num_feats = feat_forward.shape[1]
        num_inputs = x.shape[1]
        gradients_all = np.zeros((T, num_feats, num_inputs))
        grad_vals = self.session.run(self.grad_ops, feed_dict=feed_dict)
        for j, gv in enumerate(grad_vals):
            gradients_all[:, j, :] = gv
        print("next action")
        size_ls = Du
        l = np.zeros((T,))
        lu = np.zeros((T,size_ls))
        luu = np.zeros((T, size_ls, size_ls))
        for t in range(T):
            l[t] = (feat_forward[t] - tgt[t]).dot(np.eye(20)/(2.0)).dot(feat_forward[t] - tgt[t])
            lu[t,:] = (feat_forward[t] - tgt[t]).dot(gradients_all[t])
            luu[t,:,:] = gradients_all[t].T.dot(gradients_all[t])           
        final_l += l
        final_lu += lu
        final_luu += luu
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
