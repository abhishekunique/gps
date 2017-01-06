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

class CostDevRs(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        #may need to change
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)
        self.init_feature_space()

    def init_feature_space(self):
        """ Helper method to initialize the tf networks used """
        
        val_vars = pickle.load(open(self._hyperparams['load_file'], 'rb'))
        g = tf.Graph()
        self.graph = g
        n_layers = 5
        layer_size = 30
        dim_hidden = (n_layers - 1)*[layer_size]
        feature_layers = []
        dim_input = 8#14
        num_feats = 30
        with g.as_default():
            nn_input = tf.placeholder("float", [None, dim_input], name='nn_input_state1')
            w0 = init_weights((dim_input ,dim_hidden[0]), name='w0_state1')
            b0 = init_bias((dim_hidden[0],), name='b0_state1')
            w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='w1_state1' )
            b1 = init_bias((dim_hidden[1],), name='b1_state1')
            w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='w2_state1')
            b2 = init_bias((dim_hidden[2],), name='b2_state1')

            layer0 = tf.nn.relu(tf.matmul(nn_input, w0) + b0)
            layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
            layer2 = tf.matmul(layer1, w2) + b2
            feature_layers = layer2
            gradients = tf.gradients(layer2, nn_input)
            init_op = tf.initialize_all_variables()
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

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        val_vars = pickle.load(open(self._hyperparams['load_file'], 'rb'))
        with self.graph.as_default():
            self.var_list_feat = {}
            for v in tf.trainable_variables():
                self.var_list_feat[v.name] = v
            for k,v in self.var_list_feat.items():
                if k in val_vars:       
                    assign_op = v.assign(val_vars[k])
                    self.session.run(assign_op)

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
        feed_dict = {self.input: x}
        feat_forward = self.session.run(self.feature_layers, feed_dict=feed_dict)
        num_feats = feat_forward.shape[1]
        num_inputs = x.shape[1]
        gradients_all = np.zeros((T, num_feats, num_inputs))
        grad_vals = self.session.run(self.grad_ops, feed_dict=feed_dict)
        for j, gv in enumerate(grad_vals):
            gradients_all[:, j, :] = gv
        print("next")
        size_ls = 28
        l = np.zeros((T,))
        ls = np.zeros((T,size_ls))
        lss = np.zeros((T, size_ls, size_ls))
        for t in range(T):
            l[t] = (feat_forward[t] - tgt[t]).dot(np.eye(30)/(2.0)).dot(feat_forward[t] - tgt[t])
            grad_mult = (feat_forward[t] - tgt[t]).dot(gradients_all[t])

            ls[t, 0:4] = grad_mult[0:4]
            ls[t, 5:9] = grad_mult[4:8]
            # ls[t, 10:13] = grad_mult[8:11]
            # ls[t, 19:22] = grad_mult[11:14]
            hess_mult = gradients_all[t].T.dot(gradients_all[t])

            lss[t,0:4,0:4] = hess_mult[0:4, 0:4]
            lss[t,5:9,0:4] = hess_mult[4:8, 0:4]
            # lss[t,10:13,0:4] = hess_mult[8:11, 0:4]
            # lss[t,19:22,0:4] = hess_mult[11:14, 0:4]

            lss[t,0:4,5:9] = hess_mult[0:4, 4:8]
            lss[t,5:9,5:9] = hess_mult[4:8, 4:8]
            # lss[t,10:13,5:9] = hess_mult[8:11, 4:8]
            # lss[t,19:22,5:9] = hess_mult[11:14, 4:8]

            # lss[t,0:4,10:13] = hess_mult[0:4, 8:11]
            # lss[t,5:9,10:13] = hess_mult[4:8, 8:11]
            # lss[t,10:13,10:13] = hess_mult[8:11, 8:11]
            # lss[t,19:22,10:13] = hess_mult[11:14, 8:11]

            # lss[t,0:4,19:22] = hess_mult[0:4, 11:14]
            # lss[t,5:9,19:22] = hess_mult[4:8, 11:14]
            # lss[t,10:13,19:22] = hess_mult[8:11, 11:14]
            # lss[t,19:22,19:22] = hess_mult[11:14, 11:14]

        final_l += l
        final_lx += ls
        final_lxx += lss
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
