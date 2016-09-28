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

class CostDevRs3D(Cost):
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
        n_layers = 5
        layer_size = 60
        dim_hidden = (n_layers - 1)*[layer_size]
        feature_layers = []
        dim_input = 22
        num_feats = 60
        with g.as_default():
            nn_input = tf.placeholder("float", [None, dim_input], name='nn_input1')
            w_input = init_weights((dim_input ,dim_hidden[0]), name='w_input1')
            b_input = init_bias((dim_hidden[0],), name='b_input1')
            w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='w1_1' )
            b1 = init_bias((dim_hidden[1],), name='b1_1')
            w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='w2_1')
            b2 = init_bias((dim_hidden[2],), name='b2_1')
            w3 = init_weights((dim_hidden[2], dim_hidden[3]), name='w3_1')
            b3 = init_bias((dim_hidden[3],), name='b3_1')
            w_output = init_weights((dim_hidden[3], dim_input), name='w_output1')
            b_output = init_bias((dim_input,), name = 'b_output1')
            layer0 = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
            layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
            feature_layers = layer2
            layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
            output = tf.matmul(layer3, w_output) + b_output
            gradients = tf.gradients(layer2, nn_input)
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
        x = np.concatenate([x[:, 0:5], x[:, 7:12], x[:, 14:20], x[:, 32:38]], axis=1)
        feed_dict = {self.input: x}
        feat_forward = self.session.run(self.feature_layers, feed_dict=feed_dict)
        num_feats = feat_forward.shape[1]
        num_inputs = x.shape[1]
        gradients_all = np.zeros((T, num_feats, num_inputs))
        grad_vals = self.session.run(self.grad_ops, feed_dict=feed_dict)
        for j, gv in enumerate(grad_vals):
            gradients_all[:, j, :] = gv
        print("next")
        size_ls = 50
        l = np.zeros((T,))
        ls = np.zeros((T,size_ls))
        lss = np.zeros((T, size_ls, size_ls))
        for t in range(T):
            l[t] = (feat_forward[t] - tgt[t]).dot(np.eye(60)/(2.0)).dot(feat_forward[t] - tgt[t])
            grad_mult = (feat_forward[t] - tgt[t]).dot(gradients_all[t])

            ls[t, 0:5] = grad_mult[0:5]
            ls[t, 7:12] = grad_mult[5:10]
            ls[t, 14:20] = grad_mult[10:16]
            ls[t, 32:38] = grad_mult[16:22]
            hess_mult = gradients_all[t].T.dot(gradients_all[t])

            lss[t,0:5,0:5] = hess_mult[0:5, 0:5]
            lss[t,7:12,0:5] = hess_mult[5:10, 0:5]
            lss[t,14:20,0:5] = hess_mult[10:16, 0:5]
            lss[t,32:38,0:5] = hess_mult[16:22, 0:5]

            lss[t,0:5,7:12] = hess_mult[0:5, 5:10]
            lss[t,7:12,7:12] = hess_mult[5:10, 5:10]
            lss[t,14:20,7:12] = hess_mult[10:16, 5:10]
            lss[t,32:38,7:12] = hess_mult[16:22, 5:10]

            lss[t,0:5,14:20] = hess_mult[0:5, 10:16]
            lss[t,7:12,14:20] = hess_mult[5:10, 10:16]
            lss[t,14:20,14:20] = hess_mult[10:16, 10:16]
            lss[t,32:38,14:20] = hess_mult[16:22, 10:16]

            lss[t,0:5,32:38] = hess_mult[0:5, 16:22]
            lss[t,7:12,32:38] = hess_mult[5:10, 16:22]
            lss[t,14:20,32:38] = hess_mult[10:16, 16:22]
            lss[t,32:38,32:38] = hess_mult[16:22, 16:22]

        final_l += l
        final_lx += ls
        final_lxx += lss
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux