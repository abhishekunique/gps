""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier
import tensorflow as tf

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
        g = tf.Graph()
        n_layers = 4
        layer_size = 60
        dim_hidden = (n_layers - 1)*[layer_size]
        feature_layers = []
        dim_input = 20
        nn_input = tf.placeholder("float", [None, dim_input], name='nn_input1')
        w_input = init_weights((dim_input ,dim_hidden[0]), name='w_input1')
        b_input = init_bias((dim_hidden[0],), name='b_input1')
        w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='w1_1' )
        b1 = init_bias((dim_hidden[1],), name='b1_1')
        w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='w2_1')
        b2 = init_bias((dim_hidden[2],), name='b2_1')
        w_output = init_weights((dim_hidden[2], dim_input), name='w_output1')
        b_output = init_bias((dim_input,), name = 'b_output1')
        layer0 = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
        layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        feature_layers = layer2
        output = tf.matmul(layer2, w_output) + b_output
        gradients = tf.gradients(neg_cos_angle, nn_input)
        self.feature_layers = feature_layers
        self.gradients = gradients
        self.input = nn_input
        self.session = tf.Session(graph=g)
        init_op = tf.initialize_all_variables()
        self.session.run(init_op)
        import pickle
        val_vars = pickle.load(open('/home/abhigupta/gps/subspace_weights.pkl', 'rb'))
        for k,v in self.var_list_feat.items():
            if k in val_vars:   
                print(k)         
                assign_op = v.assign(val_vars[k])
                self.sess.run(assign_op)

    def eval(self, sample, policy_opt):
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

        for data_type in self._hyperparams['data_types']:
            config = self._hyperparams['data_types'][data_type]
            tgt = config['target_feats']
            x = sample.get_obs()
            feed_dict = {self.input: x}
            feat_forward = self.session.run(self.feature_layers, feed_dict=feed_dict)
            grads = self.session.run(self.gradients, feed_dict=feed_dict)
            size_ls = 20
            l = np.zeros((T,))
            ls = np.zeros((T,size_ls))
            lss = np.zeros((T, size_ls, size_ls))
            for t in range(T):
                l[t] = (feat_forward[t] - tgt[t]).dot(np.eye(size_ls)/(2.0)).dot(feat_forward[t] - tgt[t])
                grad_l = (feat_forward[t] - tgt[t]).dot(grads)
                ls[t,:] = grad_l
                lss[t,:,:] = grad_l.T.dot(grad_l)
            final_l += l
            sample.agent.pack_data_x(final_lx, ls, data_types=[data_type])
            sample.agent.pack_data_x(final_lxx, lss,
                                     data_types=[data_type, data_type])
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
