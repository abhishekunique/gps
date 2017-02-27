""" This file defines the state target cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION, IMAGE_FEATURES

import tensorflow as tf
import pickle
def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

def conv2d(img, w, b):
    #print img.get_shape().dims[3].value
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))
def get_xavier_weights(filter_shape, poolsize=(2, 2), name=None):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32), name=name)

def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

class CostImageFeatures(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        #may need to change
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config)
        self.robot_number = config['robot_number']
        #self.init_feature_space()
        self.count = 0
        self.cond = config['cond']

    def init_feature_space(self):
        """ Helper method to initialize the tf networks used """
        #val_vars = pickle.load(open(self._hyperparams['load_file'], 'rb'))
        # g = tf.Graph()
        #self.graph = g
        n_convlayers = 3
        pool_size = 2
        filter_size = 5
        im_height = 64; im_width = 80; num_channels = 3;
        num_feats = 32
        num_filters = [16,16, 16]
        conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
        num_feats = 32
        robot_number = self.robot_number
        self.num_feats = num_feats
        # with g.as_default():
            
        #     weights = {
        #         'wc1': get_xavier_weights([filter_size, filter_size, 3, num_filters[0]], (pool_size, pool_size), name='wc1'), # 5x5 conv, 3 input, 32 outputs
        #         'wc2': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2'), # 5x5 conv, 32 inputs, 64 outputs
        #         'wc3': get_xavier_weights([filter_size, filter_size, num_filters[1], num_filters[2]], (pool_size, pool_size), name='wc3'),
        #         # 'wc4': get_xavier_weights([filter_size, filter_size, num_filters[1], num_filters[2]], (pool_size, pool_size), name='wc4'),
        #         # 'wc5': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc5'),
        #         # 'wc6': get_xavier_weights([filter_size, filter_size,  num_channels,num_filters[0]], (pool_size, pool_size), name='wc5'),
        #     }

        #     biases = {
        #         'bc1': init_bias([num_filters[0]], name='bc1'),
        #         'bc2': init_bias([num_filters[1]], name='bc2'),
        #         'bc3': init_bias([num_filters[2]], name='bc3'),
        #         # 'bc4': init_bias([num_filters[1]], name='bc4'),
        #         # 'bc5': init_bias([num_filters[0]], name='bc5'),
        #         # 'bc6': init_bias([num_channels], name='bc6'),
        #     }
        #     state_input = tf.placeholder("float", [None, num_channels*im_height*im_width], name='nn_input_state' + str(robot_number))
        #     image_input = tf.reshape(state_input, [-1, num_channels, im_width, im_height])
        #     image_input = tf.transpose(image_input, perm=[0,2,3,1])
        #     #appending into lists
        #     batch = tf.shape(state_input)[0]
        #     ### STATE EMBEDDING ###
        #     conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'])
        #     conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])
        #     conv_layer_2 = conv2d(img=conv_layer_1, w=weights['wc3'], b=biases['bc3'])

        #     _, num_rows, num_cols, num_fp = conv_layer_2.get_shape()
        #     num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
        #     x_map = np.empty([num_rows, num_cols], np.float32)
        #     y_map = np.empty([num_rows, num_cols], np.float32)
        #     for i in range(num_rows):
        #         for j in range(num_cols):
        #             x_map[i, j] = (i - num_rows / 2.0)# / num_rows
        #             y_map[i, j] = (j - num_cols / 2.0)# / num_cols
        #     x_map = tf.convert_to_tensor(x_map)
        #     y_map = tf.convert_to_tensor(y_map)

        #     x_map = tf.reshape(x_map, [num_rows * num_cols])
        #     y_map = tf.reshape(y_map, [num_rows * num_cols])

        #     # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
        #     features = tf.reshape(tf.transpose(conv_layer_2, [0,3,1,2]),
        #                           [-1, num_rows*num_cols])

        #     softmax = tf.nn.softmax(features)
        #     fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
        #     fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)
        #     fp = tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])
        #     init_op = tf.initialize_all_variables()
        #     self.feature_layers = fp
        #     self.input = state_input
        #     # col_sum = tf.reduce_sum(self.feature_layers, 0)
        #     # split_feats = tf.split(0, num_feats, col_sum)
        #     # grad_ops = []
        #     # for j in range(num_feats):
        #     #     grad_ops += tf.gradients(split_feats[j], self.input)
        #     # self.grad_ops = grad_ops
        # self.session = tf.Session(graph=g)
        # self.session.run(init_op)
        # print "val vars",val_vars.keys()
        # with g.as_default():
        #     self.var_list_feat = {}
        #     for v in tf.trainable_variables():
        #         self.var_list_feat[v.name] = v
        #     # import IPython
        #     # IPython.embed()
        #     for k,v in self.var_list_feat.items():
        #         #print k
        #         for k2 in val_vars.keys():
        #             if k2 in k:
        #                 print("COST LOAD")
        #                 print(k)
        #                 assign_op = v.assign(val_vars[k2])
        #                 self.session.run(assign_op)
    def animate(self, images, name):
        import moviepy.editor as mpy
        
        def make_frame(n):
            tmp = images[n,:,:,:]
            return tmp
        #clip = mpy.VideoClip(make_frame, duration=5)
        clip = mpy.ImageSequenceClip([images[i] for i in range(100)], fps=20)
        clip.write_gif("/home/coline/Desktop/samples_strike_test/sample"+name+".gif",fps=20)
        return clip
    def add_feat(self,images, fp, i):
        T,_,_,_ = images.shape
        for t in range(T):
            x = int(fp[t,i*2]*40+40)
            y = int(fp[t,i*2+1]*32+32)
            images[t,x,y,:] = [250,250,102]#255
            images[t,max(0, x-1),y,:] = [250,250,102]
            images[t,min(79, x+1),y,:] = [250,250,102]
            images[t,x,max(y,0),:] = [250,250,102]
            images[t,x,min(y+1,63),:] = [250,250,102]
        return images
    def add_target_feat(self,images, fp, i):
        T,_,_,_ = images.shape
        for t in range(T):
            x = int(fp[t,i*2]*40+40)
            y = int(fp[t,i*2+1]*32+32)
            images[t,x,y,:] = [250,102,250]#255
            images[t,max(0, x-1),y,:] = [250,102,250]
            # images[t,min(79, x+1),y,:] = [250,102,250]
            images[t,x,max(y,0),:] = [250,102,250]
            # images[t,x,min(y+1,63),:] = [250,102,250]
        return images

    def make_clip(self, images, features, tgt):
        for feat in range(features.shape[1]/2):
            images = self.add_feat(images, features, feat)
            images = self.add_target_feat(images, tgt, feat)

        name = "r"+str(self.robot_number)+"c"+str(self.cond)+"s"+str(self.count)
        #self.count+=1
        self.animate(images, name)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        # val_vars = pickle.load(open(self._hyperparams['load_file'], 'rb'))
        # with self.graph.as_default():
        #     self.var_list_feat = {}
        #     for v in tf.trainable_variables():
        #         self.var_list_feat[v.name] = v
        #     for k,v in self.var_list_feat.items():
        #         if k in val_vars:       
        #             assign_op = v.assign(val_vars[k])
        #             self.session.run(assign_op)
        feat_idx = self._hyperparams['feat_idx']
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
        imgs = sample.get_obs()
        imgs = np.transpose(imgs.reshape(T,3,80,64), [0,2,3,1]).astype(np.uint8)
        x = sample.get_X()
        #feed_dict = {self.input: imgs}
        feat_forward = sample.get(IMAGE_FEATURES)#self.session.run(self.feature_layers, feed_dict=feed_dict)
        num_feats = feat_forward.shape[1]
        for t in range(T):
            # final_l[t] = (feat_forward[t] - tgt[t]).dot(np.eye(num_feats)/(2.0)).dot(feat_forward[t] - tgt[t])
            final_l[t] = np.sum(np.square(feat_forward[t] - tgt[t]))/2

        final_lx[:,feat_idx] = feat_forward- tgt
        if (self.count % 15) ==0:
           self.make_clip(imgs, feat_forward, tgt)
        print "mean cost", np.mean(final_l)
        self.count+=1
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
