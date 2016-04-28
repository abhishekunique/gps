import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np

def get_xavier_weights(filter_shape, poolsize=(2, 2), name=None):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32), name=name)


def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)


def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.batch_matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def euclidean_loss_layer(a, b, precision, batch_size):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    scale_factor = tf.constant(2*batch_size, dtype='float')
    uP = batched_matrix_vector_multiply(a-b, precision)

    uPu = tf.reduce_sum(uP*(a-b))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor


def get_input_layer(dim_input, dim_output):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, dim_input[0], dim_input[1], dim_input[2]], name='nn_input')
    pose_label = tf.placeholder("float", [None, dim_output], name='pose_label')
    return net_input, pose_label

def conv2d(img, w, b):
    #print img.get_shape().dims[3].value
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def get_mlp_layers(mlp_input, number_layers, dimension_hidden, robot_number):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step) + 'rn' + str(robot_number))
        weights.append(cur_weight)
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b_' + str(layer_step) + 'rn' + str(robot_number))
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top, weights, biases


def convolutional_network(dim_input=(64,80,3), dim_output=3, batch_size=128, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input:  (h,w,c) tuple for input size.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    num_channels = dim_input[2]
    im_width = dim_input[1]
    im_height = dim_input[0]
    n_layers = 3
    layer_size = 20
    dim_hidden = 16
    pool_size = 2
    filter_size = 3
    nn_input, pose_labels = get_input_layer(dim_input, dim_output)
    # image goes through 2 convnet layers
    num_filters = [10,20]
    # Store layers weight & bias
    weights = {
        'wc1': get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rn'), # 5x5 conv, 1 input, 32 outputs
        'pretrain_w': init_weights([num_filters[1]*2, dim_hidden], name='pretrain_w'),
        'pretrain_out': init_weights([dim_hidden, 3], name='pretrain_out'),
    }
    biases = {
        'bc1': init_bias([num_filters[0]], name="biasconv1rn" ),
        'pretrain_b': init_bias([dim_hidden], name='pretrain_b'),
        'pretrain_outb': init_bias([3], name='pretrain_outb'),
    }
    weights['wc2'] = get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rnshared') # 5x5 conv, 32 inputs, 64 outputs
    biases['bc2'] = init_bias([num_filters[1]], name='bc2rnshared')
    conv_layer_0 = conv2d(img=nn_input, w=weights['wc1'], b=biases['bc1'])

    conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

    full_y = np.tile(np.arange(im_width), (im_height,1))
    full_x = np.tile(np.arange(im_height), (im_width,1)).T
    full_x = tf.convert_to_tensor(np.reshape(full_x, [-1,1]), dtype=tf.float32)
    full_y = tf.convert_to_tensor(np.reshape(full_y, [-1,1] ), dtype=tf.float32)
    feature_points = []
    for filter_number in range(num_filters[1]):
        conv_filter_chosen = conv_layer_1[:,:,:,filter_number]
        conv_filter_chosen = tf.reshape(conv_filter_chosen, [-1, im_width*im_height])
        conv_softmax = tf.nn.softmax(conv_filter_chosen)
        feature_points_x = tf.matmul(conv_softmax, full_x)
        feature_points_y = tf.matmul(conv_softmax, full_y)
        feature_points.append(feature_points_x)
        feature_points.append(feature_points_y)
    full_feature_points = tf.concat(concat_dim=1, values=feature_points)
    pretrain_h = tf.nn.relu(tf.matmul(full_feature_points, weights['pretrain_w']) + biases['pretrain_b'])
    pretrain_output = tf.matmul(pretrain_h, weights['pretrain_out']) + biases['pretrain_outb']
    pretrain_loss = tf.nn.l2_loss(pretrain_output - pose_labels)
    out_dict = {'input': nn_input, 'pose_labels': pose_labels, 'loss': pretrain_loss}

    return out_dict
