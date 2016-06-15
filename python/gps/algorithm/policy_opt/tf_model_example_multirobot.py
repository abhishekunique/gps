""" This file provides an example tensorflow network used to define a policy. """

import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np


def init_weights_shared(shape, name=None, stddev=None):
    if stddev is None:
        stddev = 0.01
    weights = tf.get_variable("weights" + str(name), shape,
        initializer=tf.random_normal_initializer(stddev=stddev))
    return weights

def get_xavier_weights_shared(filter_shape, poolsize=(2, 2), name=None):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))
    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    wts = tf.get_variable("xavier_weights" + str(name), filter_shape,
        initializer=tf.random_uniform_initializer(low, high, dtype=tf.float32))
    return wts

def get_xavier_weights(filter_shape, poolsize=(2, 2), name=None):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32), name=name)

def init_bias_shared(shape, name=None):
    biases = tf.get_variable("biases" + str(name), shape,
        initializer=tf.constant_initializer(0.0))
    return biases

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


def get_input_layer(dim_input, dim_output, robot_number, num_robots=None):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input' + str(robot_number))
    action = tf.placeholder('float', [None, dim_output], name='action' + str(robot_number))
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision' + str(robot_number))   
    return net_input, action, precision

def get_mlp_layers(mlp_input, number_layers, dimension_hidden, robot_number):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    layers = []
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
        layers.append(cur_top)

    return cur_top, weights, biases, layers

def multi_input_multi_output_images_shared_conv2(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network shared conv2'
    
    fc_vars = []
    last_conv_vars = []
    num_robots = len(dim_input)
    nnets = []
    st_idx = []
    im_idx = []
    i = []
    for robot_number in range(num_robots):
        st_idx.append([])
        im_idx.append([])
        i.append(0)
    #need to fix whatever this is 
    with tf.variable_scope("shared_wts"):
        for robot_number, robot_params in enumerate(network_config):
            n_layers = 3
            layer_size = 20
            dim_hidden = (n_layers - 1)*[layer_size]
            dim_hidden.append(dim_output[robot_number])
            pool_size = 2
            filter_size = 3
            for sensor in robot_params['obs_include']:
                dim = robot_params['sensor_dims'][sensor]
                if sensor in robot_params['obs_image_data']:
                    im_idx[robot_number] = im_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                else:
                    st_idx[robot_number] = st_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                i[robot_number] += dim

            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

            state_input = nn_input[:, 0:st_idx[robot_number][-1]+1]
            image_input = nn_input[:, st_idx[robot_number][-1]+1:im_idx[robot_number][-1]+1]

            # image goes through 2 convnet layers
            num_filters = network_config[robot_number]['num_filters']

            im_height = network_config[robot_number]['image_height']
            im_width = network_config[robot_number]['image_width']
            num_channels = network_config[robot_number]['image_channels']
            image_input = tf.reshape(image_input, [-1, num_channels, im_width, im_height])
            image_input = tf.transpose(image_input, perm=[0,3,2,1])

                # Store layers weight & bias
            weights = {
                'wc1': get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rn' + str(robot_number)), # 5x5 conv, 1 input, 32 outputs
                # 'wc2': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rn' + str(robot_number)), # 5x5 conv, 1 input, 32 outputs

            }

            biases = {
                'bc1': init_bias([num_filters[0]], name='bc1rn' + str(robot_number)),
                # 'bc2': init_bias([num_filters[1]], name='bc2rn' + str(robot_number)),
            }
            # weights['wc1'] = get_xavier_weights_shared([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rnshared')
            # biases['bc1'] = init_bias_shared([num_filters[0]], name='bc1rnshared')
            weights['wc2'] = get_xavier_weights_shared([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rnshared') # 5x5 conv, 32 inputs, 64 outputs
            biases['bc2'] = init_bias_shared([num_filters[1]], name='bc2rnshared')

            tf.get_variable_scope().reuse_variables()
            conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'])

            conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])


            full_y = np.tile(np.arange(im_width), (im_height,1))
            full_x = np.tile(np.arange(im_height), (im_width,1)).T
            full_x = tf.convert_to_tensor(np.reshape(full_x, [-1,1]), dtype=tf.float32)
            full_y = tf.convert_to_tensor(np.reshape(full_y, [-1,1] ), dtype=tf.float32)
            feature_points = []
            f_x = []
            f_y = []
            for filter_number in range(num_filters[1]):
                conv_filter_chosen = conv_layer_1[:,:,:,filter_number]
                conv_filter_chosen = tf.reshape(conv_filter_chosen, [-1, im_width*im_height])
                conv_softmax = tf.nn.softmax(conv_filter_chosen)
                feature_points_x = tf.matmul(conv_softmax, full_x)
                feature_points_y = tf.matmul(conv_softmax, full_y)
                f_x.append(feature_points_x)
                f_y.append(feature_points_y)
                feature_points.append(feature_points_x)
                feature_points.append(feature_points_y)
            full_feature_points = tf.concat(concat_dim=1, values=feature_points)
            f_x = tf.concat(concat_dim=1, values=f_x)
            f_y = tf.concat(concat_dim=1, values=f_y)
            fc_input = tf.concat(concat_dim=1, values=[full_feature_points, state_input])
            fc_output, weights_FC, biases_FC = get_mlp_layers(fc_input, n_layers, dim_hidden, robot_number=robot_number)
            fc_vars += weights_FC
            fc_vars += biases_FC
            last_conv_vars.append(fc_input)
            loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss],
                                               feature_points=full_feature_points))
    return nnets, fc_vars, last_conv_vars

def multi_input_multi_output_images_shared_conv1conv2(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    
    fc_vars = []
    last_conv_vars = []
    num_robots = len(dim_input)
    nnets = []
    st_idx = []
    im_idx = []
    i = []
    for robot_number in range(num_robots):
        st_idx.append([])
        im_idx.append([])
        i.append(0)
    #need to fix whatever this is 
    all_vars = []
    with tf.variable_scope("shared_wts"):
        for robot_number, robot_params in enumerate(network_config):
            n_layers = 3
            layer_size = 20
            dim_hidden = (n_layers - 1)*[layer_size]
            dim_hidden.append(dim_output[robot_number])
            pool_size = 2
            filter_size = 3
            for sensor in robot_params['obs_include']:
                dim = robot_params['sensor_dims'][sensor]
                if sensor in robot_params['obs_image_data']:
                    im_idx[robot_number] = im_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                else:
                    st_idx[robot_number] = st_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                i[robot_number] += dim

            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

            state_input = nn_input[:, 0:st_idx[robot_number][-1]+1]
            image_input = nn_input[:, st_idx[robot_number][-1]+1:im_idx[robot_number][-1]+1]

            # image goes through 2 convnet layers
            num_filters = network_config[robot_number]['num_filters']

            im_height = network_config[robot_number]['image_height']
            im_width = network_config[robot_number]['image_width']
            num_channels = network_config[robot_number]['image_channels']
            image_input = tf.reshape(image_input, [-1, num_channels, im_width, im_height])
            image_input = tf.transpose(image_input, perm=[0,3,2,1])

                # Store layers weight & bias
            weights = {
                # 'wc1': get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rn' + str(robot_number)), # 5x5 conv, 1 input, 32 outputs
                # 'wc2': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rn' + str(robot_number)), # 5x5 conv, 1 input, 32 outputs

            }

            biases = {
                # 'bc1': init_bias([num_filters[0]], name='bc1rn' + str(robot_number)),
                # 'bc2': init_bias([num_filters[1]], name='bc2rn' + str(robot_number)),
            }
            weights['wc1'] = get_xavier_weights_shared([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rnshared')
            biases['bc1'] = init_bias_shared([num_filters[0]], name='bc1rnshared')
            weights['wc2'] = get_xavier_weights_shared([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rnshared') # 5x5 conv, 32 inputs, 64 outputs
            biases['bc2'] = init_bias_shared([num_filters[1]], name='bc2rnshared')

            tf.get_variable_scope().reuse_variables()
            conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'])

            conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])


            full_y = np.tile(np.arange(im_width), (im_height,1))
            full_x = np.tile(np.arange(im_height), (im_width,1)).T
            full_x = tf.convert_to_tensor(np.reshape(full_x, [-1,1]), dtype=tf.float32)
            full_y = tf.convert_to_tensor(np.reshape(full_y, [-1,1] ), dtype=tf.float32)
            feature_points = []
            f_x = []
            f_y = []
            for filter_number in range(num_filters[1]):
                conv_filter_chosen = conv_layer_1[:,:,:,filter_number]
                conv_filter_chosen = tf.reshape(conv_filter_chosen, [-1, im_width*im_height])
                conv_softmax = tf.nn.softmax(conv_filter_chosen)
                feature_points_x = tf.matmul(conv_softmax, full_x)
                feature_points_y = tf.matmul(conv_softmax, full_y)
                f_x.append(feature_points_x)
                f_y.append(feature_points_y)
                feature_points.append(feature_points_x)
                feature_points.append(feature_points_y)
            full_feature_points = tf.concat(concat_dim=1, values=feature_points)
            f_x = tf.concat(concat_dim=1, values=f_x)
            f_y = tf.concat(concat_dim=1, values=f_y)
            fc_input = tf.concat(concat_dim=1, values=[full_feature_points, state_input])
            fc_output, weights_FC, biases_FC = get_mlp_layers(fc_input, n_layers, dim_hidden, robot_number=robot_number)
            fc_vars += weights_FC
            fc_vars += biases_FC
            last_conv_vars.append(fc_input)
            loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss],
                                               feature_points=full_feature_points))
            all_vars.append(weights)
            all_vars.append(biases)
    return nnets, fc_vars, last_conv_vars, all_vars, [conv_layer_0, conv_layer_1, full_feature_points, fc_input, fc_output]


def multi_input_multi_output_images_shared(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    
    fc_vars = []
    last_conv_vars = []
    num_robots = len(dim_input)
    nnets = []
    st_idx = []
    im_idx = []
    i = []
    for robot_number in range(num_robots):
        st_idx.append([])
        im_idx.append([])
        i.append(0)
    #need to fix whatever this is 
    all_vars = []
    with tf.variable_scope("shared_wts"):
        for robot_number, robot_params in enumerate(network_config):
            n_layers = 3
            layer_size = 20
            dim_hidden = (n_layers - 1)*[layer_size]
            dim_hidden.append(dim_output[robot_number])
            pool_size = 2
            filter_size = 3
            for sensor in robot_params['obs_include']:
                dim = robot_params['sensor_dims'][sensor]
                if sensor in robot_params['obs_image_data']:
                    im_idx[robot_number] = im_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                else:
                    st_idx[robot_number] = st_idx[robot_number] + list(range(i[robot_number], i[robot_number]+dim))
                i[robot_number] += dim

            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

            state_input = nn_input[:, 0:st_idx[robot_number][-1]+1]
            image_input = nn_input[:, st_idx[robot_number][-1]+1:im_idx[robot_number][-1]+1]

            # image goes through 2 convnet layers
            num_filters = network_config[robot_number]['num_filters']

            im_height = network_config[robot_number]['image_height']
            im_width = network_config[robot_number]['image_width']
            num_channels = network_config[robot_number]['image_channels']
            image_input = tf.reshape(image_input, [-1, num_channels, im_width, im_height])
            image_input = tf.transpose(image_input, perm=[0,3,2,1])

                # Store layers weight & bias
            weights = {
                'wc1': get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rn' + str(robot_number)), # 5x5 conv, 1 input, 32 outputs
                'wc2': get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rn' + str(robot_number)), # 5x5 conv, 1 input, 32 outputs

            }

            biases = {
                'bc1': init_bias([num_filters[0]], name='bc1rn' + str(robot_number)),
                'bc2': init_bias([num_filters[1]], name='bc2rn' + str(robot_number)),
            }
            # weights['wc1'] = get_xavier_weights_shared([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rnshared')
            # biases['bc1'] = init_bias_shared([num_filters[0]], name='bc1rnshared')
            # weights['wc2'] = get_xavier_weights_shared([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rnshared') # 5x5 conv, 32 inputs, 64 outputs
            # biases['bc2'] = init_bias_shared([num_filters[1]], name='bc2rnshared')

            # tf.get_variable_scope().reuse_variables()
            conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'])

            conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])


            full_y = np.tile(np.arange(im_width), (im_height,1))
            full_x = np.tile(np.arange(im_height), (im_width,1)).T
            full_x = tf.convert_to_tensor(np.reshape(full_x, [-1,1]), dtype=tf.float32)
            full_y = tf.convert_to_tensor(np.reshape(full_y, [-1,1] ), dtype=tf.float32)
            feature_points = []
            f_x = []
            f_y = []
            for filter_number in range(num_filters[1]):
                conv_filter_chosen = conv_layer_1[:,:,:,filter_number]
                conv_filter_chosen = tf.reshape(conv_filter_chosen, [-1, im_width*im_height])
                conv_softmax = tf.nn.softmax(conv_filter_chosen)
                feature_points_x = tf.matmul(conv_softmax, full_x)
                feature_points_y = tf.matmul(conv_softmax, full_y)
                f_x.append(feature_points_x)
                f_y.append(feature_points_y)
                feature_points.append(feature_points_x)
                feature_points.append(feature_points_y)
            full_feature_points = tf.concat(concat_dim=1, values=feature_points)
            f_x = tf.concat(concat_dim=1, values=f_x)
            f_y = tf.concat(concat_dim=1, values=f_y)
            fc_input = tf.concat(concat_dim=1, values=[full_feature_points, state_input])
            fc_output, weights_FC, biases_FC = get_mlp_layers(fc_input, n_layers, dim_hidden, robot_number=robot_number)
            fc_vars += weights_FC
            fc_vars += biases_FC
            last_conv_vars.append(fc_input)
            loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss],
                                               feature_points=full_feature_points))
            all_vars.append(weights)
            all_vars.append(biases)
    return nnets, fc_vars, last_conv_vars, all_vars, [conv_layer_0, conv_layer_1, full_feature_points, fc_input, fc_output]

def get_loss_layer(mlp_out, action, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    return euclidean_loss_layer(a=action, b=mlp_out, precision=precision, batch_size=batch_size)


def example_tf_network(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    An example of how one might want to specify a network in tensorflow.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
    Returns:
        a TfMap object used to serialize, inputs, outputs, and loss.
    """
    n_layers = 2
    dim_hidden = (n_layers - 1) * [40]
    dim_hidden.append(dim_output)

    nn_input, action, precision = get_input_layer(dim_input, dim_output)
    mlp_applied = get_mlp_layers(nn_input, n_layers, dim_hidden)
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])


def example_tf_network_multi(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    
    
    num_robots = len(dim_input)
    nnets = []
    with tf.variable_scope("shared_wts"):
        for robot_number, robot_params in enumerate(network_config):
            n_layers = 4
            layer_size = 60
            dim_hidden = (n_layers - 1)*[layer_size]
            dim_hidden.append(dim_output[robot_number])

            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

            state_input = nn_input

            fc_output, weights_FC, biases_FC, layers = get_mlp_layers(state_input, n_layers, dim_hidden, robot_number=robot_number)

            loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss]))

    return nnets, None, None, weights_FC + biases_FC, layers


def example_tf_network_multi_contrastive(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    
    
    num_robots = len(dim_input)
    nnets = []
    feature_layers = []
    with tf.variable_scope("shared_wts"):
        for robot_number, robot_params in enumerate(network_config):
            n_layers = 4
            layer_size = 60
            dim_hidden = (n_layers - 1)*[layer_size]
            dim_hidden.append(dim_output[robot_number])

            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

            state_input = nn_input

            fc_output, weights_FC, biases_FC, layers = get_mlp_layers(state_input, n_layers, dim_hidden, robot_number=robot_number)
            feature_layers.append(layers[-2])
            
            loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
            if robot_number == 1:
                contrastive = tf.nn.l2_loss(feature_layers[0]-feature_layers[1])
                loss = loss + contrastive
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss]))

    return nnets, None, None, weights_FC + biases_FC, contrastive

def conv2d(img, w, b):
    #print img.get_shape().dims[3].value
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
