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

def get_mlp_layers_shared(mlp_input, number_layers, dimension_hidden, robot_number):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    layers = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights_shared([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step))
        weights.append(cur_weight)
        cur_bias = init_bias_shared([dimension_hidden[layer_step]], name='b_' + str(layer_step))
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias
        layers.append(cur_top)
    return cur_top, weights, biases, layers


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

    return nnets, None

def model_fc_shared(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
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
    all_vars = []
    n_layers = 4
    layer_size = 60
    dim_hidden = (n_layers - 1)*[layer_size]

    with tf.variable_scope("shared_wts"):
        shared_weights = {
            # 'w0' : init_weights((dim_input[0], dim_hidden[0]), name='w0'),
            'w1' : init_weights((dim_hidden[0], dim_hidden[1]), name='w1'),
            'w2' : init_weights((dim_hidden[1], dim_hidden[2]), name='w2'),
            # 'w3' : init_weights((dim_hidden[2], dim_output[0]), name='w3'),
            # 'b0' : init_bias((dim_hidden[0],), name='b0'),
            'b1' : init_bias((dim_hidden[1],), name='b1'),
            'b2' : init_bias((dim_hidden[2],), name='b2'),
            # 'b3' : init_bias((dim_output[0],), name='b3'),
        }

        for robot_number, robot_params in enumerate(network_config):
            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

            w_input = init_weights((dim_input[robot_number],dim_hidden[0]), name='w_input' + str(robot_number))
            b_input = init_bias((dim_hidden[0],), name='b_input'+str(robot_number))

            w_output = init_weights((dim_hidden[-1], dim_output[robot_number]), name='w_output'+str(robot_number))
            b_output = init_bias((dim_output[robot_number],), name = 'b_output'+str(robot_number))

            layer0 = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
            layer1 = tf.nn.relu(tf.matmul(layer0, shared_weights['w1']) + shared_weights['b1'])
            layer2 = tf.nn.relu(tf.matmul(layer1, shared_weights['w2']) + shared_weights['b2'])
            output = tf.matmul(layer2, w_output) + b_output

            loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)

            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
            fc_vars = None

    return nnets, None, None, all_vars, [],


def model_fc_shared_diff(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
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
    all_vars = []
    n_layers = 4
    layer_size = 60
    dim_diff = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    index_shared = [11, 12, 13, 17, 18, 19]
    index_indiv = [0,1,2,3,4,5,6,7,8,9,10,14,15,16]
    with tf.variable_scope("shared_wts"):
        shared_weights = {
            'w0' : init_weights((len(index_shared), dim_hidden[0]), name='w0'),
            'w1' : init_weights((dim_hidden[0], dim_hidden[1]), name='w1'),
            'w2' : init_weights((dim_hidden[1], dim_hidden[2]), name='w2'),
            # 'w_diff' : init_weights((len(index_indiv), dim_diff), name='w_diff'),
            'w3' : init_weights((dim_hidden[2] + dim_diff, dim_output[0]), name='w3'),
            'b0' : init_bias((dim_hidden[0],), name='b0'),
            'b1' : init_bias((dim_hidden[1],), name='b1'),
            'b2' : init_bias((dim_hidden[2],), name='b2'),
            # 'b_diff' : init_bias((dim_diff,), name='b_diff'),
            'b3' : init_bias((dim_output[0],), name='b3'),
        }

        for robot_number, robot_params in enumerate(network_config):
            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)
            shared_input = tf.concat(1, [nn_input[:, 11:14], nn_input[:, 17:]])
            indiv_input = tf.concat(1, [nn_input[:, 0:11], nn_input[:, 14:17]])
            # w_input = init_weights((dim_input[robot_number],dim_hidden[0]), name='w_input' + str(robot_number))
            # b_input = init_bias((dim_hidden[0],), name='b_input'+str(robot_number))

            # w_diff = init_weights((len(index_indiv), dim_diff), name='w_diff'+str(robot_number))
            # b_diff = init_bias((dim_diff,), name = 'b_diff'+str(robot_number))
 
            layer0 = tf.nn.relu(tf.matmul(shared_input, shared_weights['w0']) + shared_weights['b0'])
            layer1 = tf.nn.relu(tf.matmul(layer0, shared_weights['w1']) + shared_weights['b1'])
            layer2 = tf.nn.relu(tf.matmul(layer1, shared_weights['w2']) + shared_weights['b2'])
            layer_diff = tf.nn.relu(tf.matmul(indiv_input, w_diff) + b_diff)
            fc_input = tf.concat(concat_dim=1, values=[layer2, layer_diff])
            output = tf.matmul(fc_input, shared_weights['w3']) + shared_weights['b3']

            loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)

            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
            fc_vars = None

    return nnets, None, None, all_vars, [],

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


def multitask_multirobot_fc(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a list of dictionaries containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    #need to create taskrobot_mapping
    taskrobot_mapping = np.asarray([[0, 1], [2, 3], [4, None]])
    num_robots = taskrobot_mapping.shape[1]
    num_tasks = taskrobot_mapping.shape[0]
    task_list = [None]*(len(dim_input))
    robot_list = [None]*(len(dim_output))
    for robot_number in range(num_robots):
        for task_number in range(num_tasks):
            if taskrobot_mapping[task_number][robot_number] is not None:
                task_list[taskrobot_mapping[task_number][robot_number]] = task_number
                robot_list[taskrobot_mapping[task_number][robot_number]] = robot_number
    nnets = []
    n_layers = 4
    layer_size = 60
    dim_hidden = (n_layers - 1)*[layer_size]
    shared_weights = {}
    dim_robot_specific_list = [12, 14]
    dim_task_specific_list = [6, 16, 6]
    dim_robot_output_list = [3, 4]
    dim_diff = 20

    for robot_number in range(num_robots):
        #special case possible
        dim_robot_output = dim_robot_output_list[robot_number]
        dim_robot_specific = dim_robot_specific_list[robot_number]
        shared_weights['w3_rn_' + str(robot_number)] = init_weights((dim_hidden[1], dim_hidden[2]), name='w3_rn_' + str(robot_number))
        shared_weights['b3_rn_' + str(robot_number)] = init_bias((dim_hidden[2],), name='b3_rn_' + str(robot_number))
        shared_weights['wdiff_rn_' + str(robot_number)] = init_weights((dim_robot_specific, dim_diff), name='wdiff_rn_' + str(robot_number))
        shared_weights['bdiff_rn_' + str(robot_number)] = init_bias((dim_diff,), name='bdiff_rn_' + str(robot_number))
        shared_weights['wout_rn_' + str(robot_number)] = init_weights((dim_hidden[2] + dim_diff, dim_robot_output), name='wout_rn_' + str(robot_number))
        shared_weights['bout_rn_' + str(robot_number)] = init_bias((dim_robot_output,), name='bout_rn_' + str(robot_number))

    for task_number in range(num_tasks):
        dim_task_input = dim_task_specific_list[task_number]
        shared_weights['w1_tn_' + str(task_number)] = init_weights((dim_task_input, dim_hidden[0]), name='w1_tn_' + str(task_number))
        shared_weights['b1_tn_' + str(task_number)] = init_bias((dim_hidden[0],), name='b1_tn_' + str(task_number))
        shared_weights['w2_tn_' + str(task_number)] = init_weights((dim_hidden[0], dim_hidden[1]), name='w2_tn_' + str(task_number))
        shared_weights['b2_tn_' + str(task_number)] = init_bias((dim_hidden[1],), name='b2_tn_' + str(task_number))

    # import IPython
    # IPython.embed()
    for robot_number, robot_params in enumerate(network_config):
        robot_index = robot_list[robot_number]
        task_index = task_list[robot_number]

        nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

        if robot_index == 0 and task_index == 0:
            robot_input = tf.concat(1, [nn_input[:, 0:9], nn_input[:, 12:15]])
            task_input = tf.concat(1, [nn_input[:, 9:12], nn_input[:, 15:]])  
        elif robot_index == 1 and task_index == 0:
            robot_input = tf.concat(1, [nn_input[:, 0:11], nn_input[:, 14:17]])
            task_input = tf.concat(1, [nn_input[:, 11:14], nn_input[:, 17:]])    
        elif robot_index == 0 and task_index == 1:
            robot_input = tf.concat(1, [nn_input[:, 0:3], nn_input[:, 5:8], nn_input[:, 10:13], nn_input[:, 19:22]])
            task_input = tf.concat(1, [nn_input[:, 3:5], nn_input[:, 8:10], nn_input[:, 13:19], nn_input[:, 22:]])
        elif robot_index == 1 and task_index == 1:
            robot_input = tf.concat(1, [nn_input[:, 0:4], nn_input[:, 6:10], nn_input[:, 12:15], nn_input[:, 21:24]])
            task_input = tf.concat(1, [nn_input[:, 4:6], nn_input[:, 10:12], nn_input[:, 15:21], nn_input[:, 24:]])  
        elif robot_index == 0 and task_index == 2:
            robot_input = tf.concat(1, [nn_input[:, 0:9], nn_input[:, 12:15]])
            task_input = tf.concat(1, [nn_input[:, 9:12], nn_input[:, 15:]])  
        elif robot_index == 1 and task_index == 2:
            robot_input = tf.concat(1, [nn_input[:, 0:11], nn_input[:, 14:17]])
            task_input = tf.concat(1, [nn_input[:, 11:14], nn_input[:, 17:]])  

        layer1 = tf.nn.relu(tf.matmul(task_input, shared_weights['w1_tn_' + str(task_index)]) + shared_weights['b1_tn_' + str(task_index)])
        layer2 = tf.nn.relu(tf.matmul(layer1, shared_weights['w2_tn_' + str(task_index)]) + shared_weights['b2_tn_' + str(task_index)])
        layer3 = tf.nn.relu(tf.matmul(layer2, shared_weights['w3_rn_' + str(robot_index)]) + shared_weights['b3_rn_' + str(robot_index)])
        layer_diff = tf.nn.relu(tf.matmul(robot_input, shared_weights['wdiff_rn_' + str(robot_index)]) + shared_weights['bdiff_rn_' + str(robot_index)])
        lastlayer_input = tf.concat(concat_dim=1, values=[layer3, layer_diff])
        output = tf.matmul(lastlayer_input, shared_weights['wout_rn_' + str(robot_index)]) + shared_weights['bout_rn_' + str(robot_index)]

        loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
    return nnets, None, None, shared_weights, None


def multitask_forward(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a list of dictionaries containing inputs, outputs, and the loss function representing scalar loss.
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    #need to create taskrobot_mapping
   
    num_robots = 1
    num_tasks = 1
    nnets = []
    n_layers = 4
    layer_size = 60
    dim_hidden = (n_layers - 1)*[layer_size]
    shared_weights = {}
    dim_robot_specific_list = [12, 14]
    dim_task_specific_list = [6, 16, 6]
    dim_robot_output_list = [3, 4]
    dim_diff = 20

    for robot_number in range(1,2):
        #special case possible
        dim_robot_output = dim_robot_output_list[robot_number]
        dim_robot_specific = dim_robot_specific_list[robot_number]
        shared_weights['w3_rn_' + str(robot_number)] = init_weights((dim_hidden[1], dim_hidden[2]), name='w3_rn_' + str(robot_number))
        shared_weights['b3_rn_' + str(robot_number)] = init_bias((dim_hidden[2],), name='b3_rn_' + str(robot_number))
        shared_weights['wdiff_rn_' + str(robot_number)] = init_weights((dim_robot_specific, dim_diff), name='wdiff_rn_' + str(robot_number))
        shared_weights['bdiff_rn_' + str(robot_number)] = init_bias((dim_diff,), name='bdiff_rn_' + str(robot_number))
        shared_weights['wout_rn_' + str(robot_number)] = init_weights((dim_hidden[2] + dim_diff, dim_robot_output), name='wout_rn_' + str(robot_number))
        shared_weights['bout_rn_' + str(robot_number)] = init_bias((dim_robot_output,), name='bout_rn_' + str(robot_number))

    for task_number in range(2,3):
        dim_task_input = dim_task_specific_list[task_number]
        shared_weights['w1_tn_' + str(task_number)] = init_weights((dim_task_input, dim_hidden[0]), name='w1_tn_' + str(task_number))
        shared_weights['b1_tn_' + str(task_number)] = init_bias((dim_hidden[0],), name='b1_tn_' + str(task_number))
        shared_weights['w2_tn_' + str(task_number)] = init_weights((dim_hidden[0], dim_hidden[1]), name='w2_tn_' + str(task_number))
        shared_weights['b2_tn_' + str(task_number)] = init_bias((dim_hidden[1],), name='b2_tn_' + str(task_number))

    # import IPython
    # IPython.embed()
    for robot_number, robot_params in enumerate(network_config):
        robot_index = 1
        task_index = 2
        
        nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

        if robot_index == 0 and task_index == 0:
            robot_input = tf.concat(1, [nn_input[:, 0:9], nn_input[:, 12:15]])
            task_input = tf.concat(1, [nn_input[:, 9:12], nn_input[:, 15:]])  
        elif robot_index == 1 and task_index == 0:
            robot_input = tf.concat(1, [nn_input[:, 0:11], nn_input[:, 14:17]])
            task_input = tf.concat(1, [nn_input[:, 11:14], nn_input[:, 17:]])    
        elif robot_index == 0 and task_index == 1:
            robot_input = tf.concat(1, [nn_input[:, 0:3], nn_input[:, 5:8], nn_input[:, 10:13], nn_input[:, 19:22]])
            task_input = tf.concat(1, [nn_input[:, 3:5], nn_input[:, 8:10], nn_input[:, 13:19], nn_input[:, 22:]])
        elif robot_index == 1 and task_index == 1:
            robot_input = tf.concat(1, [nn_input[:, 0:4], nn_input[:, 6:10], nn_input[:, 12:15], nn_input[:, 21:24]])
            task_input = tf.concat(1, [nn_input[:, 4:6], nn_input[:, 10:12], nn_input[:, 15:21], nn_input[:, 24:]])  
        elif robot_index == 0 and task_index == 2:
            robot_input = tf.concat(1, [nn_input[:, 0:9], nn_input[:, 12:15]])
            task_input = tf.concat(1, [nn_input[:, 9:12], nn_input[:, 15:]])  
        elif robot_index == 1 and task_index == 2:
            robot_input = tf.concat(1, [nn_input[:, 0:11], nn_input[:, 14:17]])
            task_input = tf.concat(1, [nn_input[:, 11:14], nn_input[:, 17:]])  

        layer1 = tf.nn.relu(tf.matmul(task_input, shared_weights['w1_tn_' + str(task_index)]) + shared_weights['b1_tn_' + str(task_index)])
        layer2 = tf.nn.relu(tf.matmul(layer1, shared_weights['w2_tn_' + str(task_index)]) + shared_weights['b2_tn_' + str(task_index)])
        layer3 = tf.nn.relu(tf.matmul(layer2, shared_weights['w3_rn_' + str(robot_index)]) + shared_weights['b3_rn_' + str(robot_index)])
        layer_diff = tf.nn.relu(tf.matmul(robot_input, shared_weights['wdiff_rn_' + str(robot_index)]) + shared_weights['bdiff_rn_' + str(robot_index)])
        lastlayer_input = tf.concat(concat_dim=1, values=[layer3, layer_diff])
        output = tf.matmul(lastlayer_input, shared_weights['wout_rn_' + str(robot_index)]) + shared_weights['bout_rn_' + str(robot_index)]

        loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
    return nnets, None, None, shared_weights, None

def invariant_subspace_test(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    num_robots = len(dim_input)
    nnets = []
    n_layers = 5
    layer_size = 60
    dim_hidden = (n_layers - 1)*[layer_size]
    feature_layers = []
    weight_dict = {}
    for robot_number, robot_params in enumerate(network_config):
        indiv_losses = []
        nn_input = tf.placeholder("float", [None, dim_input[robot_number]], name='nn_input' + str(robot_number))
        w_input = init_weights((dim_input[robot_number],dim_hidden[0]), name='w_input' + str(robot_number))
        b_input = init_bias((dim_hidden[0],), name='b_input'+str(robot_number))
        w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='w1_' + str(robot_number))
        b1 = init_bias((dim_hidden[1],), name='b1_' + str(robot_number))
        w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='w2_' + str(robot_number))
        b2 = init_bias((dim_hidden[2],), name='b2_' + str(robot_number))
        w3 = init_weights((dim_hidden[2], dim_hidden[3]), name='w3_' + str(robot_number))
        b3 = init_bias((dim_hidden[3],), name='b3_' + str(robot_number))
        w_output = init_weights((dim_hidden[3], dim_input[robot_number]), name='w_output'+str(robot_number))
        b_output = init_bias((dim_input[robot_number],), name = 'b_output'+str(robot_number))
        layer0 = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
        layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        feature_layers.append(layer2)
        layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
        output = tf.matmul(layer3, w_output) + b_output
        loss = tf.nn.l2_loss(nn_input - output) #euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        indiv_losses.append(loss)
        if robot_number == 1:
            contrastive = tf.nn.l2_loss(feature_layers[0]-feature_layers[1])
            scale_factor = 1.0
            contrastive = contrastive*scale_factor
            indiv_losses.append(contrastive)
            #might need to scale here
            loss = loss + contrastive
        nnets.append(TfMap.init_from_lists([nn_input, None, None], [output], [loss], layer2, indiv_losses))
        weight_dict[w_input.name] = w_input
        weight_dict[b_input.name] = b_input
        weight_dict[w1.name] = w1
        weight_dict[b1.name] = b1
        weight_dict[w2.name] = w2
        weight_dict[b2.name] = b2
        weight_dict[w3.name] = w3
        weight_dict[b3.name] = b3
        weight_dict[w_output.name] = w_output
        weight_dict[b_output.name] = b_output
    return nnets, weight_dict

def invariant_subspace_test_action(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    num_robots = len(dim_input)
    nnets = []
    n_layers = 4
    layer_size = 20
    dim_hidden = (n_layers - 1)*[layer_size]
    feature_layers = []
    weight_dict = {}
    for robot_number, robot_params in enumerate(network_config):
        indiv_losses = []
        nn_input = tf.placeholder("float", [None, dim_input[robot_number]], name='action_nn_input10' + str(robot_number))
        w_input = init_weights((dim_input[robot_number],dim_hidden[0]), name='aw_input10' + str(robot_number))
        b_input = init_bias((dim_hidden[0],), name='ab_input10'+str(robot_number))
        w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='aw1_10' + str(robot_number))
        b1 = init_bias((dim_hidden[1],), name='ab1_10' + str(robot_number))
        w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='aw2_10' + str(robot_number))
        b2 = init_bias((dim_hidden[2],), name='ab2_10' + str(robot_number))
        w_output = init_weights((dim_hidden[2], dim_input[robot_number]), name='aw_output10'+str(robot_number))
        b_output = init_bias((dim_input[robot_number],), name = 'ab_output10'+str(robot_number))
        layer0 = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
        layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
        feature_layers.append(layer1)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        output = tf.matmul(layer2, w_output) + b_output
        loss = tf.nn.l2_loss(nn_input - output) #euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        indiv_losses.append(loss)
        if robot_number == 1:
            contrastive = tf.nn.l2_loss(feature_layers[0]-feature_layers[1])
            scale_factor = 0.5
            contrastive = contrastive*scale_factor
            indiv_losses.append(contrastive)
            #might need to scale here
            loss = loss + contrastive
        nnets.append(TfMap.init_from_lists([nn_input, None, None], [output], [loss], layer1, indiv_losses))
        weight_dict[w_input.name] = w_input
        weight_dict[b_input.name] = b_input
        weight_dict[w1.name] = w1
        weight_dict[b1.name] = b1
        weight_dict[w2.name] = w2
        weight_dict[b2.name] = b2
        weight_dict[w_output.name] = w_output
        weight_dict[b_output.name] = b_output
    return nnets, weight_dict

def double_contrastive_invariance(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    num_robots = len(dim_input)
    nnets = []
    n_layers = 7
    layer_size = 60
    dim_hidden = (n_layers - 1)*[layer_size]
    dim_hidden[2] = 100
    feature_layers_cl1 = []
    feature_layers_cl2 = []
    weight_dict = {}
    dim_input_robotspecific = [12, 14]
    dim_input_taskspecific = [6, 6]
    for robot_number, robot_params in enumerate(network_config):
        nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)
        if robot_number == 0:
            robotspecific_input = tf.concat(1, [nn_input[:, 0:9], nn_input[:, 12:15]])
            taskspecific_input = tf.concat(1, [nn_input[:, 9:12], nn_input[:, 15:18]])  
        elif robot_number == 1:
            robotspecific_input = tf.concat(1, [nn_input[:, 0:11], nn_input[:, 14:17]])
            taskspecific_input = tf.concat(1, [nn_input[:, 11:14], nn_input[:, 17:20]])  

        w0 = init_weights((dim_input_robotspecific[robot_number], dim_hidden[0]), name='w0_rn_' + str(robot_number))
        b0 = init_bias((dim_hidden[0],), name='b0_rn_'+str(robot_number))
        # weight_dict[w0.name] = w0
        # weight_dict[b0.name] = b0

        w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='w1_rn_' + str(robot_number))
        b1 = init_bias((dim_hidden[1],), name='b1_rn_' + str(robot_number))
        # weight_dict[w1.name] = w1
        # weight_dict[b1.name] = b1

        w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='w2_rn_' + str(robot_number))
        b2 = init_bias((dim_hidden[2],), name='b2_rn_' + str(robot_number))
        # weight_dict[w2.name] = w2
        # weight_dict[b2.name] = b2

        w3 = init_weights((dim_hidden[2] + dim_input_taskspecific[robot_number], dim_hidden[3]), name='w3_rn_' + str(robot_number))
        b3 = init_bias((dim_hidden[3],), name='b3_rn_' + str(robot_number))
        # weight_dict[w3.name] = w3
        # weight_dict[b3.name] = b3

        w4 = init_weights((dim_hidden[3], dim_hidden[4]), name='w4_rn_' + str(robot_number))
        b4 = init_bias((dim_hidden[4],), name='b4_rn_' + str(robot_number))
        # weight_dict[w4.name] = w4
        # weight_dict[b4.name] = b4

        w5 = init_weights((dim_hidden[4], dim_hidden[5]), name='w5_rn_' + str(robot_number))
        b5 = init_bias((dim_hidden[5],), name='b5_rn_' + str(robot_number))
        # w5 = init_weights((dim_hidden[4], dim_output[robot_number]), name='w5_rn_' + str(robot_number))
        # b5 = init_bias((dim_output[robot_number],), name='b5_rn_' + str(robot_number))
        # weight_dict[w5.name] = w5
        # weight_dict[b5.name] = b5

        w6 = init_weights((dim_hidden[5], dim_output[robot_number]), name='w6_' + str(robot_number))
        b6 = init_bias((dim_output[robot_number],), name='b6_' + str(robot_number))
        # weight_dict[w6.name] = w6
        # weight_dict[b6.name] = b6

        # w7 = init_weights((dim_hidden[6], dim_output[robot_number]), name='w7_'+str(robot_number))
        # b7 = init_bias((dim_output[robot_number],), name = 'b7_'+str(robot_number))
        # weight_dict[w7.name] = w7
        # weight_dict[b7.name] = b7

        layer0 = tf.nn.relu(tf.matmul(robotspecific_input, w0) + b0)
        layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        feature_layers_cl1.append(layer2)
        layer2 = tf.concat(1, [layer2, taskspecific_input])
        layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
        layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)
        # layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)
        layer5 = tf.nn.relu(tf.matmul(layer4, w5) + b5)
        # feature_layers_cl2.append(layer5)
        # layer6 = tf.nn.relu(tf.matmul(layer5, w6) + b6)
        output = tf.matmul(layer5, w6) + b6
        loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        if robot_number == 1:
            contrastive_1 = tf.nn.l2_loss(feature_layers_cl1[0] - feature_layers_cl1[1])
            # contrastive_2 = tf.nn.l2_loss(feature_layers_cl2[0] - feature_layers_cl2[1])
            #might need to scale here
            weight_cl1 = 1.0
            # weight_cl2 = 1.0
            loss = loss + weight_cl1*contrastive_1 #+ weight_cl2*contrastive_2
        nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
    return nnets, None


def unsup_domain_confusion(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    num_robots = len(dim_input)
    nnets = []
    n_layers = 5
    layer_size = 60
    dim_hidden = (n_layers - 1)*[layer_size]
    feature_layers = []
    weight_dict = {}
    gen_vars = []


    ### creating discriminator variables ###
    wdisc1 = init_weights((dim_hidden[2], dim_hidden_disc[0]), name='wdisc1')
    bdisc1 = init_bias((dim_hidden_disc[0],), name='bdisc1')
    wdisc2 = init_weights((dim_hidden_disc[0], dim_hidden_disc[1]), name='wdisc2')
    bdisc2 = init_bias((ddim_hidden_disc[1],), name='bdisc2')
    dc_vars = [dc_w1, dc_b1, dc_w2, dc_b2]
    dc_var_dict = {}
    for var in dc_vars:
        dc_var_dict[var.name] = var

    dc_weight = 1.0
    dc_loss = []
    ### end discriminator variables ###
   
    for robot_number, robot_params in enumerate(network_config):
        indiv_losses = []
        nn_input = tf.placeholder("float", [None, dim_input[robot_number]], name='nn_input' + str(robot_number))


        ### Variable declaration ####
        w_input = init_weights((dim_input[robot_number],dim_hidden[0]), name='w_input' + str(robot_number))
        b_input = init_bias((dim_hidden[0],), name='b_input'+str(robot_number))
        w1 = init_weights((dim_hidden[0], dim_hidden[1]), name='w1_' + str(robot_number))
        b1 = init_bias((dim_hidden[1],), name='b1_' + str(robot_number))
        w2 = init_weights((dim_hidden[1], dim_hidden[2]), name='w2_' + str(robot_number))
        b2 = init_bias((dim_hidden[2],), name='b2_' + str(robot_number))
        w3 = init_weights((dim_hidden[2], dim_hidden[3]), name='w3_' + str(robot_number))
        b3 = init_bias((dim_hidden[3],), name='b3_' + str(robot_number))
        w_output = init_weights((dim_hidden[3], dim_input[robot_number]), name='w_output'+str(robot_number))
        b_output = init_bias((dim_input[robot_number],), name = 'b_output'+str(robot_number))
        gen_vars += [w_input, b_input, w1, b1, w2, b2, w3, b3, w_output, b_output]
        ### End variable declaration ####
       


        ### Start net forward computation ####
        layer0 = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
        layer1 = tf.nn.relu(tf.matmul(layer0, w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        feature_layers.append(layer2)
        layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
        output = tf.matmul(layer3, w_output) + b_output
        ### End net forward computation ####


        ### Computation of discriminator ###
        disc0 = tf.nn.relu(tf.matmul(layer2, wdisc1) + bdisc1)
        disc1 = tf.matmul(disc0, wdisc2) + bdisc2)
        ### End computation of discriminator ###

        ### l2 autoencoder loss function ####
        loss = tf.nn.l2_loss(nn_input - output) #euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        indiv_losses.append(loss)
        ### end l2 autoencoder loss function ####

        ### Terms for unsupervised domain confusion ###
        dc_softmax =  tf.log(tf.nn.softmax(disc1))
        dc_entropy = -1.0/num_robots*tf.reduce_sum(dc_softmax)
        dc_currrobot_loss = -tf.reduce_sum(dc_softmax[:,robot_number])  
        dc_loss.append(dc_currrobot_loss)
        loss = loss + dc_weight*dc_entropy
        ### End terms for unsupervised domain confusion ###
        
        ### Creating TfMap object with appropriate losses ###
        nnets.append(TfMap.init_from_lists([nn_input, None, None], [output], [loss], layer2, indiv_losses))

    for var in gen_vars:
        weight_dict[var.name] = var

    other = {}
    other['dc_loss'] = dc_loss
    other['dc_vars'] = dc_vars
    return nnets, weight_dict, other


def conv2d(img, w, b):
    #print img.get_shape().dims[3].value
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

