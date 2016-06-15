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
    n_layers = 4
    layer_size = 60
    dim_hidden = (n_layers - 1)*[layer_size]

    with tf.variable_scope("shared_wts"):
        shared_weights = {
            'w1' : init_weights((dim_hidden[0], dim_hidden[1]), name='w1'),
            'w2' : init_weights((dim_hidden[1], dim_hidden[2]), name='w2'),
            'b1' : init_bias((dim_hidden[1],), name='b1'),
            'b2' : init_bias((dim_hidden[2],), name='b2'),
        }

        for robot_number, robot_params in enumerate(network_config):
            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)
            w_input = init_weights((dim_input[robot_number],dim_hidden[0]), name='w_input' + str(robot_number))
            b_input = init_bias((dim_hidden[0],), name='b_input'+str(robot_number))
            mlp_input = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
            print dim_hidden[1:]
            print n_layers-2
            # fc_output, weights_FC, biases_FC, layers = get_mlp_layers_shared(mlp_input, n_layers-2, dim_hidden[1:], robot_number=robot_number)
            layer1 = tf.nn.relu(tf.matmul(mlp_input, shared_weights['w1']) + shared_weights['b1'])
            layer2 = tf.nn.relu(tf.matmul(layer1, shared_weights['w2']) + shared_weights['b2'])

            w_output = init_weights((dim_hidden[-1], dim_output[robot_number]), name='w_output'+str(robot_number))
            b_output = init_bias((dim_output[robot_number],), name = 'b_output'+str(robot_number))
            output = tf.matmul(layer2, w_output) + b_output

            loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
    return nnets, None, None, None, None

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
    for robot_number in range(num_robots):
        #special case possible
        dim_input_thisrobot = dim_input[taskrobot_mapping[0][robot_number]]
        shared_weights['w1_rn_' + str(robot_number)] = init_weights((dim_input_thisrobot, dim_hidden[0]), name='w1_rn_' + str(robot_number))
        shared_weights['b1_rn_' + str(robot_number)] = init_bias((dim_hidden[0],), name='b1_rn_' + str(robot_number))

    for task_number in range(num_tasks):
        shared_weights['w2_tn_' + str(task_number)] = init_weights((dim_hidden[0], dim_hidden[1]), name='w2_tn_' + str(task_number))
        shared_weights['b2_tn_' + str(task_number)] = init_bias((dim_hidden[1],), name='b2_tn_' + str(task_number))
        shared_weights['w3_tn_' + str(task_number)] = init_weights((dim_hidden[1], dim_hidden[2]), name='w3_tn_' + str(task_number))
        shared_weights['b3_tn_' + str(task_number)] = init_bias((dim_hidden[2],), name='b3_tn_' + str(task_number))

    for robot_number, robot_params in enumerate(network_config):
        robot_index = robot_list[robot_number]
        task_index = task_list[robot_number]
        nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)
        layer1 = tf.nn.relu(tf.matmul(nn_input, shared_weights['w1_rn_' + str(robot_index)]) + shared_weights['b1_rn_' + str(robot_index)])
        layer2 = tf.nn.relu(tf.matmul(layer1, shared_weights['w2_tn_' + str(task_index)]) + shared_weights['b2_tn_' + str(task_index)])
        layer3 = tf.nn.relu(tf.matmul(layer2, shared_weights['w3_tn_' + str(task_index)]) + shared_weights['b3_tn_' + str(task_index)])
        w_output = init_weights((dim_hidden[-1], dim_output[robot_number]), name='w_output'+str(robot_number))
        b_output = init_bias((dim_output[robot_number],), name = 'b_output'+str(robot_number))
        output = tf.matmul(layer3, w_output) + b_output
        loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
        nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
    return nnets, None, None, None, None

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

