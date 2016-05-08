""" This file provides an example tensorflow network used to define a policy. """

import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np

from gps.algorithm.policy_opt.tf_model_example_multirobot import *
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
    feature_layers = []
    with tf.variable_scope("shared_wts"):

        for robot_number, robot_params in enumerate(network_config):
            inv = []
            n_layers = 4
            layer_size = 60
            dim_hidden = (n_layers - 1)*[layer_size]
            #dim_hidden.append(dim_output[robot_number])


            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)
            w_input = init_weights((dim_input[robot_number],dim_hidden[0]), name='w_input' + str(robot_number))
            b_input = init_bias((dim_hidden[0],), name='b_input'+str(robot_number))
            mlp_input = tf.nn.relu(tf.matmul(nn_input, w_input) + b_input)
            print dim_hidden[1:]
            print n_layers-2
            fc_output, weights_FC, biases_FC, layers = get_mlp_layers_shared(mlp_input, n_layers-2, dim_hidden[1:], robot_number=robot_number)

            w_output = init_weights((dim_hidden[-1], dim_output[robot_number]), name='w_output'+str(robot_number))
            b_output = init_bias((dim_output[robot_number],), name = 'b_output'+str(robot_number))
            output = tf.matmul(fc_output, w_output) + b_output
            loss = euclidean_loss_layer(a=action, b=output, precision=precision, batch_size=batch_size)
            tf.get_variable_scope().reuse_variables()
            feature_layers.append(layers[-1])
            if robot_number == 1:
                contrastive = tf.nn.l2_loss(feature_layers[0]-feature_layers[1])
                loss = loss + contrastive
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [output], [loss]))
            fc_vars = None
    return nnets, fc_vars, last_conv_vars, all_vars, []

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
    indiv_net_vars = []
    indiv_fc_vars = []
    indiv_conv_vars = []
    with tf.variable_scope("shared_wts"):
        for robot_number, robot_params in enumerate(network_config):
            n_layers = 4
            layer_size = 60
            dim_hidden = (n_layers - 1)*[layer_size]
            dim_hidden.append(dim_output[robot_number])

            nn_input, action, precision = get_input_layer(dim_input[robot_number], dim_output[robot_number], robot_number)

            state_input = nn_input

            fc_output, weights_FC, biases_FC = get_mlp_layers(state_input, n_layers, dim_hidden, robot_number=robot_number)

            loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss]))
            inv = weights_FC + biases_FC
            indiv_net_vars.append(inv)
            indiv_fc_vars.append([None])


    return nnets, None, None, weights_FC + biases_FC, [], indiv_net_vars, indiv_fc_vars
