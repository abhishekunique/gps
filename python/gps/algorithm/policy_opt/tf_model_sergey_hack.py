""" This file provides an example tensorflow network used to define a policy. """

import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np
from gps.algorithm.policy_opt.tf_model_example_multirobot import *

def multi_input_multi_output_images_shared_fcvars(dim_input=[27, 27], dim_output=[7, 7], batch_size=25, network_config=None):
    """
    An example a network in theano that has both state and image inputs.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss
    """
    # List of indices for state (vector) data and image (tensor) data in observation.
    print 'making multi-input/output-network'
    num_robots = len(dim_input)
    nnets = []
    st_idx = []
    im_idx = []
    i = []
    fc_vars = []
    last_conv_vars = []
    for robot_number in range(num_robots):
        st_idx.append([])
        im_idx.append([])
        i.append(0)
    #need to fix whatever this is 
    variable_separations = []
    with tf.variable_scope("shared_wts"):
        for robot_number, robot_params in enumerate(network_config):
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
            image_input = tf.reshape(image_input, [-1, im_width, im_height, num_channels])

            #need to resolve this
            dim_hidden = 42
            pool_size = 2
            filter_size = 3
            # we pool twice, each time reducing the image size by a factor of 2.
            conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
            #print conv_out_size
            #print len(st_idx)
            print state_input.get_shape().dims[1].value
            first_dense_size = conv_out_size + len(st_idx[robot_number])  #state_input.get_shape().dims[1].value

            # Store layers weight & bias

            weights = {
                'wc1': get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size), name='wc1rn' + str(robot_number)), # 5x5 conv, 1 input, 32 outputs
                'wd1': init_weights([first_dense_size, dim_hidden], name='wd1rn' + str(robot_number)),
                'out': init_weights([dim_hidden, dim_output[robot_number]], name='outwrn' + str(robot_number))
            }

            biases = {
                'bc1': init_bias([num_filters[0]], name='bc1rn' + str(robot_number)),
                'bd1': init_bias([dim_hidden], name='bd1rn' + str(robot_number)),
                'out': init_bias([dim_output[robot_number]], name='outbrn' + str(robot_number))
            }
            weights['wc2'] = get_xavier_weights_shared([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size), name='wc2rnshared') # 5x5 conv, 32 inputs, 64 outputs
            biases['bc2'] = init_bias_shared([num_filters[1]], name='bc2rnshared')
            tf.get_variable_scope().reuse_variables()
            conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'])

            conv_layer_0 = max_pool(conv_layer_0, k=pool_size)

            conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

            conv_layer_1 = max_pool(conv_layer_1, k=pool_size)

            conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])

            fc_input = tf.concat(concat_dim=1, values=[conv_out_flat, state_input])

            h_1 = tf.nn.relu(tf.matmul(fc_input, weights['wd1']) + biases['bd1'])
            fc_output = tf.matmul(h_1, weights['out']) + biases['out']

            loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, batch_size=batch_size)
            variable_separations.append([weights['wc1'], biases['bc1'], weights['wc2'], biases['bc2'], weights['wd1'], biases['bd1'], weights['out'], biases['out']])
            nnets.append(TfMap.init_from_lists([nn_input, action, precision], [fc_output], [loss]))
            last_conv_vars+= [weights['wc2'], biases['bc2']]
            fc_vars += [weights['wd1'], biases['bd1'], weights['out'], biases['out']]

    return nnets, variable_separations, fc_vars, last_conv_vars
