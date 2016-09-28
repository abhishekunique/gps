""" This file defines the torque (action) cost. """
import copy

import numpy as np

from gps.algorithm.cost.cost import Cost
from gps.proto.gps_pb2 import JOINT_ANGLES, END_EFFECTOR_POINTS, \
        END_EFFECTOR_POINT_JACOBIANS
import tensorflow as tf

class CostTF(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        Cost.__init__(self, hyperparams)
        self.initialized = False

    def initTFGrad(self, Xshape, Ushape, JXshape):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_input = tf.placeholder(tf.float32, shape=(None, Xshape[1]), name="x_input")
            self.u_input = tf.placeholder(tf.float32, shape=(None, Ushape[1]), name="u_input")
            self.jx_input = tf.placeholder(tf.float32, shape=(None, JXshape[1], JXshape[2]), name="jx_input")
            self.loss = self._hyperparams['tf_loss'](self._hyperparams, self.x_input, self.u_input, self.jx_input)
            self.lx = tf.gradients(self.loss, self.x_input)[0]
            self.lu = tf.gradients(self.loss, self.u_input)[0]
            no_u = self.lu is None
            no_x = self.lx is None
            tf_xshape = tf.shape(self.x_input)
            tf_ushape = tf.shape(self.u_input)
            def j(f, x, T, x_dim):
                #TODO(andrew): kinda hacky, document this
                sep_grads = [tf.reshape(tf.gradients(f[:, i], x)[0], (T, 1, x_dim)) for i in range(x_dim)]
                combined_grads = tf.concat(1, sep_grads)
                return tf.transpose(combined_grads, perm=[0, 2, 1])
            if no_u:
                self.lu = tf.zeros(tf_ushape)
                self.luu = tf.zeros((tf_ushape[0], tf_ushape[1], tf_ushape[1]))
            else:
                self.luu = j(self.lu, self.u_input, tf_ushape[0], Ushape[1])
            if no_x:
                self.lx = tf.zeros(tf_xshape)
                self.lxx = tf.zeros((tf_xshape[0], tf_xshape[1], tf_xshape[1]))
            else:
                self.lxx = j(self.lx, self.x_input, tf_xshape[0], Xshape[1])
            if no_u or no_x:
                self.lux = tf.zeros((tf_xshape[0], tf_ushape[1], tf_xshape[1]))
            else:
                self.lux = j(self.lu, self.x_input, tf_xshape[0], Ushape[1])
            init_op = tf.initialize_local_variables()
        self.session = tf.Session(graph=self.graph)
        self.session.run(init_op)
        self.initialized = True

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        sample_x = sample.get_X()
        sample_u = sample.get_U()
        jx = sample.get(END_EFFECTOR_POINT_JACOBIANS)

        if not self.initialized:
            self.initTFGrad(sample_x.shape, sample_u.shape, jx.shape)

        tf_loss, tf_lx, tf_lu, tf_lxx, tf_luu, tf_lux = self.session.run([self.loss, self.lx, self.lu, self.lxx, self.luu, self.lux],
                                            {self.x_input: sample_x,
                                             self.u_input: sample_u,
                                             self.jx_input: jx})

        return tf_loss, tf_lx, tf_lu, tf_lxx, tf_luu, tf_lux
