import tensorflow as tf
import numpy as np

def check_list_and_convert(the_object):
    if isinstance(the_object, list):
        return the_object
    return [the_object]


class TfMap:
    """ a container for inputs, outputs, and loss in a tf graph. This object exists only
    to make well-defined the tf inputs, outputs, and losses used in the policy_opt_tf class."""

    def __init__(self, input_tensor, target_output_tensor, precision_tensor, output_op, loss_op, dc_onehot=None, 
                 pretraining_tgt=None, pretraining_loss=None, feature_points=None):
        self.input_tensor = input_tensor
        self.target_output_tensor = target_output_tensor
        self.precision_tensor = precision_tensor
        self.output_op = output_op
        self.loss_op = loss_op
        self.dc_onehot = dc_onehot
        self.pretraining_loss_op = pretraining_loss
        self.pretraining_tgt = pretraining_tgt
        self.feature_points = feature_points

    @classmethod
    def init_from_lists(cls, inputs, outputs, loss, dc_onehots=None, pretraining_tgt=None, pretraining_loss=None,
                        feature_points=None):
        inputs = check_list_and_convert(inputs)
        outputs = check_list_and_convert(outputs)
        loss = check_list_and_convert(loss)
        if dc_onehots:
            dc_onehots = check_list_and_convert(dc_onehots)
        else:
            dc_onehots = [None]
        if len(inputs) < 3:  # pad for the constructor if needed.
            inputs += [None]*(3 - len(inputs))
        return cls(inputs[0], inputs[1], inputs[2], outputs[0], loss[0], dc_onehot=dc_onehots[0], 
                   pretraining_tgt=pretraining_tgt, pretraining_loss =pretraining_loss,
                   feature_points=feature_points)

    def get_input_tensor(self):
        return self.input_tensor

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def get_target_output_tensor(self):
        return self.target_output_tensor

    def set_target_output_tensor(self, target_output_tensor):
        self.target_output_tensor = target_output_tensor

    def get_precision_tensor(self):
        return self.precision_tensor

    def set_precision_tensor(self, precision_tensor):
        self.precision_tensor = precision_tensor

    def get_output_op(self):
        return self.output_op

    def set_output_op(self, output_op):
        self.output_op = output_op

    def get_loss_op(self):
        return self.loss_op

    def set_loss_op(self, loss_op):
        self.loss_op = loss_op

    def get_dc_onehot(self):
        return self.dc_onehot

    def set_dc_onehot(self, dc_onehot):
        self.dc_onehot = dc_onehot



class TfSolver:
    """ A container for holding solver hyperparams in tensorflow. Used to execute backwards pass. """
    def __init__(self, loss_scalar, solver_name='adam', base_lr=None, lr_policy=None,
                 momentum=None, weight_decay=None, robot_number=0, fc_vars=None, 
                 last_conv_vars=None, dc_vars=None, pretraining_loss=None):
        self.base_lr = base_lr
        self.lr_policy = lr_policy
        self.momentum = momentum
        self.solver_name = solver_name
        self.loss_scalar = loss_scalar
        if self.lr_policy != 'fixed':
            raise NotImplementedError('learning rate policies other than fixed are not implemented')

        self.weight_decay = weight_decay
        if weight_decay is not None:
            trainable_vars = tf.trainable_variables()
            loss_with_reg = self.loss_scalar
            for var in trainable_vars:
                if dc_vars is None or var not in dc_vars:
                    loss_with_reg += self.weight_decay*tf.nn.l2_loss(var)
            self.loss_scalar = loss_with_reg
        if dc_vars is None:
            self.solver_op = self.get_solver_op()
        else:
            var_list = tf.trainable_variables()
            new_var_list = []
            for var in var_list:
                if var not in dc_vars:
                    new_var_list.append(var)
            self.solver_op = self.get_solver_op(var_list=new_var_list)
        if fc_vars is not None:
            self.fc_vars = fc_vars
            self.last_conv_vars = last_conv_vars
            self.fc_solver_op = self.get_solver_op(var_list=fc_vars)
        if pretraining_loss is not None:
            self.pretraining_loss = pretraining_loss
            self.pretraining_solver_op = self.get_solver_op(loss=self.pretraining_loss)
        self.trainable_variables = tf.trainable_variables()

    def get_solver_op(self, var_list=None, loss=None):
        solver_string = self.solver_name.lower()
        if var_list is None:
            var_list = tf.trainable_variables()
        if loss is None:
            loss = self.loss_scalar
        if solver_string == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.base_lr,
                                          beta1=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.base_lr,
                                             decay=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.base_lr,
                                              momentum=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.base_lr,
                                             initial_accumulator_value=self.momentum).minimize(loss, var_list=var_list)
        elif solver_string == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.base_lr).minimize(loss, var_list=var_list)
        else:
            raise NotImplementedError("Please select a valid optimizer.")

    def get_last_conv_values(self, sess, feed_dict, num_values, batch_size):
        i = 0
        values = []
        while i < num_values:
            batch_dict = {}
            start = i
            end = min(i+batch_size, num_values)
            for k,v in feed_dict.iteritems():
                batch_dict[k] = v[start:end]
            batch_vals = sess.run(self.last_conv_vars, batch_dict)
            values.append(batch_vals)
            i = end
        final_values = [None for v in self.last_conv_vars]
        i = 0

        for v in range(len(self.last_conv_vars)):
            final_values[v] = np.concatenate([values[i][v] for i in range(len(values))])
        return final_values

    def __call__(self, feed_dict, sess, device_string="/cpu:0", use_fc_solver=False):
        with tf.device(device_string):
            if use_fc_solver:
                loss = sess.run([self.loss_scalar, self.fc_solver_op], feed_dict)
            else:
                loss = sess.run([self.loss_scalar, self.solver_op], feed_dict)
            return loss[0]

class DcTfSolver:
    """ A container for holding solver hyperparams in tensorflow. Used to execute backwards pass. """
    def __init__(self, loss_scalar, solver_name='adam', base_lr=None, lr_policy=None,
                 momentum=None, weight_decay=None, robot_number=0, dc_vars=None):
        self.base_lr = base_lr
        self.lr_policy = lr_policy
        self.momentum = momentum
        self.solver_name = solver_name
        self.loss_scalar = loss_scalar
        if self.lr_policy != 'fixed':
            raise NotImplementedError('learning rate policies other than fixed are not implemented')

        self.weight_decay = weight_decay
        if weight_decay is not None:
            trainable_vars = dc_vars
            loss_with_reg = self.loss_scalar
            for var in trainable_vars:
                loss_with_reg += self.weight_decay*tf.nn.l2_loss(var)
                self.loss_scalar = loss_with_reg

        self.solver_op = self.get_solver_op(var_list=dc_vars)
        self.trainable_variables = tf.trainable_variables()

    def get_solver_op(self, var_list=None):
        solver_string = self.solver_name.lower()

        if solver_string == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.base_lr,
                                          beta1=self.momentum).minimize(self.loss_scalar, var_list=var_list)
        elif solver_string == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.base_lr,
                                             decay=self.momentum).minimize(self.loss_scalar, var_list=var_list)
        elif solver_string == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.base_lr,
                                              momentum=self.momentum).minimize(self.loss_scalar, var_list=var_list)
        elif solver_string == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.base_lr,
                                             initial_accumulator_value=self.momentum).minimize(self.loss_scalar, var_list=var_list)
        elif solver_string == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.base_lr).minimize(self.loss_scalar, var_list=var_list)
        else:
            raise NotImplementedError("Please select a valid optimizer.")

    def __call__(self, feed_dict, sess, device_string="/cpu:0", use_fc_solver=False):
        with tf.device(device_string):
            loss = sess.run([self.loss_scalar, self.solver_op], feed_dict)
            return loss[0]
