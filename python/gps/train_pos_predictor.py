import numpy as np
import tensorflow as tf
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

from gps.pose_prediction.tf_model import convolutional_network



prefix = 'pretraining_data/'


img_data = np.load(prefix+'image_data.npy')
pose_labels = np.load(prefix+'pos_labels_data.npy')
num_data = img_data.shape[0]
img_h =img_data.shape[1]
img_w =img_data.shape[2]
img_channels =img_data.shape[3]

iterations = 1000
batch_size = 128

session = tf.Session()

model = convolutional_network(img_data.shape[1:], pose_labels.shape[1], batch_size=batch_size)
solver_op = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9).minimize(model['loss'])
op = tf.initialize_all_variables()
session.run(op)
indices = np.arange(num_data)
np.random.shuffle(indices)

feed_dict = {}
for itr in xrange(iterations):
    batch_index_indices = np.mod(np.arange((itr * batch_size), (itr * batch_size)+batch_size), num_data)
    batch_indices = indices[batch_index_indices]
    feed_dict[model['input']] = img_data[batch_indices]
    feed_dict[model['pose_labels']] = pose_labels[batch_indices]
    train_loss = session.run([model['loss'], solver_op], feed_dict)
    if itr % 100 == 0 :
        print "itr", itr, "train_loss", train_loss
