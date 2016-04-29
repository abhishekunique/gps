import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

from gps.pose_prediction.tf_model import convolutional_network, convolutional_network_dc

domain_confusion = True

prefix = 'pretraining_data/'


img_data = np.load(prefix+'image_data.npy')
pose_labels = np.load(prefix+'pos_labels.npy')
robot_labels = np.load(prefix+'robot_labels.npy')
num_robots= robot_labels.max()+1

num_data = img_data.shape[0]
robot_onehots = np.zeros((num_data, num_robots))
robot_onehots[np.arange(num_data),robot_labels] = 1


img_h =img_data.shape[1]
img_w =img_data.shape[2]
img_channels =img_data.shape[3]

iterations = 10000
batch_size = 128

session = tf.Session()

if domain_confusion:
    model = convolutional_network_dc(img_data.shape[1:], pose_labels.shape[1], 
                                     num_robots=2, batch_size=batch_size, dc_weight=1)
    dc_solver_op = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9).minimize(
        model['loss'],var_list= model['dc_vars'] )
    var_list = tf.trainable_variables()
    new_var_list = []
    for var in var_list:
        if var not in model['dc_vars']:
            new_var_list.append(var)
    var_list = new_var_list
else:
    model = convolutional_network(img_data.shape[1:], pose_labels.shape[1], batch_size=batch_size)
    var_list = tf.trainable_variables()

solver_op = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9).minimize(model['loss'],
                                                                           var_list=var_list)
op = tf.initialize_all_variables()
session.run(op)
indices = np.arange(num_data)
np.random.shuffle(indices)

def single_img_with_fp(obs, itr, im_number):
    fps = model['feature_points']
    feed_dict = {}
    feed_dict[model['input']] = np.array([obs])
    fps_vals = session.run(fps, feed_dict)[0]
    fx_v = fps_vals[::2].astype(int)
    fy_v = fps_vals[1::2].astype(int)
    img = obs
    img[fx_v, fy_v, :] = 0
    img[fx_v, fy_v, 2] = 254
    mpimg.imsave('feature_point_test'+str(itr) + '_' +str(im_number)+'.png', img)
    return img


feed_dict = {}

for itr in xrange(iterations):
    batch_index_indices = np.mod(np.arange((itr * batch_size), (itr * batch_size)+batch_size), num_data)
    batch_indices = indices[batch_index_indices]
    feed_dict[model['input']] = img_data[batch_indices]
    feed_dict[model['pose_labels']] = pose_labels[batch_indices]
    feed_dict[model['robot_onehots']] = robot_onehots[batch_indices]
    train_loss = session.run([model['loss'],model['pose_loss'], solver_op], feed_dict)
    dc_loss = session.run([model['dc_loss'], model['dc_output'], dc_solver_op], feed_dict)
    if itr % 100 == 0 :
        pose_error = train_loss[1]/float(batch_size)
        dc_acc = dc_loss[1][np.arange(batch_size), robot_labels[batch_indices]].sum()/batch_size
        # dc_acc = dc_loss[1][np.arange(batch_size)][robot_labels[batch_indices]].sum()/float(batch_size)
        print "itr", itr, "train_loss", train_loss[0]/float(batch_size), "dc_loss", dc_loss[0]/float(batch_size), "dc_acc", dc_acc, "pose error", pose_error

        for im in range(10):
            single_img_with_fp(img_data[batch_indices[im],:,:,:], itr, im)


