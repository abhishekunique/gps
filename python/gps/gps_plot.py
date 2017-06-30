""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
from os import listdir
from os.path import isfile, join
import sys
import copy
import argparse
import threading
import time
import matplotlib.pyplot as plt
# Add gps/python to path so that imports work.
sys.path.append('/home/abhigupta/gps/python')
# from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE
import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == "__main__":
	data_logger = DataLogger()
	data_dir = '/home/abhigupta/gps/experiments/mjc_test_actop/data_files/'
	onlyfiles = []
	for itr in range(8):
		onlyfiles.append('pol_sample_itr_%02d_rn_00.pkl' % itr)

	targets = [np.asarray([0., 0., -1.7]),np.asarray([0.7, 0., 0.]), np.asarray([0.3, 0.0, 0.5]),
              np.asarray([0.7, 0., -1.]), np.asarray([.5, 0.0, 0.3]),np.asarray([.7, 0.0, -0.3]),
#              np.array([-0.8, 0.0, 0.5]),np.array([-0.3, 0.0, -0.8])]
               np.array([0., 0., -1.3]), np.array([0.5, 0., -1])]
	targets = [(t + np.array([0.8, 0.0, 0.5])) for t in targets]

	dists = []
	for f in onlyfiles:
		traj_sample_list = data_logger.unpickle(data_dir + f)
		dist_list = [0]*len(traj_sample_list)
		for i, cond_list in enumerate(traj_sample_list):
			dist = 0
			for sample in cond_list._samples:
				sample_ee = sample._data[END_EFFECTOR_POINTS][-20:]
				dist += np.mean(np.linalg.norm(sample_ee - targets[i], axis=1))
			dist_mean = dist/float(len(cond_list._samples))
			dist_list[i] = dist_mean
		dists.append(dist_list)
	import IPython
	IPython.embed()
	np.save("mjc_multirobot_reach_sharedconv", cond_dists)
	# for cond in range(4):
	# 	dist_to_plot = [dists[i][cond] for i in range(len(dists))]
	# 	fig1 = plt.figure()
	# 	ax1 = fig1.add_subplot(111)
	# 	ax1.plot(range(len(dists)), dist_to_plot)
	# plt.show()
