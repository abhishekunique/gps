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
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == "__main__":
	data_logger = DataLogger()
	data_dir = '/home/abhigupta/CS280_finalproj_data/rs_data/4link_push_baseline/data_files/'
	onlyfiles = []
	for itr in range(23):
		onlyfiles.append('traj_sample_itr_%02d_00.pkl' % itr)
	target = []
	dists = []
	for f in onlyfiles:
		traj_sample_list = data_logger.unpickle(data_dir + f)
		import IPython
		IPython.embed()
		dist = 0
		for sample in traj_sample_list._samples:
			sample_ee = sample._data[END_EFFECTOR_POINTS]
			T = sample_ee.shape[0]
			for t in range(T):
				dist += (sample_ee[t] - target).T.dot(sample_ee[t] - target)
		dists.append(dist)
	plt.plot(range(len(onlyfiles)), dists)
	plt.show()



