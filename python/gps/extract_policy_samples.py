import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import IPython
import numpy as np
# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.proto.gps_pb2 import ACTION, RGB_IMAGE, END_EFFECTOR_POINTS


all_offsets = [np.asarray([0., 0., -1.7]),np.asarray([0.7, 0., 0.]), np.asarray([0.3, 0.0, 0.5]),
              np.asarray([0.7, 0., -1.]), np.asarray([.5, 0.0, 0.3]),np.asarray([.7, 0.0, -0.3]),
#              np.array([-0.8, 0.0, 0.5]),np.array([-0.3, 0.0, -0.8])]
               np.array([0., 0., -1.3]), np.array([0.5, 0., -1])]

offsets = [
    [np.array([1.0, 0.0, 0.45]),np.array([0.4, 0.0, 0.45])],
    [np.array([1.0, 0.0, -0.5]),np.array([0.6, 0.0, -0.5])],
    [np.array([0.6, 0.0, 0.65]),np.array([0.2, 0.0, 0.65])],
    [np.array([0.8, 0.0, -0.65]),np.array([0.6, 0.0, -0.65])],
    
    [np.array([0.8, 0.0, 0.5]),np.array([0.3, 0.0, 0.5])],
    [np.array([0.8, 0.0, -0.5]),np.array([0.4, 0.0, -0.5])],
    [np.array([0.7, 0.0, 0.6]),np.array([0.4, 0.0, 0.6])],
    [np.array([0.75, 0.0, -0.55]),np.array([0.45, 0.0, -0.55])],
]

# if __name__ == "__main__":
#     data_logger = DataLogger()
#     data_dir = '/home/abhigupta/gps/experiments/fc_4link_push/data_files/'
#     onlyfiles = []
#     for itr in range(9):
#         onlyfiles.append('pol_sample_itr_%02d_rn_00.pkl' % itr)
#         targets = [offsets[i][1] for i in range(len(offsets))]
#         dists = []
#     print onlyfiles
#     print "------------"
#     for f in onlyfiles:
#         traj_sample_list = data_logger.unpickle(data_dir + f)
#         print data_dir + f
#         dist_list = [0]*len(traj_sample_list)
#         for i, cond_list in enumerate(traj_sample_list):
#             dist = 0
#             for sample in cond_list._samples:
#                 sample_ee = sample._data[END_EFFECTOR_POINTS][-20:,3:6]
#                 dist += np.mean(np.linalg.norm(sample_ee - targets[i]-np.array([0.8, 0.0, 0.5]), axis=1))
#             dist_mean = dist/float(len(cond_list._samples))
#             dist_list[i] = dist_mean
#         dists.append(dist_list)
#     cond_dists = np.asarray(dists)
#     np.save(data_dir+"fc_3link_push_base", cond_dists)

if __name__ == "__main__":
    data_logger = DataLogger()
    data_dir = '/home/abhigupta/gps/experiments/mjc_multirobot_reach_domain_confusion/data_files/'
    # data_dir = '/home/coline/abhishek_gps/gps/experiments/mjc_multirobot_reach_domain_confusion/data_files/'
    onlyfiles = []
    for itr in range(9):
        onlyfiles.append('pol_sample_ee_itr_%02d_rn_00.pkl' % itr)
        targets = all_offsets#[offsets[i][1] for i in range(len(offsets))]
        print targets
        dists = []
    for f in onlyfiles:
        traj_sample_list = data_logger.unpickle(data_dir + f)
        print data_dir + f
        dist_list = [0]*len(traj_sample_list)
        for i, cond_list in enumerate(traj_sample_list):
            dist = 0
            for sample in cond_list:
                sample_ee = sample[-20:, :]
                print "ee_shape", sample_ee.shape
                dist += np.mean(np.linalg.norm(sample_ee - targets[i]- np.array([0.8, 0.0, 0.5]), axis=1))
            dist_mean = dist/float(len(cond_list))
            print "mean", dist_mean
            dist_list[i] = dist_mean
        print dist_list
        dists.append(dist_list)
    cond_dists = np.asarray(dists)
    np.save(data_dir+"/mjc_3link_reach_dc", cond_dists)

    # np.save("mjc_multirobot_reach_domain_confusion_r1", cond_dists)
