import time
import warnings
import os, sys
import random

random.seed(0)
import numpy as np
from scipy.io import loadmat
import open3d as o3d
from open3d import geometry, utility
import pandas as pd

from zbb_normal_saveall import normal_save                 
from zbb_generate_train8192 import excel_writer_down_mul    
from zbb_generate_loss10240 import mutiprocess                 


root = "/media/zbb/1bf51c0b-09ea-4f74-889a-ca3b043f57a3/data/KittiOdometry/velo/dataset"
outpath = "/media/zbb/1bf51c0b-09ea-4f74-889a-ca3b043f57a3/data/KittiOdometry_zbb"

if __name__ == '__main__':

    # datalist = ['kitti_noise', 'kitti', 'Radiate', 'Boreas']
    datalist = ['kitti']

    for seq in range(0, 11):
        seq = '{0:02d}'.format(int(seq))

        print(seq)
        print(outpath)

        if datalist[0] == 'kitti_noise':
            lidar_path = os.path.join(root, seq)
            save_root = os.path.join(outpath, 'sequences', seq)
        elif datalist[0] == 'kitti':
            lidar_path = os.path.join(root, 'sequences', seq, 'velodyne')
            save_root = os.path.join(outpath, 'sequences', seq)

        normal_path = os.path.join(save_root, "normal_r4k50")
        mat_path = os.path.join(save_root, "pointground")

        # # generate loss 10240
        normal_save(lidar_path, normal_path, seq, datalist[0])
        mutiprocess([["org"]], [2], 1, lidar_path, normal_path, mat_path, save_root, datalist[0])

        # # generate train 8192
        excel_writer_down_mul(lidar_path, mat_path, save_root, datalist[0], "cartesian", [0.3], 8192, "ucuzdd", True, 0.5, -10)
