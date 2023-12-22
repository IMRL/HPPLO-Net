# import multiprocessing
import time
import warnings
import os, sys
from open3d import *
import numpy as np

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def get_pcd_from_numpy(pcd_np):
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(pcd_np[:, :3])
    return pcd


def sumpcd_prepross(points):
    sumpcd = get_pcd_from_numpy(points)
    sumpcd.estimate_normals(search_param=geometry.KDTreeSearchParamHybrid(radius=4, max_nn=50),
                            fast_normal_computation=False)
    sumpcd.normalize_normals()
    sumnor = np.asarray(sumpcd.normals)
    sumpcd = geometry.KDTreeFlann(sumpcd)
    return sumpcd, sumnor


def normal_process(points, normal):
    toO = -1 * points
    result = np.sum(toO * normal, axis=1) / (
            np.linalg.norm(toO, 2, axis=1) * np.linalg.norm(normal, 2, axis=1))
    direct = np.sign(result)
    direct[direct == 0] = 1
    return direct[:, None] * normal


def mul_process(seq, bin, lidar_path, save_path, datalist):

    t0 = time.time()
    scan_path = os.path.join(lidar_path, bin) #kitti
    savepath = os.path.join(save_path, bin)
    if os.path.exists(savepath):
        return

    if datalist == 'kitti':
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))[:, :3]

    elif datalist == 'kitti_noise':
        scan = np.load(scan_path) 
        scan = scan.reshape((-1, 6))[:, :3]

    elif datalist == 'radiate':
        scan = pd.read_csv(scan_path, header=None).to_numpy()[:, :3]

    sumpcd, sumnor = sumpcd_prepross(scan)

    sumnor = normal_process(scan, sumnor)
    np.save(savepath, sumnor)
    t1 = time.time()
    print(seq, bin, t1 - t0)


# seqlist = range(0,11)
def normal_save(lidar_path, save_path, seq, datalist):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    a = os.listdir(lidar_path)
    a.sort()
    for bin in a:
        mul_process(seq, bin, lidar_path, save_path, datalist)