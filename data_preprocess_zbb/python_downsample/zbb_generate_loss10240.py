
import os
from scipy.spatial.transform import Rotation
from scipy import spatial
import random
random.seed(0)
import warnings
import pandas as pd

from open3d import utility, geometry
from scipy.io import loadmat
import numpy as np
import time


# opt = "A:"
warnings.filterwarnings("ignore")


datasize = np.array([4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201,
                     921, 1061, 3281, 631, 1091, 1731, 491, 1081, 4981, 831, 2721,
                     -1, -1, -1, -1, -1, -1, -1, -1,
                     3817, 6103, -1, 8278, 11424, 9248, 5351, 4705, 5233, -1,
                     6443, 20801, 4701, 11846, 16596, 7538, 14014, 5677, 10903, 9765, 9868, 41529], dtype=np.int32)


def avg_sample(points, npoints):
    a = random.sample(range(points.shape[0]), npoints)
    idx = np.array(a)
    idx.sort()
    return idx


def make_pointcloud_with_normal(points, normal=None):
    point_cloud = geometry.PointCloud()
    point_cloud.points = utility.Vector3dVector(points)
    if normal is not None:
        point_cloud.normals = utility.Vector3dVector(normal)
    return point_cloud


def downsample(points, voxelsize=-1, normals=None):
    p0 = make_pointcloud_with_normal(points, normals)
    if voxelsize > 0:
        p0 = p0.voxel_down_sample(voxelsize)
    if normals is not None:
        return np.concatenate([np.array(p0.points),  np.array(p0.normals)], axis=1)
    else:
        return np.array(p0.points)


def noground_n(points, normal, flag, deep, voxelsize, npoints, maxnormal):
    # get ground points
    other = flag["other"].squeeze(1) - 1
    points = points[other, :]
    normal = normal[other, :]

    other2 = np.abs(normal[:, 2]) <= maxnormal
    points = points[other2, :]
    normal = normal[other2, :]

    oth_downver = downsample(points, voxelsize)
    # testflag = False
    while oth_downver.shape[0] > npoints:
        voxelsize += 0.01
        # testflag = True
        oth_downver = downsample(points, voxelsize)


    while oth_downver.shape[0] < npoints and voxelsize > 0.02:
        voxelsize -= 0.01
        oth_downver = downsample(points, voxelsize)

    if voxelsize <= 0.02:
        return downsample(points, 0, normal), 0


    oth_downver = downsample(points, voxelsize, normal)
    if oth_downver.shape[0] > npoints:
        fps_idx = avg_sample(oth_downver, npoints)
        return oth_downver[fps_idx, :], voxelsize
    else:
        return oth_downver, voxelsize


def spherical_down(points, deep):
    if deep == "deepplusplus":
        depmin = 2
        depsize = 64
    elif deep == "deepplus":
        depmin = 2
        depsize = 48
    elif deep == "deep":
        depmin = 2
        depsize = 32
    elif deep == "normal":
        depmin = 3
        depsize = 16


    depth = np.linalg.norm(points, 2, axis=1)
    depthindex = (depth.round() - depmin).astype(np.int32)
    dcheck = np.logical_and(0 <= depthindex, depthindex < depsize)

    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]


    yaw = np.arctan2(scan_y, scan_x) / np.pi * 180
    pitch = np.arcsin(scan_z / depth) / np.pi * 180


    proj_x = (180 - yaw) / 360

    proj_y = (4 - pitch) / (4 + 8.72)
    proj_y2 = 1.0 + (-8.72 - pitch) / (-8.72 + 24.8)
    proj_y[pitch < -8.72] = proj_y2[pitch < -8.72]
    proj_y /= 2


    xcheck = np.logical_and(0 <= proj_x, proj_x < 1)
    ycheck = np.logical_and(0 <= proj_y, proj_y < 1)
    check = np.logical_and(np.logical_and(xcheck, ycheck), dcheck)

    return check


def pointsave(deeptype, voxelsize, npoints, maxnormal, lidar_path, normal_path, mat_path, save_root, datalist):

    # for seq in range(0, 11):
    seqlen = len(os.listdir(lidar_path))

    if maxnormal == 1:
        savepath = os.path.join(save_root, "npointsloss_", str(npoints))
    else:
        savepath = os.path.join(save_root,  "npointsloss_", str(npoints) + "_" + str(int(maxnormal * 100)))

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    mul_saver(seqlen, mat_path, lidar_path, savepath, normal_path, deeptype, voxelsize, npoints, maxnormal, datalist)


def mul_saver(seqlen, mat_path, lidarpath, savepath, normalpath, deeptype, voxelsize, npoints, maxnormal, datalist):

    if datalist in ['kitti_noise', 'kitti',]:
        start = 0
        end = seqlen
    
    elif datalist in ['Radiate']:
        start = 1
        end = seqlen+1

    for i in range(start, end):
        t1 = time.time()
        bin = '{0:06d}'.format(i)

        if datalist in ['kitti',]:
            scan = np.fromfile(os.path.join(lidarpath, bin + ".bin"), dtype=np.float32)
            points = scan.reshape((-1, 4))[:, :3]

        elif datalist == 'kitti_noise':
            scan = np.load(os.path.join(lidarpath, bin + ".npy")) 
            points = scan.reshape((-1, 6))[:, :3]

        if datalist in ['kitti', 'Boreas']:
            normal = np.load(os.path.join(normalpath, bin + ".bin.npy"))
        elif datalist in ['kitti_noise']:
            normal = np.load(os.path.join(normalpath, bin + ".npy"))

        flag = loadmat(os.path.join(mat_path, bin + ".mat"))
        points, voxelnewsize = noground_n(points, normal, flag, deeptype, voxelsize=voxelsize, npoints=npoints, maxnormal=maxnormal)
        #[10240, 6] 0.33999999
        
        if points.shape[0] < npoints:
            psize = points.shape[0]
            distance = psize / (npoints - psize)
            get = [distance * i for i in range(0, npoints - psize)]
            get = np.round(np.array(get)).astype(np.int)
            get = np.clip(get, 0, points.shape[0] - 1)

            points = np.concatenate((points, points[get]), axis=0)

        np.save(os.path.join(savepath, bin + ".npy"), points)
        t2 = time.time()
        print(bin, deeptype, "%.2f" % voxelnewsize, points.shape[0], t2 - t1)


def mutiprocess(deeplist, npointslist, maxnormal, lidar_path, normal_path, mat_path, save_root, datalist):
    for j in npointslist:
        for deeptype in deeplist:
            index = j
            voxelsize = 0.02 * (20 - index)
            npoints = 1024 * (8 + index)
            pointsave(deeptype, voxelsize, npoints, maxnormal, lidar_path, normal_path, mat_path, save_root, datalist)



# if __name__ == '__main__':

#     deeplist = ["org"]
#     mutiprocess(deeplist, [2], 1, lidar_path, normal_path, mat_path, save_root, datalist)


