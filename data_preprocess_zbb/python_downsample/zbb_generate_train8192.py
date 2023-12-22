import time
# import torch
import warnings
# import pandas as pd
import os, sys
import random

random.seed(0)

import numpy as np

from scipy.io import loadmat
import open3d as o3d
from open3d import geometry, utility
import pandas as pd



warnings.filterwarnings("ignore")

datasize = np.array([4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201,
                     921, 1061, 3281, 631, 1091, 1731, 491, 1081, 4981, 831, 2721,
                     5671, 1798, 1799], dtype=np.int32)


def checklimit(data, min, max):
    return np.sum(data < min) + np.sum(data > max) > 0


def sumpcd_prepross(points):
    def normal_process(points, normal):            #让normal统一朝向   不然结果输出来，可能相邻点差180度
        toO = -1 * points                         #向所有点。这等同于绕原点旋转点180度
        result = np.sum(toO * normal, axis=1) / (
                np.linalg.norm(toO, 2, axis=1) * np.linalg.norm(normal, 2, axis=1))   #用点积公式计算反向点与法向量之间的角度的余弦值。除以范数的乘积是为了规范化结果。
        direct = np.sign(result)                 #确定每个点相对于法向量的方向。sign函数对于负数返回-1，对于正数返回1，对于零返回0
        direct[direct == 0] = 1                  #改变与法向量角度为零的点的方向为1。这意味着这些点被认为与法向量同方向
        return direct[:, None] * normal          #将方向乘以法向量。这将根据每个点的方向对法向量进行缩放。
    def get_pcd_from_numpy(pcd_np):
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(pcd_np[:, :3])
        return pcd
    sumpcd = get_pcd_from_numpy(points)
    sumpcd.estimate_normals(search_param=geometry.KDTreeSearchParamHybrid(radius=4, max_nn=50),
                            fast_normal_computation=False)
    sumpcd.normalize_normals()
    sumnor = np.asarray(sumpcd.normals)
    sumnor = normal_process(points, sumnor)
    return sumnor




def cartesian(points, normal, deep, nvoxel, priority):
    if deep == "sparse":
        ximin = -70
        xrate = 0.4
        ximax = 70
        yimin = -40
        yrate = 0.4
        yimax = 40
        zimin = -3
        zrate = 0.5
        zimax = 2
    elif deep == "selfvoxello":
        ximin = -68.8
        xrate = 0.1
        ximax = 68.8
        yimin = -40
        yrate = 0.1
        yimax = 40
        zimin = -4
        zrate = 0.2
        zimax = 4
    elif deep == "voxelnet":
        ximin = -70.4
        xrate = 0.2
        ximax = 70.4
        yimin = -40
        yrate = 0.2
        yimax = 40
        zimin = -3
        zrate = 0.4
        zimax = 1
    else:
        if len(deep) == 3:
            xrate, yrate, zrate = deep
        else:  
            xrate = yrate = zrate = deep[0]
        ximin = -70
        ximax = 70
        yimin = -40
        yimax = 40
        zimin = -3
        zimax = 2
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    depth = np.linalg.norm(points[:, :2], 2, axis=1)

    xmax = int((ximax - ximin) / xrate)
    ymax = int((yimax - yimin) / yrate)
    zmax = int((zimax - zimin) / zrate)

    proj_x = np.floor((scan_x - ximin) / xrate).astype(np.int32)
    proj_y = np.floor((scan_y - yimin) / yrate).astype(np.int32)
    proj_z = np.floor((scan_z - zimin) / zrate).astype(np.int32)

    xcheck = np.logical_and(0 <= proj_x, proj_x < xmax)
    ycheck = np.logical_and(0 <= proj_y, proj_y < ymax)
    zcheck = np.logical_and(0 <= proj_z, proj_z < zmax)
    check = np.logical_and(np.logical_and(xcheck, ycheck), zcheck)
    proj_x = proj_x[check]
    proj_y = proj_y[check]
    proj_z = proj_z[check]
    depth = depth[check]
    #################将点云数据映射到体素空间，即将每个点的X、Y、Z坐标转换为对应的体素索引（proj_x, proj_y, proj_z）##################


    points = points[check]    
    normal = normal[check] 
    voxelall = proj_z * 1000000 + proj_y * 1000 + proj_x                  
    pointkeep = voxelall.shape[0]
    pointsid = np.lexsort((depth, voxelall))                           
    remission = np.zeros_like(depth)                                                                        
    sortdata = np.concatenate((points, remission[:, None], normal, depth[:, None], voxelall[:, None]), axis=1)[pointsid]   
    
    id = np.where(np.diff(sortdata[:, -1]))[0]  
    sortlist = np.split(sortdata, id + 1)       
   

    value, firstindex, count = np.unique(voxelall[pointsid], return_index=True, return_counts=True)
   
    depthchoose = np.round(depth[pointsid][firstindex]).astype(np.int32)   
    if priority == "ucdduz":
        voxelchoose = np.argsort(count * 10000 + (100 - depthchoose) * 100 + value // 1000000)[::-1][:nvoxel]
    elif priority == "ucuzdd":
        voxelchoose = np.argsort(count * 10000 + value // 1000000 * 100 + (100 - depthchoose))[::-1][:nvoxel]  
    else:
        raise NotImplementedError()

    voxelmsg = []  # count+depid+yid+xid+vercen3+norcen3
    for i in range(voxelchoose.shape[0]):
        nowpointsmsg = sortlist[voxelchoose[i]]            
        cendata = np.mean(nowpointsmsg[:, :7], axis=0) 
        nowpointsmsg[:, -1] = range(nowpointsmsg.shape[0])  # TODO
        nowpointsmsg[:, -2] = i  # TODO
        vercen = cendata[:3]
        remcen = cendata[3:4]
        normal = cendata[4:7]        #zbb

        voxelnum = value[voxelchoose[i]]
        proj_x_i = voxelnum % 1000
        proj_y_i = voxelnum % 1000000 // 1000
        proj_z_i = voxelnum // 1000000
        nowvoxelmsg = np.array([count[voxelchoose[i]], proj_z_i, proj_y_i, proj_x_i]+ vercen.tolist() + normal.tolist() + remcen.tolist())    
        #insert the normal between vercen and remcen
        voxelmsg.append(nowvoxelmsg[None, :])
    voxelmsg = np.concatenate(voxelmsg, axis=0) 

  
    voxelsize = voxelmsg.shape[0]
    if voxelsize < nvoxel:
        distance = voxelsize / (nvoxel - voxelsize)
        get = [distance * i for i in range(0, nvoxel - voxelsize)]
        get = np.round(np.array(get)).astype(np.int32)
        get = np.clip(get, 0, voxelmsg.shape[0] - 1)

        voxelmsg = np.concatenate((voxelmsg, voxelmsg[get]), axis=0)    
        pass

    return voxelmsg[:, 4:10], voxelsize




def mul_saver(i, name, deeptype, lidarpath, savepath, matpath, nvoxel, priority, compress, dropprate, minz, datalist):

    bin = '{0:06d}'.format(i)

    if datalist in ['kitti',]:
            scan = np.fromfile(os.path.join(lidarpath, bin + ".bin"), dtype=np.float32)
            scan = scan.reshape((-1, 4))[:, :3]

    elif datalist in ['kitti_noise']:
            scan = np.load(os.path.join(lidarpath, bin + ".npy")) 
            scan = scan.reshape((-1, 6))[:, :3]

    elif datalist in ['Radiate']:
            scan_path = os.path.join(lidarpath, bin + ".csv")
            scan = pd.read_csv(scan_path, header=None).to_numpy()[:, :3]


    ###################################select no ground points ######################################## zbb
    normal = sumpcd_prepross(scan) 

    flag = loadmat(os.path.join(matpath, bin + ".mat"))
    flag_other = flag["other"]
    other = flag_other.squeeze(1) - 1
    scan = scan[other]

    normal = normal[other] 

    scan_sort = np.argsort(scan[:, -1])[int(scan.shape[0] * dropprate):]  
    scan = scan[scan_sort]
    normal = normal[scan_sort]
    z_mask = scan[:, -1] >= minz
    scan = scan[z_mask]
    normal = normal[z_mask]


    t1 = time.time()
    voxelmsg, voxellen = cartesian(scan, normal, deeptype, nvoxel, priority)      #train 8192
    t2 = time.time()
    savename = bin + ".npz"
    savename_path = savepath
    if not os.path.exists(savename_path):
        os.makedirs(savename_path)
    if compress:
        np.savez_compressed(os.path.join(savename_path, savename), voxelmsg=voxelmsg, voxellen=voxellen)

    print(savename, voxellen, dropprate, t2 - t1)


def excel_writer_down_mul(lidarpath, matpath, save_root, datalist, name, deeptype, nvoxel, priority, compress, dropprate=0.4, minz=-1.5):


    if isinstance(deeptype, list):
        savepath = os.path.join(save_root, "voxeldata", 'mean_normal_' + name + "_" + str(deeptype[0]) + "_" + str(nvoxel) + "_" + str(int(dropprate * 100)) + "_" + str(minz) + "_" + priority)
    else:
        savepath = os.path.join(save_root, "voxeldata", 'mean_normal_' + name + "_" + deeptype + "_" + str(nvoxel) + "_" + str(int(dropprate * 100)) + "_" + str(minz))
    if compress:
        savepath += "_com_dense_flowonly"
    else:
        savepath += "_dense_flowonly"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    seqlen = len(os.listdir(lidarpath))

    if datalist in ['kitti_noise', 'kitti',]:
        start = 0
        end = seqlen
    
    elif datalist in ['Radiate']:
        start = 1
        end = seqlen+1

    for i in range(start, end):
        mul_saver(i, name, deeptype, lidarpath, savepath, matpath, nvoxel, priority, compress, dropprate, minz, datalist)


def mul_viewer2(i, savepath, nvoxel):
    bin = '{0:06d}'.format(i)
    voxel = np.load(os.path.join(savepath, bin + ".npz"))
    voxelpoints = voxel["voxelpoints"]
    voxelmsg=voxel["voxelmsg"]
    voxellen=voxel["voxellen"]
    maxnpoints = int(voxelmsg[:, 0].max())
    newvoxelpoints = np.zeros_like(voxelmsg)

    cnterpoints = voxelmsg[:voxellen, None, 4:]
    cnterpoints = np.repeat(cnterpoints, voxelpoints.shape[1], axis=1)
    newvoxelpoints[:, :, 6] = cnterpoints



if __name__ == '__main__':
    
    excel_writer_down_mul("cartesian", [0.3], 8192, "ucuzdd", True, 0.5, -10, lidarpath, matpath, save_root, datalist)
    