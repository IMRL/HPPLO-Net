import os

import numpy as np
from scipy.spatial.transform import Rotation


# from glovar import timedir, epoch


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            values = line.split()
            # The only non-float64 values in these files are dates, which
            # we don't care about anyway
            try:
                data[values[0]] = np.array(
                    [float(x) for x in values[1:]]).reshape(3, 4)
            except ValueError:
                pass
    return data


def quat_trans2matrix(quat, trans):
    w, x, y, z = quat
    a, b, c = trans

    matrix = np.zeros((4, 4), dtype=np.float64)
    # matrix[0:3, 0:3] = self.quat2mat(quat)

    matrix[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
    matrix[1, 0] = 2 * x * y + 2 * w * z
    matrix[2, 0] = 2 * x * z - 2 * w * y

    matrix[0, 1] = 2 * x * y - 2 * w * z
    matrix[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
    matrix[2, 1] = 2 * y * z + 2 * w * x

    matrix[0, 2] = 2 * x * z + 2 * w * y
    matrix[1, 2] = 2 * y * z - 2 * w * x
    matrix[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2

    matrix[0, 3] = a
    matrix[1, 3] = b
    matrix[2, 3] = c
    matrix[3, 3] = 1

    return matrix


def euler_trans2matrix(euler, trans):
    a, b, c = trans
    r = Rotation.from_euler('xyz', euler, degrees=True)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0:3, 0:3] = r.as_matrix()
    matrix[0, 3] = a
    matrix[1, 3] = b
    matrix[2, 3] = c
    matrix[3, 3] = 1
    return matrix


def rota_trans2matrix(rota, trans):
    a, b, c = trans
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0:3, 0:3] = rota
    matrix[0, 3] = a
    matrix[1, 3] = b
    matrix[2, 3] = c
    matrix[3, 3] = 1
    return matrix


# local:(len, 7)
def local2world(root, local, SQUE, outtxt, iseuler):
    world = np.zeros((local.shape[0] + 1, 12))
    world[0] = np.loadtxt(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt")[0]

    Tr = read_calib_file(root + "data/calib/" + "{0:02d}".format(SQUE) + ".txt")['Tr:']
    TrI = np.zeros((4, 4), dtype=np.float64)
    TrI[:3, :3] = Tr[:, :3].T
    TrI[:3, 3] = -1 * Tr[:, :3].T.dot(Tr[:, 3])
    TrI[3, 3] = 1

    matrix = np.zeros_like(TrI)
    matrix[:-1, :] = world[0].reshape(3, 4)
    matrix[3, 3] = 1
    # a = Rotation.from_euler('xyz', local[:, 3:], degrees=True)
    # aaa = a.as_quat()
    with open(os.path.join(outtxt, "{0:02d}".format(SQUE) + "_pred.txt"), "w+") as f:
        for i in range(local.shape[0]):
            if iseuler:
                add = euler_trans2matrix(local[i, 3:], local[i, :3])
            else:
                add = quat_trans2matrix(local[i, 3:], local[i, :3])
            matrix = np.dot(matrix, add)
            result = Tr.dot(matrix).dot(TrI)
            # result = matrix[:3, :]
            world[i + 1] = result.reshape(-1)
            for j in range(11):
                f.write(("%e " % (world[i + 1, j])))
            f.write(("%e\n" % (world[i + 1, 11])))

    return world


def local2world_fast(root, local, SQUE, outtxt):
    filepath = os.path.join(outtxt, "{0:02d}".format(SQUE) + "_pred.txt")
    world = np.zeros((local.shape[0] + 1, 12))
    # if os.path.exists(filepath):
    #     return np.loadtxt(filepath)
    gtdata = np.loadtxt(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt")
    world[0] = gtdata[0]

    if SQUE >= 30:
        TrI = np.eye(4, dtype=np.float64)
        Tr = np.eye(4, dtype=np.float64)
    else:
        Tr = read_calib_file(root + "data/calib/" + "{0:02d}".format(SQUE) + ".txt")['Tr:']
        TrI = np.zeros((4, 4), dtype=np.float64)
        TrI[:3, :3] = Tr[:, :3].T
        TrI[:3, 3] = -1 * Tr[:, :3].T.dot(Tr[:, 3])
        TrI[3, 3] = 1

    matrix = np.zeros_like(TrI)
    matrix[:-1, :] = world[0].reshape(3, 4)
    matrix[3, 3] = 1

    for i in range(local.shape[0]):
        if local.shape[1] == 6:
            add = euler_trans2matrix(local[i, 3:], local[i, :3])
        elif local.shape[1] == 7:
            add = quat_trans2matrix(local[i, 3:], local[i, :3])
        else:
            # if local.shape[1] == 11:
            #     local = np.concatenate((local, np.ones((local.shape[0], 1))), axis=1)
            add = rota_trans2matrix(local[i, 3:].reshape(3, 3), local[i, :3])
        matrix = np.dot(matrix, add)
        if SQUE >= 30:
            result = matrix[:3, :]
        else:
            result = Tr.dot(matrix).dot(TrI)
        # result = Tr.dot(matrix).dot(T)
        # result = matrix[:3, :]
        world[i + 1] = result.reshape(-1)
    while world.shape[0] < gtdata.shape[0]:
        world = np.concatenate((world, world[-1][None, :]), axis=0)
    np.savetxt(filepath, world, fmt='%.6e')
    return world


def local2world_fast2(root, local, SQUE, outtxt):
    filepath = os.path.join(outtxt, "{0:02d}".format(SQUE) + "_pred.txt")
    world = np.zeros((local.shape[0] + 1, 12))
    if os.path.exists(filepath):
        return np.loadtxt(filepath)
    world[0] = np.loadtxt(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt")[0]

    Tr = read_calib_file(root + "data/calib/" + "{0:02d}".format(SQUE) + ".txt")['Tr:']
    P0 = read_calib_file(root + "data/calib/" + "{0:02d}".format(SQUE) + ".txt")['Tr:']
    one = np.array([0,0,0,1]).reshape(1,4)
    P0 = np.concatenate((P0, one), axis=0)
    Tr = np.concatenate((Tr, one), axis=0)
    # TrI = np.zeros((4, 4), dtype=np.float64)
    # TrI[:3, :3] = Tr[:, :3].T
    # TrI[:3, 3] = -1 * Tr[:, :3].T.dot(Tr[:, 3])
    # TrI[3, 3] = 1

    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[:-1, :] = world[0].reshape(3, 4)
    matrix[3, 3] = 1

    for i in range(local.shape[0]):
        if local.shape[1] == 6:
            add = euler_trans2matrix(local[i, 3:], local[i, :3])
        elif local.shape[1] == 7:
            add = quat_trans2matrix(local[i, 3:], local[i, :3])
        else:
            add = rota_trans2matrix(local[i, 3:].reshape(3, 3), local[i, :3])
        matrix = np.dot(matrix, add)
        result = (P0.dot(Tr)).dot(matrix)
        world[i + 1] = result[:3, :].reshape(-1)
    np.savetxt(filepath, world, fmt='%.6e')
    return world

def localsepara(root, timedir, epoch):
    SQUE = set()
    local = np.loadtxt(os.path.join(root + "result/", timedir, epoch + ".txt"))
    dirroot = root + "data/pred_local/" + timedir + "-" + epoch

    if not os.path.exists(dirroot):
        os.mkdir(dirroot)
    for data in local:
        # print(data)
        len = data.shape[0]
        if SQUE.__contains__(data[0]):
            with open(os.path.join(dirroot, '{0:02d}'.format(int(data[0])) + ".txt"), "a+") as f:
                for i in range(len - 1):
                    f.write(("%e " % (data[i])))
                f.write(("%e\n" % (data[len - 1])))
        else:
            with open(os.path.join(dirroot, '{0:02d}'.format(int(data[0])) + ".txt"), "w+") as f:
                for i in range(len - 1):
                    f.write(("%e " % (data[i])))
                f.write(("%e\n" % (data[len - 1])))
        SQUE.add(data[0])

    return SQUE


def localsepara_fast(root, timedir, epoch):
    if os.path.exists(os.path.join(root + "result/", timedir, "data")):
        datatxt = os.path.join(root + "result/", timedir, "data", epoch + ".txt")
    else:
        datatxt = os.path.join(root + "result/", timedir, epoch + ".txt")
    if not os.path.exists(datatxt):
        return None
    local = np.loadtxt(datatxt)
    dirsum = root + "data/pred_local/" + timedir
    if not os.path.exists(dirsum):
        os.mkdir(dirsum)
    dirroot = os.path.join(dirsum, epoch)
    SQUE = np.unique(local[:, 0]).astype(np.int)
    if not os.path.exists(dirroot):
        os.mkdir(dirroot)
    local = local[local[:, 1] >= 0]
    # local = local[np.argsort(local[:, 1], axis=0)]
    for i in SQUE:
        filepath = os.path.join(dirroot, '{0:02d}'.format(int(i)) + ".txt")
        if os.path.exists(filepath):
            continue
        data = local[local[:, 0] == i, :]
        np.savetxt(filepath, data, fmt='%.6e')
    return SQUE


def get_world(root, timedir, epoch, SQUE):
    # SQUE = 4
    world = np.loadtxt(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt")
    pred_local = np.loadtxt(
        os.path.join(root + "data/pred_local", timedir,  epoch, '{0:02d}'.format(SQUE) + ".txt"))[
                 :, 3:]
    outdir = root + "data/pred_world/" + timedir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outtxt = os.path.join(outdir, epoch)
    if not os.path.exists(outtxt):
        os.mkdir(outtxt)
    pred_world = local2world_fast(root, pred_local, SQUE, outtxt)
    # print(np.max(np.abs(pred_world - world)))
    return pred_world.shape[0]


def relative_error(root, timedir, epoch, SQUE):
    def error(src, tar):
        mse = np.mean((src - tar) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(src - tar))
        len = np.sum(np.abs(tar))
        rate = np.sum(np.abs(src - tar)) / np.sum(np.abs(tar))
        return np.array((mse, rmse, mae, len, rate))[:, None]

    def rota2euler(input):
        n, c = input.shape
        if c == 9:
            r = Rotation.from_matrix(input.reshape(n, 3, 3))
            return r.as_euler('xyz', degrees=True)
        elif c == 4:
            r = Rotation.from_quat(input)
            return r.as_euler('xyz', degrees=True)
        else:
            return input

    def error_detail(src, tar):
        mse = np.mean((src - tar) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(src - tar), axis=0)
        rate = np.sum(np.abs(src - tar), axis=0) / np.sum(np.abs(tar), axis=0)
        len = np.sum(np.abs(tar), axis=0)
        return np.concatenate((mse[None, :], rmse[None, :], mae[None, :], len[None, :], rate[None, :]), axis=0)

    # outtxt = os.path.join(root + "data/pred_local/" + timedir, epoch, "rel_err.txt")
    my = np.loadtxt(os.path.join(root + "data/pred_local", timedir, epoch, '{0:02d}'.format(SQUE) + ".txt"))
    gt = np.loadtxt(os.path.join(root + "data/gt_local_euler", '{0:02d}'.format(SQUE) + ".txt"))
    while my.shape[0] < gt.shape[0]:
        my = np.concatenate((my, my[-1][None, :]), axis=0)
    # if my.shape[1] == 11:
    #     my = np.concatenate((my, np.ones((my.shape[0], 1))), axis=1)
    mypose = my[:, 3:6]
    gtpose = gt[:, 3:6]
    myeuler = rota2euler(my[:, 6:])
    gteuler = gt[:, 6:]
    terr = error(mypose, gtpose)
    rerr = error(myeuler, gteuler)
    tderr = error_detail(mypose, gtpose)
    rderr = error_detail(myeuler, gteuler)
    return np.concatenate((np.concatenate((tderr, terr), axis=1), np.concatenate((rderr, rerr), axis=1)), axis=0)

# def gtget_world(root, SQUE, iseuler):
#     # SQUE = 4
#     world = np.loadtxt(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt")
#     if iseuler:
#         gt_local = np.loadtxt(root + "data/gt_local_euler/" + '{0:02d}'.format(SQUE) + ".txt")[:, 3:]
#         outtxt = root + "data/gt_local_euler_RMSE/"
#     else:
#         gt_local = np.loadtxt(root + "data/syqpose/pred_absolute/" + '{0:02d}'.format(SQUE) + ".txt")[:, 3:]
#         outtxt = root + "data/syq_pred_absolute_RMSE/"
#     if not os.path.exists(outtxt):
#         os.mkdir(outtxt)
#     pred_world = local2world_fast(root, gt_local, SQUE, outtxt, iseuler)
#     print(np.max(np.abs(pred_world - world)))

def gtget_world(root, SQUE, iseuler):
    # SQUE = 4
    world = np.loadtxt(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt")
    if iseuler:
        gt_local = np.loadtxt(root + "data/gt_local_euler/" + '{0:02d}'.format(SQUE) + ".txt")[:, 3:]
        outtxt = root + "data/gt_local_euler_RMSE/"
    else:
        gt_local = np.loadtxt(root + "data/gt_local/" + '{0:02d}'.format(SQUE) + ".txt")[:, 3:]
        # gt_local = np.loadtxt(root + "data/syqpose/pred_relative/" + '{0:02d}'.format(SQUE) + ".txt")[:, 3:]
        outtxt = root + "data/gt_local_RMSE/"
    if not os.path.exists(outtxt):
        os.mkdir(outtxt)
    pred_world = local2world_fast(root, gt_local, SQUE, outtxt)
    print(np.max(np.abs(pred_world - world)))

def icpget_world(root, icpdir, SQUE, icproot):
    # SQUE = 4
    world = np.loadtxt(os.path.join(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt"))
    gt_local = np.loadtxt(os.path.join(root, "data",icproot, icpdir, '{0:02d}'.format(SQUE) + ".txt"))[:, 3:]
    # gt_local[:, 3:] = np.rad2deg(gt_local[:, 3:])
    outtxt = os.path.join(root, "data",icproot, icpdir + "_RMSE")
    if not os.path.exists(outtxt):
        os.mkdir(outtxt)
    pred_world = local2world_fast(root, gt_local, SQUE, outtxt)
    print(np.max(np.abs(pred_world - world)))


if __name__ == '__main__':
    root = "/data/Unsupodo/"
    SQUE = 10
    # iseuler = False
    # world = np.loadtxt(root + "data/gt_world/" + '{0:02d}'.format(SQUE) + ".txt")
    # pred_local = np.loadtxt(os.path.join(root + "data/pred_local/syqnet", '{0:02d}'.format(SQUE) + ".txt"))
    # outtxt = root + "data/pred_world/syqnet"
    # if not os.path.exists(outtxt):
    #     os.mkdir(outtxt)
    # pred_world = local2world_fast(root, pred_local, SQUE, outtxt, iseuler)

    timedir = "(2020-11-03 18-28-50)-(TITANx2)-full"
    epoch = "180test"
    a = relative_error(root, timedir, epoch, SQUE)
    # print(np.max(np.abs(pred_world - world)))