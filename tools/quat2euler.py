import torch
from scipy.spatial.transform import Rotation
import os
import numpy as np
root = "../../data/gt_local"
out = "../../data/gt_local_euler"


def quat2mat(quat):
    x, y, z, w = quat
    # print(np.shape(x))
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    xw, yw, zw = x * w, y * w, z * w
    xy, xz, yz = y * x, z * x, z * y

    rotMat = torch.stack([x2 + y2 - z2 - w2, 2 * yz - 2 * xw, 2 * yw + 2 * xz,
                          2 * yz + 2 * xw, x2 - y2 + z2 - w2, 2 * zw - 2 * xy,
                          2 * yw - 2 * xz, 2 * zw + 2 * xy, x2 - y2 - z2 + w2]).view(3, 3).type(
        torch.float64)
    return rotMat

def euler2mat(euler):
    x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    one = torch.ones_like(cx)
    zero = torch.zeros_like(cx)
    # print(np.shape(x))
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx]).view(3, 3).type(torch.float64)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy]).view(3, 3).type(torch.float64)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one]).view(3, 3).type(torch.float64)

    rotMat = torch.mm(torch.mm(Rz, Ry), Rx)
    return rotMat


for file in os.listdir(root):
    inpath = os.path.join(root, file)
    outpath = os.path.join(out, file)
    data = np.loadtxt(inpath, dtype=np.float64)
    outdata = None
    keep = data[:, :6]
    old = data[:, 6:]
    old = old[:, (1, 2, 3, 0)]
    r = Rotation.from_quat(old)
    outdata = r.as_euler('xyz', degrees=True)
    # outdata[outdata > 90] -= 180
    # outdata[outdata < -90] += 180
    # outdata = outdata / 180.0 * np.pi
    raota = r.as_matrix()
    # r2 = Rotation.from_euler('xyz', outdata, degrees=True)
    # quat2 = r2.as_quat()
    # print(np.sum(quat2 - old))
    # output = np.hstack((keep, outdata))
    # print(np.max(np.abs(old[:, 1])), np.max(np.abs(old[:, 2])), np.max(np.abs(old[:, 3])))
    print(file)
    print(np.max(outdata[:, 0]), np.max(outdata[:, 1]), np.max(outdata[:, 2]))
    print(np.min(outdata[:, 0]), np.min(outdata[:, 1]), np.min(outdata[:, 2]))
    # np.savetxt(outpath, output, fmt='%.6e')

# for i in range(outdata.shape[0]):
#     c = euler2mat(torch.Tensor(outdata[i])).numpy()
#     d = quat2mat(torch.Tensor(data[i, 6:])).numpy()
#     print(np.sum(np.abs(d - c)))
    # print(c.reshape(-1, 9))
    # print(raota[i].reshape(-1, 9))
"""
-1.6155941964134617e-06
-4.830038298532812e-07
2.9476081798799344e-07
-3.54073790745954e-06
7.509649047728074e-07
-7.659469268046598e-07
-1.1318709238606708e-06
-1.2677848062488783e-06
1.0478693234348264e-06
-2.258425007659723e-07
-5.104299505343004e-07
"""