# from param.common import *
from copy import copy

import numpy as np
import multiprocessing
import numpy as np
import torch
import socket
from param.titanrtx_1 import *

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    iplist = ip.split(".")
    return iplist[-2] + iplist[-1]


mac = get_host_ip()


print(torch.cuda.get_device_name(0) + "-" + str(torch.cuda.device_count()))       #TITAN RTX-1
# print("========3=========")
# exit()

if usesvdnet and head == "svd":
    outrota = True
else:
    outrota = False

if flaot64:
    ntype = np.float64
    ttype = torch.float64
    # batch_size = int(batch_size * 0.45)
    # workers = int(workers * 1.5)
else:
    ntype = np.float32
    ttype = torch.float32

if usepointnet:
    c = 3
if lonetuncer:
    uncer_features = 2
    uncerlograte = 3
else:
    uncer_features = 1
if lonetwidth:
    vhacc = True
assert not (lonetuncer == useuncer and useuncer is True)
assert not (usesharesiamsenet == useunsharesiamsenet and usesharesiamsenet is True)
assert not (depthnorm == predepthnorm and depthnorm is True)

assert IMUnormtype in ["org", "0to8", "0to8use"]
assert anglelosstype in [None, "angle", "distance", "global", "l1ang"]
assert comquat in [None, "grad", "nograd"]
assert NPYpreload in [None, "offline"]

if torch.cuda.device_count() == 1:
    parallelloss = False

workers = torch.cuda.device_count() * 4

if usedeeplio and deepliolidarmodel == "pointnet":
    useuncer = False
if deepliolidarmodel == "pointnet":
    useloampoints = True

if not loadmodel:
    startepoch = 0

if drop is not None:
    icpthre = [0.0, 10000.0]

angthrevalue = [-1, -1]
if anglelosstype == "angle":
    angthrevalue[1] = 1 - np.cos(angthre[1] * np.pi / 180)
    angthrevalue[1] = float(round(angthrevalue[1], 7))
    angthrevalue[0] = 1 - np.cos(angthre[0] * np.pi / 180)
    angthrevalue[0] = float(round(angthrevalue[0], 7))
elif anglelosstype in ["distance", "l1ang"]:
    angthrevalue[1] = (np.sin(angthre[1] * np.pi / 360) * 2) ** 2
    angthrevalue[1] = float(round(angthrevalue[1], 7))
    angthrevalue[0] = (np.sin(angthre[0] * np.pi / 360) * 2) ** 2
    angthrevalue[0] = float(round(angthrevalue[0], 7))

glothrevalue = [-1, -1]
glothrevalue[1] = 1 - np.cos(glothre[1] * np.pi / 180)
glothrevalue[1] = float(round(glothrevalue[1], 7))
glothrevalue[0] = 1 - np.cos(glothre[0] * np.pi / 180)
glothrevalue[0] = float(round(glothrevalue[0], 7))

if iseuler:
    comquat = None

if dynastep:
    step_size = 40 // continlen

if dynalr:
    lr = 1e-4 / (continlen - 1)

assert continlosslen < continlen
loaderstepsize = min(loaderstepsize, continlen - 1)

assert anglelossrate == 1.0 or anglelosstype is not None

assert continlosslen == 1 or continnetlen == 1

if len(reweight) != 1:
    assert len(reweight) == continlen - 1


if modelepoch == -1:
    modelepoch = startepoch

for i in testset + trainset:
    if 33 <= i <= 38:
        fordnewheight = True
        break

if usevaild:
    vaildset = copy(trainset)
else:
    vaildset = None

if testbatch == -1:
    testbatch = batch_size

assert useflowvoxel or not selfvoxelloss
assert not (cfgvoxel["multiflow"] and not cfgvoxel["modeltype"] == "flowonly")
onlypwcloss = cfgvoxel["onlypwcloss"]


if useflowvoxel:
    cfgvoxelimu = cfg['imu-feat-rnn'].copy()
    cfgvoxelimu["fclast"] = cfg["preprocess-net"]["fclast"]
    cfgvoxelimu["p"] = cfg["preprocess-net"]["p"]
    cfgvoxelimu["input-size"] = 3
    cfgvoxelimu['useIMU'] = useIMU
else:
    cfgvoxelimu = None


