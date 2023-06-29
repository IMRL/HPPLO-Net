import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import platform
import warnings

warnings.filterwarnings("ignore")


def is_scan_or_npy(filename):
    return any(filename.endswith(ext) for ext in ['.bin', 'npy', 'npz'])


def squebinindex(str):
    if platform.system() == 'Linux':
        data = str.split("/")
        return int(data[-4]), int(data[-1].split(".")[0])
    else:
        data = str.split("\\")
        return int(data[-4]), int(data[-1].split(".")[0])


def getindex(str):
    return int(str.split("/")[-1].split(".")[0])


def semanticpath(root, semanticdir, seqindex, bindex, last):
    seqindex = '{0:02d}'.format(int(seqindex))
    if last:
        bindex = '{0:06d}'.format(int(bindex) - 1)
    else:
        bindex = '{0:06d}'.format(int(bindex))
    return os.path.join(root, seqindex, semanticdir, bindex + ".label")



class KITTI(Dataset):

    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 iseuler,
                 ntype,
                 local=True,
                 groundtruth=True,
                 gtworldpath="./data/gt_world/",
                 usesavedata=False,
                 continlen=2,
                 gtlocalpath=None,
                 stepsize=1,
                 lossalldata=None,
                 testcompletion=False,
                 newnetname=None,
                 cfgvoxel=None,
                 showtestloss=True,
                 ):

        # save deats\

        if gtlocalpath:
            self.gtlocalpath = gtlocalpath
        else:
            if iseuler:
                self.gtlocalpath = "./data/gt_local_euler/"
            else:
                self.gtlocalpath = "./data/gt_local/"

        self.root = os.path.join(root, "sequences")
        self.gt_data = None
        self.sequences = sequences
        self.local = True
        self.ntype = ntype
        self.gt = groundtruth
        self.usesavedata = usesavedata        
        self.continlen = continlen
        self.iseuler = iseuler        
        self.testmode = False        
        self.lossalldata = lossalldata
        self.testcompletion = testcompletion
        self.forwardmodel = False
        self.seqstepsize = 1
        self.voxelname = cfgvoxel['voxeldataname']
        self.showtestloss = showtestloss

       
        self.voxelmap = {"sp_dpp_10240": "spherical_deepplusplus_10240",
                         "sp_dpp_7168_90_de": "spherical_deepplusplus_7168_90_dense",
                         "ca_0.3_7168_90_de": "cartesian_0.3_7168_90_dense",
                         "ca_0.3_8192_90_uud_de": "cartesian_0.3_8192_90_ucuzdd_dense",
                         "ca_0.3_9126_90_uud_de": "cartesian_0.3_9126_90_ucuzdd_dense",
                         "ca_0.3_7168_90_uud_c_de": "cartesian_0.3_7168_90_ucuzdd_com_dense",
                         "ca_0.3_8192_90_uud_c_de": "cartesian_0.3_8192_90_ucuzdd_com_dense",
                         "ca_0.3_9126_90_uud_c_de": "cartesian_0.3_9126_90_ucuzdd_com_dense",
                         "ca_0.3_8192_95_uud_c_de": "cartesian_0.3_8192_95_ucuzdd_com_dense",
                         "ca_0.3_8192_50_-1.4_uud_c_de": "cartesian_0.3_8192_50_-1.4_ucuzdd_com_dense",
                         "ca_0.3_8192_50_-10_uud_c_de": "cartesian_0.3_8192_50_-10_ucuzdd_com_dense",
                         "ca_0.3_8192_50_-10_uud_de": "cartesian_0.3_8192_50_-10_ucuzdd_dense",
                         "ca_0.3_8192_40_-1.5_uud_c_de": "cartesian_0.3_8192_40_-1.5_ucuzdd_com_dense",
                         "ca_0.3_9126_50_-1.4_uud_c_de": "cartesian_0.3_9126_50_-1.4_ucuzdd_com_dense",
                         "ca_0.3_9126_60_-1.3_uud_c_de": "cartesian_0.3_8192_60_-1.3_ucuzdd_com_dense",
}

        self.newnetname = newnetname

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure sequences is a list
        assert (isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_now = []
        self.points_now = []
        self.points_last = []


        # fill in with names, checking that all sequences are complete
        for k in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(k))

            print("parsing seq {}".format(seq))

            # get paths for each
            if self.voxelname is not None:
                scan_path = os.path.join(self.root, seq, "voxeldata", self.voxelmap[self.voxelname])
                if newnetname == "flowonly":
                    scan_path = os.path.join(self.root, seq, "voxeldata", self.voxelmap[self.voxelname] + "_flowonly")
            elif self.usesavedata:
                scan_path = os.path.join(self.root, seq, "unsupdata", self.unsupnpyname)
            else:
                scan_path = os.path.join(self.root, seq, "velodyne")
            if groundtruth:
                if local:
                    gt_path = os.path.join(self.gtlocalpath, seq + ".txt")
                    gt_data = np.loadtxt(gt_path, dtype=self.ntype)
                    if iseuler:
                        gt_data = gt_data.reshape((-1, 9))[:, 3:]
                    else:
                        gt_data = gt_data.reshape((-1, 10))[:, 3:]
                else:
                    gt_path = os.path.join(gtworldpath, seq + ".txt")
                    gt_data = np.loadtxt(gt_path, dtype=self.ntype)
                    gt_data = gt_data.reshape((-1, 12))
                if self.gt_data is not None:
                    self.gt_data[k] = gt_data
                else:
                    self.gt_data = {k: gt_data}


            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_path)) for f in fn if
                          is_scan_or_npy(f)]

            # check all scans have labels

            # extend list
            scan_files.sort()
            if not testcompletion:
                scan_files = scan_files[(self.continlen - 1):]
            else:
                scan_files = scan_files[1:]
            if stepsize > 1:
                lastone = scan_files.pop()
                scan_files = np.array(scan_files)
                scan_files = scan_files[range(0, scan_files.shape[0], stepsize)].tolist()
                scan_files.append(lastone)
            self.scan_now.extend(scan_files)

        self.scan_now.sort()

        if self.testmode:
            self.scan_now = self.scan_now[:18]

        print("Using {} scans from sequences {}".format(len(self.scan_now), self.sequences))


    def load_voxel(self, seqindex, bindex, notfirstflag):
        if self.voxelname.split("_")[-1] == "de":
            if self.newnetname == "flowonly" or (not notfirstflag and self.newnetname == "flowvoxel"):
                return self.load_voxel_flowonly(seqindex, bindex, True)


    def load_voxel_flowonly(self, seqindex, bindex, dense):

        datadir = os.path.join(self.root, '{0:02d}'.format(seqindex), "voxeldata", self.voxelmap[self.voxelname] + "_flowonly")

        npy_file = '{0:06d}'.format(bindex) + ".npz"
        voxel = np.load(os.path.join(datadir, npy_file))
        voxellen = voxel["voxellen"]
        
        if dense:
            voxelmsg = voxel["voxelmsg"].astype(self.ntype)
            return {"ponintnor": voxelmsg, "voxelvalid": voxellen.astype(np.int)}

    

    def load_data(self, seqindex, bindex, flag, seqstepsize):

        notfirstflag = flag != -1
        if self.testcompletion and bindex < 0:
            bindex = 0

        if self.voxelname is not None:
            data = self.load_voxel(seqindex, bindex, notfirstflag)
        

        if self.gt is not None and notfirstflag:
            data["gt_quaternion"] = self.gt_data[seqindex][bindex - 1, 3:]
            data["gt_translation"] = self.gt_data[seqindex][bindex - 1, :3]
            data["now_rotation"] = np.eye(3, dtype=self.ntype)
            data["now_translation"] = np.zeros(3, dtype=self.ntype)
        
        data["seqindex"] = seqindex
        data["bindex"] = bindex


        if self.lossalldata is not None and self.showtestloss:
         
                # if self.lossalldata is not None and (self.lossalldata.isdigit() or self.lossalldata.split("_")[0].isdigit()):
            npy_file = '{0:06d}'.format(bindex) + ".npy"
            data["lossalldata"] = np.load(os.path.join(self.root, '{0:02d}'.format(seqindex), "npointsloss",
                                                    self.lossalldata, npy_file), allow_pickle=True).astype(self.ntype)

        return data

    def __getitem__(self, index):

        if self.forwardmodel:
            return index
        seqindex, bindex = squebinindex(self.scan_now[index])
        dataseq = []
        indexlist = np.array(range(bindex - (self.continlen - 1) * self.seqstepsize, bindex + 1, self.seqstepsize))
        indexlist = np.clip(indexlist, 0, 100000)
        for i in range(len(indexlist)):
            if i == 0:
                flag = -1
            else:
                flag = indexlist[i] - indexlist[i - 1]
            dataseq.append(self.load_data(seqindex, indexlist[i], flag, self.seqstepsize))
        return dataseq

    def __len__(self):
        return len(self.scan_now)


class Parser():
    # standard conv, BN, relu
    def __init__(self,
                 root,  # directory for data
                 iseuler,
                 ntype,
                 train_sequences,  # sequences to train
                 valid_sequences=None,  # sequences to validate.
                 test_sequences=None,  # sequences to test (if none, don't get)
                 batch_size=32,  # batch size for train and val
                 workers=5,  # threads to load data
                 shuffle_train=False,
                 usesavedata=False,
                 Distributed=False,
                 continlen=2,
                 gtlocalpath=None,
                 stepsize=1,
                 lossalldata=False,
                 testcontin2=False,
                 testcompletion=False,
                 traincompletion=False,
                 testbatch=False,
                 newnetname=None,
                 cfgvoxel=None,
                 showtestloss=True):
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.workers = workers
        self.shuffle_train = shuffle_train
        self.iseuler = iseuler
        # number of classes that matters is the one for xentropy
        if testcontin2:
            testcontinlen = 2
        else:
            testcontinlen = continlen


        if self.train_sequences:
            self.train_dataset = KITTI(root=self.root,
                                       iseuler=self.iseuler,
                                       ntype=ntype,
                                       usesavedata=usesavedata,
                                       continlen=continlen,
                                       gtlocalpath=gtlocalpath,
                                       stepsize=stepsize,
                                       lossalldata=lossalldata,
                                       testcompletion=traincompletion,
                                       newnetname=newnetname,
                                       cfgvoxel=cfgvoxel,
                                       sequences=self.train_sequences)
            if Distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            else:
                train_sampler = None

            self.trainloader = DataLoader(self.train_dataset,
                                          batch_size=batch_size,
                                          shuffle=self.shuffle_train,
                                          num_workers=self.workers,
                                          sampler=train_sampler,
                                          pin_memory=True,
                                          drop_last=False)

            assert len(self.trainloader) > 0
            # self.trainiter = iter(self.trainloader)

        if self.valid_sequences:
            self.valid_dataset = KITTI(root=self.root,
                                       iseuler=self.iseuler,
                                       ntype=ntype,
                                       usesavedata=usesavedata,
                                       continlen=testcontinlen,
                                       gtlocalpath=gtlocalpath,
                                       lossalldata=lossalldata,
                                       testcompletion=testcompletion,
                                       cfgvoxel=cfgvoxel,
                                       newnetname=newnetname,
                                       showtestloss=showtestloss,
                                       sequences=self.valid_sequences)
            if Distributed:
                valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset)
            else:
                valid_sampler = None

            self.validloader = DataLoader(self.valid_dataset,
                                          batch_size=testbatch,
                                          shuffle=False,
                                          num_workers=self.workers,
                                          sampler=valid_sampler,
                                          pin_memory=True,
                                          drop_last=False)
            assert len(self.validloader) > 0
            # self.validiter = iter(self.validloader)

        if self.test_sequences:
            self.test_dataset = KITTI(root=self.root,
                                      iseuler=self.iseuler,
                                      ntype=ntype,
                                      usesavedata=usesavedata,
                                      continlen=testcontinlen,
                                      gtlocalpath=gtlocalpath,
                                      lossalldata=lossalldata,
                                      testcompletion=testcompletion,
                                      cfgvoxel=cfgvoxel,
                                      newnetname=newnetname,
                                      showtestloss=showtestloss,
                                      sequences=self.test_sequences)
            if Distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)
            else:
                test_sampler = None

            self.testloader = DataLoader(self.test_dataset,
                                         batch_size=testbatch,
                                         shuffle=False,
                                         num_workers=self.workers,
                                         sampler=test_sampler,
                                         pin_memory=True,
                                         drop_last=False)
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)


    def get_train_set(self):
        return self.trainloader

    def get_valid_set(self):
        return self.validloader

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)
