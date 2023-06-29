
import torch
import torch.nn as nn

from model.voxelnetvlad.pointPWC.pointPWC import PointPWCwithLoss
from model.voxelnetvlad.SVDnet import SVD
from model.npointloss import euler2mat, quat2mat


def xxx2mat(input):
    if input.shape[-1] == 3:
        return euler2mat(input)
    else:
        return quat2mat(input)


def get_flow_model(cfg):                      #zbb
    modeltype = cfg['modeltype']
    if modeltype == "flowonly":          
        return FlowOdometry(cfg)


class FlowOdometry(nn.Module):
    def __init__(self, cfg):
        
        super(FlowOdometry, self).__init__()

        self.flowposenetname = cfg['flowposenetname']       #SVD
        
        if self.flowposenetname == "svd":
            self.flowposenet = SVD(nettype=cfg["svdnettype"])    #po2pl  SVD solves poses  point2plane

        self.senceflownet = PointPWCwithLoss( ptype=cfg['pwctype'], usenonlocal=cfg['pwcnonlocal'], svdmsg=cfg["svdnettype"])


    def forward(self, input, needmul=False):     # input: list len 2 或 3： 
    
        if len(input) > 2:
            nowpoints, lastpoints, idx1, idx2 = input
        else:          #pop(-1), len(input)=2
            idx1, idx2 = None, None
            nowpoints, lastpoints = input
            

        # if self.multidyna:              #normal  true
        return self.senceflownet(nowpoints[:, :, :3], lastpoints[:, :, :3],
                                     nowpoints[:, :, 3:], lastpoints[:, :, 3:], needmul, idx1, idx2)



if __name__ == '__main__':
    batchsize = 2
    voxelsize = 7680
    pointsize = 10240
    voxelshape = [10, 32, 32, 256]  # depth proj_y proj_x
    voxelcoords = torch.zeros((batchsize, voxelsize, 4)).int()
    pointsid = torch.zeros((batchsize, pointsize, 2)).int()
    nowpoints = torch.rand(batchsize, pointsize, 6).cuda()
    lastpoints = torch.rand(batchsize, pointsize - 5, 6).cuda()
    voxelcoords[:, :, 0] = torch.randint(low=1, high=voxelshape[0] + 1, size=[batchsize, voxelsize])
    voxelcoords[:, :, 1] = torch.randint(low=0, high=voxelshape[1], size=[batchsize, voxelsize])
    voxelcoords[:, :, 2] = torch.randint(low=0, high=voxelshape[2], size=[batchsize, voxelsize])
    voxelcoords[:, :, 3] = torch.randint(low=0, high=voxelshape[3], size=[batchsize, voxelsize])
    pointsid[:, :, 1] = torch.randint(low=0, high=voxelshape[0], size=[batchsize, pointsize])
    pointsid[:, :, 0] = torch.randint(low=0, high=voxelsize, size=[batchsize, pointsize])

    voxel = torch.rand(batchsize, voxelsize, voxelshape[0], 13).cuda()
    voxelvalid = torch.Tensor([9169, 8402, 9061, 8537, 9469, 8083, 9292, 8974]).int()
    # voxelvalid = torch.Tensor([15169, 15402, 15061, 15537, 15469, 15083, 15292, 15974]).int()
    # pointvalid = torch.Tensor([9800, 9400, 9700, 9900, 9600, 9400, 9600, 9900]).int()
    # voxelvalid = torch.Tensor([12, 11, 12, 12, 13, 13, 12, 11]).int()
    # pointvalid = torch.Tensor([58, 54, 57, 59, 56, 54, 56, 59]).int()
    model = FlowVoxelNetVlad(voxelshape=voxelshape, voxelsize=voxelsize, inshape=6).cuda()
    stat(model, nowpoints, lastpoints, pointsid, voxelcoords, voxelvalid)
    # exit()
    x_ori, x_pos, loss = model(nowpoints, lastpoints, pointsid, voxelcoords, voxelvalid)
