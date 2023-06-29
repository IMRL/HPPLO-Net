import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import spatial
from tools import spatial as myspatial
from model.voxelnetvlad.flownet3d import pointnet2_utils
from param.summaryparam import fastpwcknn
from model.voxelnetvlad.pointPWC.pointconv_util_fastknn import estimate_normal

def kdtree_scipy(X, pts):
    tree = spatial.cKDTree(data=X)
    c = tree.query(pts)
    return c[1]


def pointutil(oldtgt, tgt, oldtgtnor):
    dist, idx = pointnet2_utils.knn(1, tgt.contiguous(), oldtgt.contiguous())
    idx = idx.long().repeat(1, 1, 3)
    return torch.gather(oldtgtnor, dim=1, index=idx)


class SVD(nn.Module):
    def __init__(self, nettype="po2po_step", losstype=None, sx=0.0, sq=-2.5):
        super(SVD, self).__init__()
        self.losstype = losstype
        self.nettype = nettype
        
        if nettype == "weight_po2pl":           #true   # point to plane ICP and SVD solve pose
            self.svdcore = weighted_SVDpo2pl()

        if self.losstype is not None:         #none
            self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=True)
            self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=True)

    def forward(self, src, tag, feat2_l0_1, flow, weight, R2=None, T2=None):
        
        if self.nettype == "weight_po2pl":
            R, T = self.svdcore(src, tag, feat2_l0_1, flow, weight)  

        
        if self.losstype is not None:
            if self.losstype == "l2":
                loss_x = F.mse_loss(T, T2)
                loss_q = F.mse_loss(R, R2)
            else:
                loss_x = F.l1_loss(T, T2)
                loss_q = F.l1_loss(R, R2)

            loss = torch.exp(-self.sx) * loss_x + torch.exp(-self.sq) * loss_q
            return R, T, loss
        else:  #true
            return R, T, None
        

class weighted_SVDpo2pl(nn.Module):       #zbb
    def __init__(self):
        super(weighted_SVDpo2pl, self).__init__()

        self.estimate_normal = estimate_normal()
    
    def forward(self, src, oldtgt, feat2_l0_1, flow, weight):   #b n 1
        
        weight = weight.squeeze()

        #####normalize weight######
        # max=torch.max(weight, 1, keepdim=True)[0] #b,1 
        # min=torch.min(weight, 1, keepdim=True)[0] #b,1       
        # weight_nor=(weight-min)/(max-min+1e-8) #归一化到0-1
        
        weight_diag = torch.diag_embed(weight)   #b n n
        
        batch_size, num_points, _ = src.shape           
        tgt = src + flow        

        oldtgtnor = oldtgt[:, :, 3:]
        oldtgt = oldtgt[:, :, :3]

        tgtnor = self.estimate_normal(tgt, oldtgt, oldtgtnor, feat2_l0_1.permute(0, 2, 1))

        y = torch.sum(tgtnor * tgt, dim=-1, keepdim=True) - torch.sum(tgtnor * src, dim=-1, keepdim=True)  #b,n,1
        A1 = (tgtnor[:, :, 2] * src[:, :, 1] - tgtnor[:, :, 1] * src[:, :, 2]).unsqueeze(2)  #b, n,1
        A2 = (tgtnor[:, :, 0] * src[:, :, 2] - tgtnor[:, :, 2] * src[:, :, 0]).unsqueeze(2)
        A3 = (tgtnor[:, :, 1] * src[:, :, 0] - tgtnor[:, :, 0] * src[:, :, 1]).unsqueeze(2)
        H = torch.cat((A1, A2, A3, tgtnor), dim=-1)  #b,n,6
        
        A = torch.matmul(torch.matmul(H.transpose(2, 1), weight_diag), H)   # b, 6, 6
        b = torch.matmul(torch.matmul(H.transpose(2, 1), weight_diag), y)   #b, 6, 1
        
        X = []
        for i in range(batch_size):
            if flow.grad_fn is None:
                x = torch.matmul(torch.pinverse(A[i].cpu()).cuda(), b[i]).squeeze()
            else:
                # print("A", A[i])
                x = torch.matmul(torch.pinverse(A[i]), b[i]).squeeze()
                
            X.append(x)
        X = torch.stack(X, dim=0)

        R = myspatial.euler_to_rotation_matrix((X[:, :3]))   # b,3,3
        t = X[:, 3:]                                         # b,3

        return R, t



def euler2mat(euler):
        euler = euler / 180 * np.pi
        x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]
        B = euler.size(0)

        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)
        one = torch.ones_like(cx)
        zero = torch.zeros_like(cx)

        Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=1).view(B, 3, 3).type(euler.dtype)
        Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=1).view(B, 3, 3).type(euler.dtype)
        Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=1).view(B, 3, 3).type(euler.dtype)

        rotMat = torch.bmm(torch.bmm(Rz, Ry), Rx)
        return rotMat

