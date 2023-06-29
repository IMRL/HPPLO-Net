import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import model.spatial as myspatial
from model.voxelnetvlad.flownet3d import pointnet2_utils as pointutils


def log(str, ptlog=True):
    pass

def euler2mat(euler):
    if euler.shape[1] == 3 and len(euler.shape) == 3:
        return euler
    euler = myspatial.deg2rad(euler)
    return myspatial.euler_to_rotation_matrix(euler)


def quat2mat(quat):
    if quat.shape[1] == 3 and len(quat.shape) == 3:
        return quat
    return myspatial.quaternion_to_rotation_matrix(quat[:, (1, 2, 3, 0)])


def pointutil(ver_last, ver_now, nor_last):
    dist, idx = pointutils.knn(1, ver_now.contiguous(), ver_last.contiguous())
    idx = idx.long().repeat(1, 1, 3)
    new_verlast = torch.gather(ver_last, dim=1, index=idx)
    new_norlast = torch.gather(nor_last, dim=1, index=idx)
    return new_verlast, new_norlast


def print_top100and10(loss, type, name, needgtloss):
    loss100, _ = torch.topk(loss, k=100)      #求tensor中某个dim的前k大或者前k小的值以及对应的index
    loss10, _ = torch.topk(loss, k=10)
    log(type + " " + name + " 100avg:%15e 10avg:%15e" % (torch.sum(loss100).item() / 100, torch.sum(loss10).item() / 10)
        , needgtloss)


class NPointLoss(nn.Module):

    def __init__(self, beta,  ttype, needgtloss=True, sx=0.0, sq=-2.5, iseuler=True, lowlimit=False,  prweight=1, fastknn=False):
        super(NPointLoss, self).__init__()
        
        self.beta = torch.Tensor([beta]).cuda().type(ttype)

        self.needgtloss = needgtloss
        self.lowlimit = lowlimit              ## true 弱化位移和旋转限制（范围大小）
        self.prweight = prweight

        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=True)            # torch.Size([1])
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=True)         # torch.Size([1])

        self.iseuler = iseuler
        self.ttype = ttype
        self.fastknn = fastknn
        
        
    def timetest(self, flag):
        print(flag, time.time() - self.starttime)
        self.starttime = time.time()
    

    def loss_batch(self, vertexmap_proj, normalmap_proj, vertexlast, normallast, needgtloss, type):
        N = vertexmap_proj.shape[0]
    
        icploss = normallast * (vertexmap_proj - vertexlast)    #  equation (4.12)  # N 3

        icplossN = torch.abs(torch.sum(icploss, dim=1))    #N
        print_top100and10(icplossN, type, "icporg", needgtloss)      #topk=10 or 100
        
        icpvaluecheck = torch.ones(N, dtype=self.ttype).cuda().bool()        
        icploss = torch.abs(torch.sum(icploss, dim=1))

        icppuresumloss = torch.sum(icplossN)

        icploss = icploss[icpvaluecheck]
        icpsumloss = torch.sum(icploss)

        log(type + " pointlen:%15e icploss:%15e icppureloss:%15e " % (
            N, icpsumloss.item() / (icploss.shape[0] + 1e-5) * 130000,
            icppuresumloss.item() / (icploss.shape[0] + 1e-5) * 130000), needgtloss)

        #除icpsumloss,icploss其他loss均为0
        return icpsumloss,icpvaluecheck, icploss

    def loss_core(self, vertexmap_proj, normalmap_proj, vertexlast, normallast, needgtloss, pointlen, type):
        icpbatchloss = torch.zeros(1).cuda().type(self.ttype)
        
        B = vertexmap_proj.shape[0]
        
        icp_count_sum = 0

        for i in range(0, B):
            # self.timetest("14")
            icpsumloss, icpvaluecheck, icploss = \
                self.loss_batch(vertexmap_proj[i, :pointlen[i]], normalmap_proj[i, :pointlen[i]],
                                vertexlast[i, :pointlen[i]], normallast[i, :pointlen[i]], needgtloss, type)

            icpbatchloss += icpsumloss
            icp_count_sum += torch.sum(icpvaluecheck).item()
            
        output = {"point_count_sum": pointlen.sum(), "icp_count_sum": icp_count_sum, "icploss": icpbatchloss}
        return output
    

    def data_trans_samesize(self, last_data, now_data, quat, trans, name, rotainput):     #点云和法向量的投影函数
        vertexmap = now_data["lossalldata"][:, :, :3].cuda()     #[10,10240,3]
        normalmap = now_data["lossalldata"][:, :, 3:].cuda()           #[10,10240,3]
        vertexlast = last_data["lossalldata"][:, :, :3].cuda()       #[10,10240,3]
        normallast = last_data["lossalldata"][:, :, 3:].cuda()
        B, N, _ = vertexmap.shape
        if rotainput and name == "model":       #true
            rota = quat
        
        ver_3 = vertexmap.permute(0, 2, 1)        #[10,3,10240]
        #normalmap_proj   vertexmap_proj为按照位姿投影的warp图像
        normalmap_proj = rota.bmm(normalmap.permute(0, 2, 1)).permute(0, 2, 1)
        vertexmap_proj = (rota.bmm(ver_3) + trans.unsqueeze(2)).permute(0, 2, 1)
        pointlen = (torch.ones(B) * N).to(vertexmap.device).type(torch.int)

        if self.fastknn:
            new_vertexlast, new_normallast = pointutil(vertexlast, vertexmap_proj, normallast)
            # KNN找对应点
        
        return normalmap_proj, vertexmap_proj, new_vertexlast, new_normallast, pointlen
    

    def getresultstr(self, B, output, prloss, type):
        if prloss is None:
            prloss = torch.zeros(1).cuda().type(self.ttype)
        datastr = type + "avg:point-N:%6d icp-N:%6d \n" \
                         "          icp:%15e  pr:%15e" % (output["point_count_sum"],
                                                                           output["icp_count_sum"] / B,
                                                                           (output["icploss"] / B).item(),
                                                                           (prloss / B).item())
        return datastr

                    
    def forward(self, last_data, now_data, quat, trans, needgtloss, rotainput):
        #last_data：dataseq[0]  now_data:dataseq[0]  uncertainty:none  quat:[10,3,3]  trans:[10,3]
        #needgtloss: false   rotainput:true

        needgtloss = False
        B = now_data["bindex"].shape[0]

        if self.lowlimit:         #true  # 弱化位移和旋转限制（范围大小）  bb
            loss_x = F.mse_loss(trans, trans.clamp(-10, 10))
            #将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量

        if self.iseuler:     #true  euler     bb
            if self.lowlimit:      # true #原值和归一化后的值的mse_loss?? bb
                loss_q1 = F.mse_loss(quat[:, :2], quat[:, :2].clamp(-15, 15))
                loss_q2 = F.mse_loss(quat[:, 2], quat[:, 2].clamp(-15, 15))

        loss_q = loss_q1 + loss_q2

        prloss = loss_x + loss_q * self.beta

        log("prloss:%15e%15e%15e%15e%15e" % (
            loss_x.item(), loss_q.item(), prloss.item(), self.sx.item(), self.sq.item()))

        normalmap_proj, vertexmap_proj, vertexlast, normallast, pointlen = self.data_trans_samesize(last_data, now_data,
                                                                                            quat, trans, "model",
                                                                                            rotainput)

        output = self.loss_core(vertexmap_proj, normalmap_proj, vertexlast, normallast, needgtloss, pointlen,
                                "model")
        modelstr = self.getresultstr(B, output, prloss, "model")
        log(modelstr)
        
        loss = output["icploss"] + prloss * self.prweight
        gtloss = 0
        gtstr = "None"

        return loss, gtloss, modelstr, gtstr
