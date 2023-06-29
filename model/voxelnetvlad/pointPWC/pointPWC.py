"""
PointPWC-Net model and losses
Author: Wenxuan Wu
Date: May 2020
"""
# from param.summaryparam import fastpwcknn
# fastpwcknn = True
import os
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import time

# if fastpwcknn:
from model.voxelnetvlad.pointPWC.pointconv_util_fastknn import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from model.voxelnetvlad.pointPWC.pointconv_util_fastknn import SceneFlowEstimatorPointConv,flow_predictor,classify_rigid,index_points_group,estimate_normal
from model.voxelnetvlad.pointPWC.pointconv_util_fastknn import index_points_gather as index_points, index_points_group, Conv1d, square_distance


from model.voxelnetvlad.pointPWC.pointconv_util_fastknn import transform, normal_gather, matrix_merge, transform_noraml
from model.voxelnetvlad.pointPWC.nonLocalNet import NonLocalNetFlow
from model.voxelnetvlad.pointPWC.nonLocalSingle import NonLocalNetSingle
from model.voxelnetvlad.SVDnet import SVD

import time

scale = 1.0

num_i=0

class PointPWCSVD(nn.Module):
    def __init__(self, ptype="org", in_dim=3, usenonlocal=(), svdmsg=None):
        super(PointPWCSVD, self).__init__()

        self.usenonlocal = usenonlocal
        self.scale = scale
        
        if ptype == "org":
            flow_nei = 32
            feat_nei = 16
            upfea = [32, 64, 128, 256, 512]
            downfea = [32, 32, 64, 64]
            npoint = [2048, 512, 256, 64]
            flowcha = [128, 128]
            flowmlp = [128, 64]
        elif ptype == "down":                       #true
            flow_nei = 16
            feat_nei = 12
            upfea = [16, 32, 64, 128, 256]
            downfea = [16, 16, 32, 32]
            npoint = [2048, 512, 256, 64]
            flowcha = [64, 64]
            flowmlp = [64, 32]
        elif ptype == "downplus":
            flow_nei = 12
            feat_nei = 8
            upfea = [8, 16, 32, 64, 128]
            downfea = [8, 8, 16, 16]
            npoint = [2048, 512, 256, 64]
            flowcha = [32, 32]
            flowmlp = [32, 16]
        else:
            assert False

        if "cost" in self.usenonlocal:    #none
            localfea = [0, 16, 32, 32]
        elif "costall" in self.usenonlocal:
            localfea = [0, 16, 32, 32]
        else:
            localfea = [0, 0, 0, 0]
        #l0: 8192


        if "single" in self.usenonlocal:
            self.nonlocalnet = NonLocalNetSingle(num_channels=upfea[0], in_dim=in_dim)
        elif "flow" in self.usenonlocal:
            self.nonlocalnet = NonLocalNetFlow(num_channels=upfea[0])
            self.level0 = Conv1d(in_dim, upfea[0])
            upfea[0] *= 2
        else:
            self.level0 = Conv1d(in_dim, upfea[0])
        self.level0_1 = Conv1d(upfea[0], upfea[0])
        self.cost0 = PointConvFlow(flow_nei, upfea[0] + downfea[0] + upfea[0] + downfea[0] + 3, localfea[0], [upfea[0], upfea[0]], usenonlocal=[])
        #                       nsample, in_channel, localfea, mlp
        self.flow0 = SceneFlowEstimatorPointConv(upfea[0] + flowmlp[-1], upfea[0], 3, flowcha, flowmlp)

         
        self.flow0_uncer = flow_predictor(in_channel_mlp=22, mlp=[128,1], bn=True)
        self.flow1_uncer = flow_predictor(in_channel_mlp=38, mlp=[128,1], bn=True)
        self.flow2_uncer = flow_predictor(in_channel_mlp=70, mlp=[128,1], bn=True)
        self.flow3_uncer = flow_predictor(in_channel_mlp=134, mlp=[128,1], bn=True)
        
        self.mask0 = classify_rigid(in_channel_mlp=20, mlp=[128,1])
        self.mask1 = classify_rigid(in_channel_mlp=36, mlp=[128,1])
        self.mask2 = classify_rigid(in_channel_mlp=68, mlp=[128,1])
        self.mask3 = classify_rigid(in_channel_mlp=131, mlp=[128,1])
        
        self.level0_2 = Conv1d(upfea[0], upfea[1])

        #l1: 2048
        self.level1 = PointConvD(npoint[0], feat_nei, upfea[1] + 3, upfea[1])
        self.cost1 = PointConvFlow(flow_nei, upfea[1] + downfea[1] + upfea[1] + downfea[1] + 3, localfea[1], [upfea[1], upfea[1]], usenonlocal=usenonlocal)
        self.flow1 = SceneFlowEstimatorPointConv(upfea[1] + flowmlp[-1], upfea[1], 3, flowcha, flowmlp)
        
        
        self.level1_0 = Conv1d(upfea[1], upfea[1])
        self.level1_1 = Conv1d(upfea[1], upfea[2])

        #l2: 512
        self.level2 = PointConvD(npoint[1], feat_nei, upfea[2] + 3, upfea[2])
        self.cost2 = PointConvFlow(flow_nei, upfea[2] + downfea[2] + upfea[2] + downfea[2] + 3, localfea[2], [upfea[2], upfea[2]], usenonlocal=usenonlocal)
        self.flow2 = SceneFlowEstimatorPointConv(upfea[2] + flowmlp[-1], upfea[2], 3, flowcha, flowmlp)
        
        
        self.level2_0 = Conv1d(upfea[2], upfea[2])
        self.level2_1 = Conv1d(upfea[2], upfea[3])

        #l3: 256
        self.level3 = PointConvD(npoint[2], feat_nei, upfea[3] + 3, upfea[3])
        self.cost3 = PointConvFlow(flow_nei, upfea[3] + downfea[3] + upfea[3] + downfea[3] + 3, localfea[3], [upfea[3], upfea[3]], usenonlocal=usenonlocal)
        self.flow3 = SceneFlowEstimatorPointConv(upfea[3], upfea[3], 0, flowcha, flowmlp)
        
        self.level3_0 = Conv1d(upfea[3], upfea[3])
        self.level3_1 = Conv1d(upfea[3], upfea[4])

        #l4: 64
        self.level4 = PointConvD(npoint[3], feat_nei, upfea[4] + 3, upfea[3])

        #deconv
        self.deconv4_3 = Conv1d(upfea[3], downfea[3])
        self.deconv3_2 = Conv1d(upfea[3], downfea[2])
        self.deconv2_1 = Conv1d(upfea[2], downfea[1])
        self.deconv1_0 = Conv1d(upfea[1], downfea[0])

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

        self.svd = SVD(nettype=svdmsg)

    def forward(self, xyz1, xyz2, color1, color2, idx1=None, idx2=None):

        #xyz1, xyz2: B, N, 3
        #color1, color2: B, N, 3

        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)      #B 3 N
        color1 = color1.permute(0, 2, 1) # B 3 N
        color2 = color2.permute(0, 2, 1) # B 3 N

        if "single" in self.usenonlocal:
            feat1_l0 = self.nonlocalnet(color1)
        elif "flow" in self.usenonlocal:
            feat1_l0 = self.level0(color1)
            feat1_l0_local = self.nonlocalnet(pc1_l0, pc2_l0, idx1)
            feat1_l0 = torch.cat((feat1_l0_local, feat1_l0), dim=1)
        else:
            feat1_l0 = self.level0(color1)

        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)    #b 32 8192


        if "single" in self.usenonlocal:
            feat2_l0 = self.nonlocalnet(color2)
        elif "flow" in self.usenonlocal:
            feat2_l0 = self.level0(color2)
            feat2_l0_local = self.nonlocalnet(pc2_l0, pc1_l0, idx2)
            feat2_l0 = torch.cat((feat2_l0_local, feat2_l0), dim=1)
        else:
            feat2_l0 = self.level0(color2)

        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)


        #l0->l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)    
        feat1_l1_2 = self.level1_0(feat1_l1)      
        feat1_l1_2 = self.level1_1(feat1_l1_2)    

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        feat2_l1_2 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1_2)


        #l1->l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)          # b, 128, 512

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
        feat2_l2_3 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2_3)


        #l2->l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        feat1_l3_4 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3_4)         # b, 256, 256

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        feat2_l3_4 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3_4)

        #l3->l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)  
        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)

        #l4-l3
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)  
        
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)        

        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)  # B, C, N
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)               

        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim = 1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim = 1)

        cost3 = self.cost3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)          
        feat3, flow3 = self.flow3(pc1_l3, feat1_l3, cost3)  

        flow3_uncer = self.flow3_uncer(pc1_l3, cost3, flow3)  
        mask3 = self.mask3(pc1_l3, cost3, None) 
        weight3 = flow3_uncer * mask3 
   

        R3, T3, _ = self.svd(pc1_l3.transpose(1, 2), torch.cat((pc2_l0, color2), dim=1).transpose(1, 2), feat2_l0_1, flow3.transpose(1, 2), weight3)
        # [B,3,3] [B,3]
        pc1_all = transform([pc1_l0, pc1_l1, pc1_l2, pc1_l3], R3, T3)
        pc1_l0, pc1_l1, pc1_l2, pc1_l3 = pc1_all

        #l3->l2

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)

        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)

        cost2 = self.cost2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)  #b, 64, 512

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim = 1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cost2, up_flow2)  #b, 32, 512   b, 3, 512

        flow2_uncer = self.flow2_uncer(pc1_l2, cost2, flow2)   # [b, 1, 512] 

        mask3_up = self.upsample(pc1_l2, pc1_l3, mask3)
        mask2 = self.mask2(pc1_l2, cost2, mask3_up)
        weight2 = flow2_uncer*mask2
        
        R2, T2, _ = self.svd(pc1_l2.transpose(1, 2), torch.cat((pc2_l0, color2), dim=1).transpose(1, 2), feat2_l0_1, flow2.transpose(1, 2), weight2)
        
        pc1_all = transform([pc1_l0, pc1_l1, pc1_l2], R2, T2)
        pc1_l0, pc1_l1, pc1_l2 = pc1_all

        #l2->l1
        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)

        cost1 = self.cost1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)        
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cost1, up_flow1)

        flow1_uncer = self.flow1_uncer(pc1_l1, cost1, flow1)   # [b, 1, 2048]
 
        mask2_up = self.upsample(pc1_l1, pc1_l2, mask2)
        mask1 = self.mask1(pc1_l1, cost1, mask2_up)
        weight1 = flow1_uncer*mask1

        R1, T1, _ = self.svd(pc1_l1.transpose(1, 2), torch.cat((pc2_l0, color2), dim=1).transpose(1, 2), feat2_l0_1, flow1.transpose(1, 2), weight1)
        
        pc1_all = transform([pc1_l0, pc1_l1], R1, T1)
        pc1_l0, pc1_l1 = pc1_all

        #l1->l0
        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)

        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)

        cost0 = self.cost0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)  #b, 16, 8192

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cost0, up_flow0)

        flow0_uncer = self.flow0_uncer(pc1_l0, cost0, flow0)   # [b, 1, 8192] 

        mask1_up = self.upsample(pc1_l0, pc1_l1, mask1)
        mask0 = self.mask0(pc1_l0, cost0, mask1_up)
        weight0 = flow0_uncer*mask0
        
        R0, T0, _ = self.svd(pc1_l0.transpose(1, 2), torch.cat((pc2_l0, color2), dim=1).transpose(1, 2), feat2_l0_1, flow0.transpose(1, 2), weight0)
        

        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]       #len 3 save index,   torch.Size([10, 2048])  [10, 512]  [10, 256]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]
        
        R, T = matrix_merge([R3, R2, R1, R0], [T3, T2, T1, T0])         #len 5

        #flows list len 4  torch.Size([10, 3, 8192])  torch.Size([10, 3, 2048])  torch.Size([10, 3, 512])  torch.Size([10, 3, 256])
        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, R, T



class PointPWCwithLoss(nn.Module):


    def __init__(self, ptype="org", usenonlocal=(), useFPFH=False, svdmsg=None,):
        super(PointPWCwithLoss, self).__init__()
        if useFPFH:
            in_dim = 36
        else:
            in_dim = 3

        self.pointpwc = PointPWCSVD(ptype=ptype , in_dim=in_dim, usenonlocal=usenonlocal, svdmsg=svdmsg)

    def forward(self, xyz1, xyz2, color1, color2, needmul, idx1=None, idx2=None):
        xyz1 = xyz1.contiguous()     #torch.Size([10, 8192, 3])    
        xyz2 = xyz2.contiguous()
        color1 = color1.contiguous() # B 3 N         torch.Size([10, 8192, 3])  
        color2 = color2.contiguous() # B 3 N

        flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, R, T = self.pointpwc(xyz1, xyz2, color1, color2, idx1, idx2)

        loss = None
            
        return R, T, loss             




def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow

def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # 
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows):
    f_curvature = 0.3
    f_smoothness = 1.0
    f_chamfer = 1.0

    #num of scale
    num_scale = len(pred_flows)

    alpha = [0.02, 0.04, 0.08, 0.16]
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        cur_pc1 = pc1[i] # B 3 N
        cur_pc2 = pc2[i]
        cur_flow = pred_flows[i] # B 3 N

        #compute curvature
        cur_pc2_curvature = curvature(cur_pc2)

        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

        chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()

        #smoothness
        smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

        #curvature
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()

        chamfer_loss += alpha[i] * chamferLoss
        smoothness_loss += alpha[i] * smoothnessLoss
        curvature_loss += alpha[i] * curvatureLoss


    total_loss = f_chamfer * chamfer_loss + f_curvature * curvature_loss + f_smoothness * smoothness_loss

    return total_loss, chamfer_loss, curvature_loss, smoothness_loss
