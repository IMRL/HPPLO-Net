"""
PointConv util functions
Author: Wenxuan Wu
Date: May 2020
"""

import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
# from sklearn.neighbors.kde import KernelDensity
from model.voxelnetvlad.flownet3d import pointnet2_utils
from model.voxelnetvlad.pointPWC.nonLocalNet import NonLocalNetCost, NonLocalNetCostAll
from model.voxelnetvlad.pointnet2 import pointnet2_utils as pointnet2_utils_pt

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv2d(nn.Module):   #zbb
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def fastknn_point(nsample, pos1_t, pos2_t):
    if not pos2_t.is_contiguous():
        pos2_t = pos2_t.contiguous()
    if not pos1_t.is_contiguous():
        pos1_t = pos1_t.contiguous()
    dist, idx = pointnet2_utils.knn(nsample, pos2_t, pos1_t)
    return idx.long()


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = fastknn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = fastknn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def normal_gather(normals, index):
    idx = index.unsqueeze(1).long().repeat(1, 3, 1)
    return torch.gather(normals, dim=-1, index=idx)

def transform(points, R, T):
    newpoints = []
    for i in range(len(points)):
        newpoints.append((torch.bmm(R, points[i]) + T.unsqueeze(-1)))

    return newpoints

def transform_noraml(normals, R):
    newnormals = []
    for i in range(len(normals)):
        newnormals.append(torch.bmm(R, normals[i]))

    return newnormals

def matrix_merge(R, T):
    new_R = [R[0]]
    now_R = R[0]
    new_T = [T[0]]
    now_T = T[0]
    for i in range(1, len(R)):
        now_T = torch.bmm(R[i], now_T.unsqueeze(-1)).squeeze(-1) + T[i]
        new_T.append(now_T)
        now_R = torch.bmm(R[i], now_R)
        new_R.append(now_R)
    new_R.reverse()              #list len 5      list[0]: torch.Size([10, 3, 3])
    new_T.reverse()              #list len 5      list[0]: torch.Size([10, 3])
    return new_R, new_T


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, localfea, mlp, bn = use_bn, use_leaky = True, usenonlocal=[]):
        super(PointConvFlow, self).__init__()
        in_channel += localfea
        self.nsample = nsample
        self.bn = bn
        self.usenonlocal = usenonlocal
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        if "cost" in self.usenonlocal:
            self.nonlocalnet = NonLocalNetCost(num_channels=localfea)
        elif "costall" in self.usenonlocal:
            self.nonlocalnet = NonLocalNetCostAll(num_channels=localfea)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = fastknn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        if "cost" in self.usenonlocal or "costall" in self.usenonlocal:
            if "cost" in self.usenonlocal:
                localfeature = self.nonlocalnet(xyz1, neighbor_xyz).transpose(1, 2).view(B, N1, 1, -1).repeat(1, 1, self.nsample, 1)
            else:
                localfeature = self.nonlocalnet(xyz1, neighbor_xyz).transpose(1, 3)
            new_points = torch.cat([new_points, localfeature], dim = -1)
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = fastknn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost

class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = fastknn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3

        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3
        knn_idx = fastknn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow 

class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])
    
class flow_predictor(nn.Module):  # mlp predict flow weight
    
    def __init__(self, in_channel_mlp, mlp, bn=True ):
        
        super(flow_predictor, self).__init__()
        
        self.mlp_convs = nn.ModuleList()

        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channel_mlp, mlp[0], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(mlp[0]),
            nn.LeakyReLU(0.2, inplace=True))

        self.mlp2 = nn.Sequential(
            nn.Conv2d(mlp[0], mlp[1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid())
        
        self.mlp_convs.append(self.mlp1)
        self.mlp_convs.append(self.mlp2)
            
    def forward(self, points_f1, cost_volume, flow):  #b, 3, 256   b, 128, 256  b, 3, 256
        
        points_concat = torch.cat([points_f1, cost_volume, flow], dim=1) # B,nchannel1+nchannel2,ndataset1

        points_concat = points_concat.unsqueeze(2)                 # [1,192,64]
        
        for i, conv in enumerate(self.mlp_convs):
            points_concat = conv(points_concat)               # b, 1, 1, 256
            
        W = points_concat.squeeze(2)           # b, 1, 256
                    
        return W                      #B, 1, npoint
    
    
class classify_rigid(nn.Module):  # mlp classify static and dynamic
    
    def __init__(self, in_channel_mlp, mlp):
        
        super(classify_rigid, self).__init__()
        
        self.mlp_convs = nn.ModuleList()

        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channel_mlp, mlp[0], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(mlp[0]),
            nn.LeakyReLU(0.2, inplace=True))

        self.mlp2 = nn.Sequential(
            nn.Conv2d(mlp[0], mlp[1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid())
        
        self.mlp_convs.append(self.mlp1)
        self.mlp_convs.append(self.mlp2)
            
    def forward(self, points_f1, cost_volume, lastmask):  #b, 3, 256   b, 128, 256  b, 1, 256
        
        if lastmask is None:
            points_concat = torch.cat([points_f1, cost_volume], dim=1)
        else:
            points_concat = torch.cat([points_f1, cost_volume, lastmask], dim=1) # B,nchannel1+nchannel2,ndataset1

        points_concat = points_concat.unsqueeze(2)                 # [1,192,64]
        
        for i, conv in enumerate(self.mlp_convs):
            points_concat = conv(points_concat)               # b, 4, 1, 256
        
        mask = points_concat.squeeze(2)           # b, 1, 256

                    
        return mask                      #B, 1, npoint


def index_points_group_pt(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils_pt.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points



class estimate_normal(nn.Module):

    def __init__(self):
        super(estimate_normal, self).__init__()
        
        mlp_list = [32, 64]
        fp_list = [128]
        
        in_channel_mlp = 35
        in_channel_fp = 192
        
        self.mlp1 = nn.ModuleList()
        self.mlp2 = nn.ModuleList()
        self.mlp3 = nn.ModuleList()
        
        self.fp = nn.ModuleList()        
        self.fc = nn.ModuleList()
        
        for i, num_out_channel in enumerate(mlp_list):
            self.mlp1.append(nn.Conv2d(in_channel_mlp, num_out_channel, 1))
            self.mlp1.append(nn.BatchNorm2d(num_out_channel))
            self.mlp1.append(nn.LeakyReLU(LEAKY_RATE, inplace=True))
            
            self.mlp2.append(nn.Conv2d(in_channel_mlp, num_out_channel, 1))
            self.mlp2.append(nn.BatchNorm2d(num_out_channel))
            self.mlp2.append(nn.LeakyReLU(LEAKY_RATE, inplace=True))
            
            self.mlp3.append(nn.Conv2d(in_channel_mlp, num_out_channel, 1))
            self.mlp3.append(nn.BatchNorm2d(num_out_channel))
            self.mlp3.append(nn.LeakyReLU(LEAKY_RATE, inplace=True))
                                                       
            in_channel_mlp = num_out_channel

        for i, num_out_channel in enumerate(fp_list):
            self.fp.append(nn.Conv2d(in_channel_fp, num_out_channel, 1))
            self.fp.append(nn.BatchNorm2d(num_out_channel))
            self.fp.append(nn.LeakyReLU(LEAKY_RATE, inplace=True)) 
            
            in_channel_fp = num_out_channel
            
        self.fc.append(nn.Conv2d(128, 1, 1))
        self.fc.append(nn.BatchNorm2d(1))
        self.fc.append(nn.LeakyReLU(LEAKY_RATE, inplace=True))
        self.fc.append(nn.Conv2d(1, 1, 1))
                

    def forward(self, des_pc, pc2_l0, color2, feat2_l0_1):
        """
        weighted calculate normal
        Input:
            des_pc: B, 3, 512 
            pc2_l0: input points position data, [B, 3, 8192]
            feat2_l0_1: input feature of pc2_l0, [B, 32, 8192]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        _, idx1 = pointnet2_utils.knn(4, des_pc.contiguous(), pc2_l0.contiguous())  #des_pc find knn in pc2_l0  (b, 512,8)
        _, idx2 = pointnet2_utils.knn(8, des_pc.contiguous(), pc2_l0.contiguous())  #des_pc find knn in pc2_l0  (b, 512,16)
        _, idx3 = pointnet2_utils.knn(16, des_pc.contiguous(), pc2_l0.contiguous())  #des_pc find knn in pc2_l0  (b, 512,32)
        
        knn1 =  index_points_group_pt(pc2_l0, idx1)   # b, 512, 8, 3    bnkc   mlp[]
        knn2 = index_points_group_pt(pc2_l0, idx2)   # b, 512, 16, 3    
        knn3 = index_points_group_pt(pc2_l0, idx3)   # b, 512, 32, 3
        
        knn1_fea =  index_points_group_pt(feat2_l0_1, idx1)  # b, 512, 8, 32    bnkc   mlp[]
        knn2_fea = index_points_group_pt(feat2_l0_1, idx2)   # b, 512, 16, 32    
        knn3_fea = index_points_group_pt(feat2_l0_1, idx3)   # b, 512, 32, 32
        
        p1 = torch.cat([knn1, knn1_fea], dim=3).permute(0, 3, 2, 1)            #bckn  b,35,8,512
        p2 = torch.cat([knn2, knn2_fea], dim=3).permute(0, 3, 2, 1)            # b,35,16,512
        p3 = torch.cat([knn3, knn3_fea], dim=3).permute(0, 3, 2, 1)            # b,35,32,512
        
        for conv in self.mlp1:
            p1 = conv(p1)                                                      # b,64,8,512
        
        for conv in self.mlp2:
            p2 = conv(p2)                                                      # b,64,16,512
            
        for conv in self.mlp3:
            p3 = conv(p3)                                                      # b,64,32,512   bckn
        
        new_p2 = torch.max(p2, 2, keepdim=True)[0]                             # b,64,1,512
        new_p3 = torch.max(p3, 2, keepdim=True)[0]                             # b,64,1,512
        
        new_p2 = new_p2.repeat(1, 1, 4, 1)                                     # b,64,8,512
        new_p3 = new_p3.repeat(1, 1, 4, 1)                                     # b,64,8,512
               
        feat_1 = torch.cat([p1, new_p2, new_p3], dim=1)                           # b,192,8,512
        
        for conv in self.fp:
            feat_1 = conv(feat_1)                                                      # b,128,8,512
        for conv in self.fc:
            feat_1 = conv(feat_1)                                                      # b,1,8,512
            
        weight = F.softmax(feat_1, dim=2)                                    # # b,1,4,512

        normal_pc2 = index_points_group_pt(color2, idx1)  # b, 512, 4, 3    bnkc   mlp[]
        weight = weight.permute(0, 3, 2, 1)                  # b, 512, 4, 1  
        normal_des_pc = torch.sum(normal_pc2 * weight, dim=2, keepdim=True)        #b, 512, 1, 3
   
        
        return normal_des_pc.squeeze()
