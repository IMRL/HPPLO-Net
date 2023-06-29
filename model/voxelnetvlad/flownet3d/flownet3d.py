import torch.nn as nn
import torch.nn.functional as F
from model.voxelnetvlad.flownet3d.util import PointNetSetAbstraction,PointNetFeaturePropogation,FlowEmbedding,PointNetSetUpConv
import torch

from model.voxelnetvlad.pointPWC.pointPWC import multiScaleChamferSmoothCurvature


class FlowNet3D(nn.Module):
    def __init__(self):
        super(FlowNet3D,self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.5, nsample=16, in_channel=3, mlp=[32,32,64], mlp2=[],group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128], mlp2=[], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256], mlp2=[], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256,256,512], mlp2=[], group_all=False)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel = 128, mlp=[128, 128, 128], pooling='max', corr_func='concat')

        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel = 256, f2_channel = 512, mlp=[], mlp2=[256, 256])
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel = 128+128, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.fp = PointNetFeaturePropogation(in_channel = 256+3, mlp = [256, 256])

        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2=nn.Conv1d(128, 3, kernel_size=1, bias=True)

    def forward(self, pc1, pc2, feature1, feature2):
        pc1 = pc1.transpose(1, 2).contiguous()
        pc2 = pc2.transpose(1, 2).contiguous()
        feature1 = feature1.transpose(1, 2).contiguous()  # B 3 N
        feature2 = feature2.transpose(1, 2).contiguous()  # B 3 N
        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)

        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        return sf.transpose(1, 2), None


class FlowNet3DwithLoss(nn.Module):

    def __init__(self, pwcloss=True):
        super(FlowNet3DwithLoss, self).__init__()

        self.flownet3d = FlowNet3D()
        self.pwcloss = pwcloss

    def forward(self, xyz1, xyz2, color1, color2, needmul):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        color1 = color1.contiguous() # B 3 N
        color2 = color2.contiguous() # B 3 N

        flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2 = self.pointpwc(xyz1, xyz2, color1, color2)

        if self.pwcloss:
            loss = multiScaleChamferSmoothCurvature(pc1, pc2, flows)


            return flows[0].transpose(1, 2), loss[0]
        else:
            return flows[0].transpose(1, 2), None


if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((1,8192,3)).cuda()
    fea = torch.randn((8,3,20480)).cuda()
    input2 = torch.randn((8,3,30960)).cuda()
    fea2 = torch.randn((8,3,30960)).cuda()
    # label = torch.randn(8,16)
    model = FlowNet3D().cuda()
    output, _ = model(input,input2,fea,fea2) # same as input
    print(output.size())

