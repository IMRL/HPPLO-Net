import torch.nn as nn
import torch
from model.voxelnetvlad.flownet3d import pointnet2_utils

class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5
        # combine the feature similarity with spatial consistency
        weight = torch.softmax(feat_attention, dim=-1)
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])
        message = self.fc_message(message)
        res = feat + message
        return res


class NonLocalNetSingle(nn.Module):
    def __init__(self, in_dim=3, num_layers=1, num_channels=128):
        super(NonLocalNetSingle, self).__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.blocks = nn.ModuleDict()
        self.layer0 = nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True),
                # nn.InstanceNorm1d(num_channels),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True)
            )
            self.blocks[f'PointCN_layer_{i}'] = layer
            self.blocks[f'NonLocal_layer_{i}'] = NonLocalBlock(num_channels)



    def forward(self, src_keypts):

        feat = self.layer0(src_keypts)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat)
        return feat