import torch.nn as nn
import torch
from model.voxelnetvlad.flownet3d import pointnet2_utils


# def pointutil(src_keypts, tgt_keypts):
#     if src_keypts.shape[-1] > 6:
#         dist, idx = pointnet2_utils.knn(1, src_keypts[:, :, 3:], src_keypts[:, :, 3:])
#     else:
#         dist, idx = pointnet2_utils.knn(1, src_keypts[:, :, :3], src_keypts[:, :, :3])
#     idx = idx.long().repeat(1, 1, 3)
#     new_tgt_keypts = torch.gather(tgt_keypts, dim=1, index=idx)
#     return src_keypts[:, :, :3].permute(0, 2, 1), new_tgt_keypts[:, :, :3].permute(0, 2, 1)


def pointutil(src_keypts, tgt_keypts, idx):
    # if src_keypts.shape[-1] > 6:
    #     dist, idx = pointnet2_utils.knn(1, src_keypts[:, :, 3:], src_keypts[:, :, 3:])
    # else:
    #     dist, idx = pointnet2_utils.knn(1, src_keypts[:, :, :3], src_keypts[:, :, :3])
    idx = idx.long().unsqueeze(-1).repeat(1, 1, 3)
    new_tgt_keypts = torch.gather(tgt_keypts, dim=1, index=idx)
    return src_keypts, new_tgt_keypts


class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention):
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
        weight = torch.softmax(attention[:, None, :, :] * feat_attention, dim=-1)
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])
        message = self.fc_message(message)
        res = feat + message
        return res


class NonLocalNetFlow(nn.Module):
    def __init__(self, in_dim=6, num_layers=1, num_channels=128):
        super(NonLocalNetFlow, self).__init__()
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

    def forward(self, src_keypts, tgt_keypts, idx):
        """
        Input:
            - corr_feat:          [bs, in_dim, num_corr]   input feature map
            - corr_compatibility: [bs, num_corr, num_corr] spatial consistency matrix
        Output:
            - feat:               [bs, num_channels, num_corr] updated feature
        """

        src_keypts, tgt_keypts = pointutil(src_keypts.permute(0, 2, 1), tgt_keypts.permute(0, 2, 1), idx)
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / 1.2 ** 2, min=0)

        corr_feat = torch.cat((src_keypts, tgt_keypts), dim=-1).permute(0, 2, 1)
        corr_feat = corr_feat - corr_feat.mean(dim=-1, keepdim=True)
        feat = self.layer0(corr_feat)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat, corr_compatibility)
        return feat


class NonLocalNetCost(nn.Module):
    def __init__(self, in_dim=6, num_layers=2, num_channels=128):
        super(NonLocalNetCost, self).__init__()
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

    def forward(self, src_keypts, tgt_keypts_all):
        """
        Input:
            - corr_feat:          [bs, in_dim, num_corr]   input feature map
            - corr_compatibility: [bs, num_corr, num_corr] spatial consistency matrix
        Output:
            - feat:               [bs, num_channels, num_corr] updated feature
        """
        tgt_keypts = tgt_keypts_all.mean(dim=2)
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / 1.2 ** 2, min=0)
        corr_feat = torch.cat((src_keypts, tgt_keypts), dim=-1).permute(0, 2, 1)
        corr_feat = corr_feat - corr_feat.mean(dim=-1, keepdim=True)
        feat = self.layer0(corr_feat)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat, corr_compatibility)

        return feat



class NonLocalNetCostAll(nn.Module):
    def __init__(self, in_dim=6, num_layers=2, num_channels=128):
        super(NonLocalNetCostAll, self).__init__()
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



    def forward(self, src_keypts, tgt_keypts_all):
        """
        Input:
            - corr_feat:          [bs, in_dim, num_corr]   input feature map
            - corr_compatibility: [bs, num_corr, num_corr] spatial consistency matrix
        Output:
            - feat:               [bs, num_channels, num_corr] updated feature
        """
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        B, N, n, C = tgt_keypts_all.shape
        feat_all = []
        for i in range(n):
            tgt_keypts = tgt_keypts_all[:, :, i] # B 3 N
            corr_compatibility = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
            corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / 1.2 ** 2, min=0)
            corr_feat = torch.cat((src_keypts, tgt_keypts), dim=-1).permute(0, 2, 1)
            corr_feat = corr_feat - corr_feat.mean(dim=-1, keepdim=True)
            feat = self.layer0(corr_feat)
            for i in range(self.num_layers):
                feat = self.blocks[f'PointCN_layer_{i}'](feat)
                feat = self.blocks[f'NonLocal_layer_{i}'](feat, corr_compatibility)
            feat_all.append(feat)
        return torch.stack(feat_all, dim=2)