import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
import time
from preprocess.map import Map
from preprocess.scanread import ScanRead


# from autolab_core import RigidTransform

def log(str, ptlog=True):
    if ptlog:
        logging.info(str)
    # print(str)


def abnormcheck(data):
    assert torch.sum(torch.isnan(data)) == 0
    assert torch.sum(torch.isinf(data)) == 0


def euler2mat(euler):
    euler = euler / 180 * np.pi
    x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]
    B = euler.size(0)

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    one = torch.ones_like(cx)
    zero = torch.zeros_like(cx)
    # print(np.shape(x))
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=1).view(B, 3, 3).cuda().type(euler.dtype)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=1).view(B, 3, 3).cuda().type(euler.dtype)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=1).view(B, 3, 3).cuda().type(euler.dtype)

    rotMat = torch.bmm(torch.bmm(Rz, Ry), Rx)
    return rotMat





def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)
    # print(np.shape(x))
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    xw, yw, zw = x * w, y * w, z * w
    xy, xz, yz = y * x, z * x, z * y

    rotMat = torch.stack([x2 + y2 - z2 - w2, 2 * yz - 2 * xw, 2 * yw + 2 * xz,
                          2 * yz + 2 * xw, x2 - y2 + z2 - w2, 2 * zw - 2 * xy,
                          2 * yw - 2 * xz, 2 * zw + 2 * xy, x2 - y2 - z2 + w2], dim=1).reshape(B, 3, 3).cuda().type(
        quat.dtype)
    return rotMat


def batch_inverse(batchrota):
    result = []
    for rota in batchrota:
        result.append(rota.inverse().unsqueeze(0))
    return torch.stack(result, dim=0)


def print_top100and10(loss, type, name, needgtloss):
    loss100, _ = torch.topk(loss, k=100)
    loss10, _ = torch.topk(loss, k=10)
    # icpdropsortloss = icpsortloss[:int(-icploss.shape[0] * self.drop)]
    # self.timetest("12")
    log(type + " " + name + " 100avg:%15e 10avg:%15e" % (torch.sum(loss100).item() / 100, torch.sum(loss10).item() / 10)
        , needgtloss)

class PointLoss(nn.Module):

    def __init__(self, lamda, beta, thetac, drop, ttype, needgtloss=True, sx=0.0, sq=-2.5, h=0.5, v=0.5, fov_h=360,
                 fov_vu=3, fov_v=26, uncerlow=0.1, uncerhigh=1, uncerlograte=3, iseuler=True, onlygt=False, usegt=False,
                 useuncer=True, vhacc=False,useorgsetting=False, lowlimit=False, usesemantic=False, anglelosstype=None,
                 doubleloss=False, lonetuncer=False, muluncerloss=False, l2icp=False, lonetgt=False, prweight=1,
                 lonetwidth=False, loamheight=False, icpthre=[0.0, 10000.0], angthre=[0.0, 10000.0], anglelossrate=1.0, equalweight=False,
                 epochthre=0, movegrid=True, l2icptheta=10000.0,parallelloss=False,fordnewheight=False):
        super(PointLoss, self).__init__()
        if lonetwidth:
            h = 0.2
        if vhacc:
            fov_v = 32
        if fordnewheight:
            fov_vu = 10.8
            fov_v = 41.6
            v = 0.8
        self.delta_h = h
        self.delta_v = v
        self.fov_h = fov_h
        self.fov_vu = fov_vu
        # self.fov_vl = fov_v
        self.uncerlow = uncerlow
        self.uncerhigh = uncerhigh
        self.vhacc = vhacc
        self.proj_H = int(fov_v / v)
        self.proj_W = int(fov_h / h)
        # self.lamda = torch.Tensor([lamda]).cuda().type(ttype)
        self.lamda = lamda
        # self.beta = torch.Tensor([beta]).cuda().type(ttype)
        self.beta = beta
        self.thetac = thetac
        self.drop = drop
        self.epochthre = epochthre
        self.nowepoch = 1
        self.needgtloss = needgtloss
        self.useorgsetting = useorgsetting
        self.lowlimit = lowlimit
        self.usesemantic = usesemantic
        self.anglelosstype = anglelosstype
        self.anglelossrate = anglelossrate
        self.doubleloss = doubleloss
        self.lonetuncer = lonetuncer
        self.uncerlograte = uncerlograte
        self.muluncerloss = muluncerloss
        self.l2icp = l2icp
        self.lonetgt = lonetgt
        self.prweight = prweight
        self.backloss = False
        self.starttime = None
        self.loamheight = loamheight
        self.icpthre = icpthre
        self.angthre = angthre
        self.equalweight = equalweight
        self.movegrid = movegrid
        self.l2icptheta = l2icptheta
        self.x_index = torch.zeros(self.proj_H * self.proj_W).type(torch.long).cuda()
        self.y_index = torch.zeros(self.proj_H * self.proj_W).type(torch.long).cuda()
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=True)
        if iseuler:
            self.sq = nn.Parameter(torch.Tensor([sx]), requires_grad=True)
        else:
            self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=True)
        self.iseuler = iseuler
        self.onlygt = onlygt
        self.usegt = usegt
        self.useuncer = useuncer
        self.ttype = ttype
        self.parallelloss = parallelloss
        for i in range(self.proj_H):
            for j in range(self.proj_W):
                self.y_index[i * self.proj_W + j] = i
                self.x_index[i * self.proj_W + j] = j

        self.modelstr = []
        self.gtstr = []
        self.gtloss = []

    def dot(self, nor, ver):
        return ver[:, 0] * nor[:, 0] + ver[:, 1] * nor[:, 1] + ver[:, 2] * nor[:, 2]

    def timetest(self, flag):
        print(flag, time.time() - self.starttime)
        self.starttime = time.time()

    def loss_core(self, vertexmap_proj, normalmap_proj, uncertainty, now_data, last_data, needgtloss, type):
        icpbatchloss = torch.zeros(1).cuda().type(self.ttype)
        angbatchloss = torch.zeros(1).cuda().type(self.ttype)
        fovbatchloss = torch.zeros(1).cuda().type(self.ttype)
        fov_count_sum = 0
        icp_count_sum = 0
        ang_count_sum = 0
        conf_count_sum = 0
        mask_count_sum = 0
        seman_count_sum = 0
        uncer_count_sum = 0
        B, H, W, _ = vertexmap_proj.shape

        normal_mask = torch.sum(torch.abs(now_data["normalmap"]), dim=-1) != 0
        normal_lastmask = torch.sum(torch.abs(last_data["normalmap"]), dim=-1) != 0
        for i in range(0, B):
            log("%d %d" % (now_data["seqindex"][i].item(), now_data["bindex"][i].item()), needgtloss)
            if self.useuncer or self.lonetuncer:
                log(type + " uncer:%15e%15e%15e" % (
                    torch.min(uncertainty[i]).item(), torch.max(uncertainty[i]).item(),
                    torch.mean(uncertainty[i]).item()),
                    needgtloss)


            scan_proj = vertexmap_proj[i].view(H * W, 3)
            x_index = torch.Tensor.clone(self.x_index)
            y_index = torch.Tensor.clone(self.y_index)

            maskcheck = ((now_data["maskmap"][i, y_index, x_index] > 0) * normal_mask[i, y_index, x_index])
            x_index = x_index[maskcheck]
            y_index = y_index[maskcheck]
            scan_proj = scan_proj[maskcheck]

            if self.movegrid:
                depth = torch.norm(scan_proj, 2, dim=1)
                # print(now_data["bindex"][i].item(), torch.sum(depth == 0))


                # depthcheck = depth > 0
                # depth = depth[depthcheck]
                # x_index = x_index[depthcheck]
                # y_index = y_index[depthcheck]
                # scan_proj = scan_proj[depthcheck]


                # self.timetest("4")
                # get depth of all points

                # depth = torch.norm(scan_proj, 2, dim=1)

                # get scan components
                scan_x = scan_proj[:, 0]
                scan_y = scan_proj[:, 1]
                scan_z = scan_proj[:, 2]

                abnormcheck(scan_proj)
                abnormcheck(depth)
                # self.timetest("5")
            # get angles of all points

                yaw = torch.atan2(scan_y, scan_x) / np.pi * 180
                pitch = torch.asin(scan_z / depth) / np.pi * 180

                abnormcheck(yaw)
                abnormcheck(pitch)
                # get projections in image coords
                proj_x = (self.fov_h / 2 - yaw) / self.delta_h
                if self.vhacc or self.loamheight:
                    proj_y = (2 - pitch) * 3.0
                    proj_y2 = 64 / 2 + (-8.83 - pitch) * 2.0
                    proj_y[pitch < -8.83] = proj_y2[pitch < -8.83]
                else:
                    proj_y = (self.fov_vu - pitch) / self.delta_v
                # self.timetest("6")
                # fovloss = torch.sum(torch.abs(proj_x[proj_x < 0]))
                # fovloss += torch.sum(torch.abs(proj_y[proj_y < 0]))
                # fovloss += torch.sum(torch.abs(proj_x[proj_x >= self.proj_W] - (self.proj_W - 1)))
                # fovloss += torch.sum(torch.abs(proj_y[proj_y >= self.proj_H] - (self.proj_H - 1)))

                abnormcheck(proj_x)
                abnormcheck(proj_y)

                fovloss = torch.sum(F.mse_loss(proj_x, proj_x.clamp(0, self.proj_W - 1))) + \
                          torch.sum(F.mse_loss(proj_y, proj_y.clamp(0, self.proj_H - 1)))
                # self.timetest("7")

                proj_rx = torch.round(proj_x)
                proj_ry = torch.round(proj_y)
                # tx = torch.round(proj_x)
                # ty = torch.round(proj_y)

                left = 0 <= proj_rx
                right = proj_rx < self.proj_W
                xfovheck = left * right

                left = 0 <= proj_ry
                right = proj_ry < self.proj_H
                yfovcheck = left * right

                fovcheck = xfovheck * yfovcheck

                # ty = proj_ry[fovcheck].type(torch.long)
                # tx = proj_rx[fovcheck].type(torch.long)
                proj_ry = proj_ry[fovcheck].type(torch.long)
                proj_rx = proj_rx[fovcheck].type(torch.long)

                y_index = y_index[fovcheck]
                x_index = x_index[fovcheck]
            else:
                fovloss = torch.zeros(1).cuda().type(self.ttype)
                fovcheck = torch.ones_like(y_index)
                proj_ry = y_index.clone()
                proj_rx = x_index.clone()
            ##########################
            # depth = depth[fovcheck]
            # scan = ScanRead(ntype=np.float32, vhacc=self.vhacc, useremiss=False, usesemantic=self.usesemantic)
            # points = scan_proj.view(-1, 3).detach().cpu().numpy()
            # scan.points = points
            # scan.do_range_projection()
            # now_map = Map(scan, ntype=np.float32, x55norm=True)
            # now_map.get_map()
            # rotanormal = normalmap.detach().cpu().numpy()
            #
            # cacunormal = np.zeros_like(rotanormal)
            #
            # order = torch.argsort(depth, descending=True)
            # proj_rx = proj_rx[order]
            # proj_ry = proj_ry[order]
            # y_index = y_index[order]
            # x_index = x_index[order]
            #
            # cacunormal[y_index.cpu().numpy(), x_index.cpu().numpy()] = \
            #     now_map.normalmap[proj_ry.type(torch.long).detach().cpu().numpy(), proj_rx.type(torch.long).detach().cpu().numpy()]
            # # cacunormal[proj_ry.type(torch.long).detach().cpu().numpy(), proj_rx.type(torch.long).detach().cpu().numpy()] = \
            # #     now_map.normalmap[y_index.cpu().numpy(), x_index.cpu().numpy()]
            # rotaver = vertexmap_proj.detach().cpu().numpy()
            #
            # # cacuver[proj_ry.type(torch.long).detach().cpu().numpy(), proj_rx.type(torch.long).detach().cpu().numpy()] = \
            # #     now_map.vertexmap[y_index.cpu().numpy(), x_index.cpu().numpy()]
            #
            # a = rotaver[y_index.cpu().numpy(), x_index.cpu().numpy()] - now_map.vertexmap[proj_ry.type(torch.long).detach().cpu().numpy(), proj_rx.type(torch.long).detach().cpu().numpy()]
            # a = np.abs(a)
            # print(np.sum(a > 1))
            # b = rotaver[y_index.cpu().numpy(), x_index.cpu().numpy()] - scan.proj_xyz[scan.proj_y, scan.proj_x]
            # b = np.abs(b)
            # print(np.sum(b > 1))
            # # b = proj_rx.detach().cpu().numpy() - scan.proj_x
            # # c = fovcheck.detach().cpu().numpy() ^ scan.proj
            # # aa = now_map.vertexmap[proj_ry.type(torch.long).detach().cpu().numpy(), proj_rx.type(torch.long).detach().cpu().numpy()]
            # # bb = vertexmap_proj.detach().cpu().numpy()[proj_ry.type(torch.long).detach().cpu().numpy(), proj_rx.type(torch.long).detach().cpu().numpy()]
            # # a = rotaver[proj_ry.type(torch.long).detach().cpu().numpy(), proj_rx.type(torch.long).detach().cpu().numpy()] - \
            # #     now_map.vertexmap[scan.proj_y, scan.proj_x]
            ########################

            # depth = depth[fovcheck]
            # self.timetest("8")

            # maskcheck = now_data["maskmap"][i, y_index, x_index] > 0

            laskmaskcheck = ((last_data["maskmap"][i, proj_ry, proj_rx] > 0) * normal_lastmask[i, proj_ry, proj_rx])
            # self.timetest("9")
            proj_ry = proj_ry[laskmaskcheck]
            proj_rx = proj_rx[laskmaskcheck]
            y_index = y_index[laskmaskcheck]
            x_index = x_index[laskmaskcheck]

            concheck = now_data["confidencemap"][i, y_index, x_index] >= self.thetac
            proj_ry = proj_ry[concheck]
            proj_rx = proj_rx[concheck]
            y_index = y_index[concheck]
            x_index = x_index[concheck]


            # depth = depth[sumcheck]
            # self.timetest("10")
            # order = torch.argsort(depth)
            # proj_rx = proj_rx[order]
            # proj_ry = proj_ry[order]
            # y_index = y_index[order]
            # x_index = x_index[order]
            semanticcheck = torch.zeros_like(concheck)
            if self.usesemantic:
                semanticcheck = now_data["semanticmap"][i, y_index, x_index] == last_data["lastsemantic"][
                    i, proj_ry, proj_rx]
                proj_ry = proj_ry[semanticcheck]
                proj_rx = proj_rx[semanticcheck]
                y_index = y_index[semanticcheck]
                x_index = x_index[semanticcheck]
            # checkset = set()
            # repeatcheck = torch.BoolTensor((order.shape[0])).cuda()
            # map = (proj_ry + 100 * proj_rx).tolist()
            # for i in range(len(map)):
            #     if map[i] in checkset:
            #         repeatcheck[i] = False
            #     else:
            #         repeatcheck[i] = True
            #         checkset.add(map[i])
            #
            # proj_rx = proj_rx[repeatcheck]
            # proj_ry = proj_ry[repeatcheck]
            # y_index = y_index[repeatcheck]
            # x_index = x_index[repeatcheck]
            # icpmap = torch.zeros((H, W)).type(self.ttype).cuda()

            # icploss = \
            #     (torch.abs(self.dot(lastnormal[i, proj_ry, proj_rx],
            #                         (vertexmap_proj[i, y_index, x_index] - lastvertex[i, proj_ry, proj_rx])) *
            #                confidencemap[i, y_index, x_index])) / uncertainty[i, y_index, x_index] + \
            #     torch.log(uncertainty[i, y_index, x_index])

            # icploss = torch.abs(torch.sum(lastnormal[i, proj_ry, proj_rx] * (vertexmap_proj[i, y_index, x_index] - lastvertex[i, proj_ry, proj_rx]), dim=1) *
            #             confidencemap[i, y_index, x_index])
            if self.anglelosstype is not None:

                if self.anglelosstype == "angle":
                    anglelossN = torch.ones_like(y_index, dtype=self.ttype) - torch.abs(torch.sum(last_data["normalmap"][i, proj_ry, proj_rx] *
                                normalmap_proj[i, y_index, x_index], dim=1)) / torch.sqrt(torch.sum(last_data["normalmap"][i, proj_ry, proj_rx] ** 2, dim=1)
                                * torch.sum(normalmap_proj[i, y_index, x_index] ** 2, dim=1))
                    # anglelossN = torch.abs(torch.sum(last_data["normalmap"][i, proj_ry, proj_rx] * normalmap_proj[i, y_index, x_index], dim=1))
                    # print("sum", anglelossN.shape[0], ">30", torch.sum(anglelossN > 1 - 0.866).item(), ">60",  torch.sum(anglelossN > 1 - 0.5).item())
                elif self.anglelosstype == "global":
                    anglelossN = (torch.ones_like(y_index, dtype=self.ttype) - torch.abs(torch.sum(last_data["vertexmap"][i, proj_ry, proj_rx] *
                                vertexmap_proj[i, y_index, x_index], dim=1)) / torch.sqrt(torch.sum(last_data["vertexmap"][i, proj_ry, proj_rx] ** 2, dim=1)
                                * torch.sum(vertexmap_proj[i, y_index, x_index] ** 2, dim=1)))
                    # anglelossN = torch.sum(last_data["vertexmap"][i, proj_ry, proj_rx] * vertexmap_proj[i, y_index, x_index], dim=1) / \
                    #             torch.sqrt(torch.sum(last_data["vertexmap"][i, proj_ry, proj_rx] ** 2, dim=1) * torch.sum(vertexmap_proj[i, y_index, x_index] ** 2, dim=1))

                elif self.anglelosstype == "distance":
                    anglelossleftpart = torch.sum(
                        (normalmap_proj[i, y_index, x_index] - last_data["normalmap"][i, proj_ry, proj_rx]) ** 2,
                        dim=1)[:, None]
                    anglelossrightpart = torch.sum(
                        (-normalmap_proj[i, y_index, x_index] - last_data["normalmap"][i, proj_ry, proj_rx]) ** 2,
                        dim=1)[:, None]
                    anglelossN = torch.min(torch.cat((anglelossleftpart, anglelossrightpart), dim=1), dim=1)[0]
                    print_top100and10(anglelossN, type, "angorg", needgtloss)
                    # angleloss = torch.sum(F.l1_loss(torch.abs(lastnormal[i, proj_ry, proj_rx]), torch.abs(normalmap_proj[i, y_index, x_index])), dim=1) * 10 \
                    #             * confidencemap[i, y_index, x_index]
                elif self.anglelosstype == "l1ang":
                    anglelossleftpartflag = torch.sum(
                        (normalmap_proj[i, y_index, x_index] - last_data["normalmap"][i, proj_ry, proj_rx]) ** 2,
                        dim=1)[:, None]
                    anglelossrightpartflag = torch.sum(
                        (-normalmap_proj[i, y_index, x_index] - last_data["normalmap"][i, proj_ry, proj_rx]) ** 2,
                        dim=1)[:, None]
                    anglelossflag = torch.min(torch.cat((anglelossleftpartflag, anglelossrightpartflag), dim=1), dim=1)[0]
                    anglelossleftpart = torch.sum(
                        torch.abs(normalmap_proj[i, y_index, x_index] - last_data["normalmap"][i, proj_ry, proj_rx]),
                        dim=1)[:, None]
                    anglelossrightpart = torch.sum(
                        torch.abs(-normalmap_proj[i, y_index, x_index] - last_data["normalmap"][i, proj_ry, proj_rx]),
                        dim=1)[:, None]
                    anglelossN = torch.min(torch.cat((anglelossleftpart, anglelossrightpart), dim=1), dim=1)[0]
                    print_top100and10(anglelossN, type, "angorg", needgtloss)
                if self.drop is not None and self.nowepoch > self.epochthre:
                    _, angtopk = torch.topk(anglelossN, k=int(anglelossN.shape[0] * self.drop))
                    angvaluecheck = torch.ones_like(anglelossN)
                    angvaluecheck[angtopk] = 0
                    angvaluecheck = angvaluecheck.bool()
                    print_top100and10(anglelossN[angvaluecheck], type, "icpdrop", needgtloss)
                elif (self.angthre[1] < 10000.0 or self.angthre[0] > 0.0) and self.nowepoch > self.epochthre:
                    if self.anglelosstype == "l1ang":
                        angvaluecheck = ((anglelossflag <= self.angthre[1]) * (anglelossflag >= self.angthre[0]))
                    else:
                        angvaluecheck = ((anglelossN <= self.angthre[1]) * (anglelossN >= self.angthre[0]))
                    # print("sum", anglelossN.shape[0], "drop", torch.sum(angvaluecheck).item())
                    # anglelossN *= 10
                    print_top100and10(anglelossN[angvaluecheck], type, "angdrop", needgtloss)
                else:
                    angvaluecheck = torch.ones_like(proj_ry).bool()
                angleloss = anglelossN * now_data["confidencemap"][i, y_index, x_index]
            else:
                angleloss = torch.zeros_like(proj_ry)
                angvaluecheck = torch.ones_like(proj_ry).bool()

            icploss = last_data["normalmap"][i, proj_ry, proj_rx] * (
                        vertexmap_proj[i, y_index, x_index] - last_data["vertexmap"][i, proj_ry, proj_rx])

            icplossN = torch.abs(torch.sum(icploss, dim=1))
            print_top100and10(icplossN, type, "icporg", needgtloss)
            if self.drop is not None and self.nowepoch > self.epochthre:
                _, icptopk = torch.topk(icplossN, k=int(icplossN.shape[0] * self.drop))
                icpvaluecheck = torch.ones_like(icplossN)
                icpvaluecheck[icptopk] = 0
                icpvaluecheck = icpvaluecheck.bool()
                print_top100and10(icplossN[icpvaluecheck], type, "icpdrop", needgtloss)
            elif (self.icpthre[1] < 10000.0 or self.icpthre[0] > 0.0) and self.nowepoch > self.epochthre:
                icpvaluecheck = ((icplossN <= self.icpthre[1]) * (icplossN >= self.icpthre[0]))
                print_top100and10(icplossN[icpvaluecheck], type, "icpdrop", needgtloss)
            else:
                icpvaluecheck = torch.ones_like(proj_ry).bool()

            if self.l2icp and self.nowepoch > self.epochthre:
                # icploss = icploss ** 2 / 2
                # {N1(v1-u1)+N2(v2-u2)+N3(v3-u3)}^2 求导为 dvi=Ni{N1(v1-u1)+N2(v2-u2)+N3(v3-u3)}
                # icploss = torch.abs(torch.sum(icploss, dim=1)) ** 2 / 2 * confidencemap[i, y_index, x_index]

                icplossN = icplossN.detach().clamp(0, self.l2icptheta)
                # print(icplossN.grad)
                icploss = torch.abs(torch.sum(icploss, dim=1) * icplossN * now_data["confidencemap"][i, y_index, x_index])
            else:
                icploss = torch.abs(torch.sum(icploss, dim=1) * now_data["confidencemap"][i, y_index, x_index])

            icppuresumloss = torch.sum(icplossN)
            if self.useuncer:
                # uncercheck = icploss > 0
                # y_index = y_index[uncercheck]
                # x_index = x_index[uncercheck]
                # icploss = icploss[uncercheck]
                # icploss /= uncertainty[i, y_index, x_index] + torch.log(uncertainty[i, y_index, x_index])
                if self.muluncerloss:
                    icploss = icploss * uncertainty[i, y_index, x_index] - torch.log(
                        uncertainty[i, y_index, x_index]) * self.uncerlograte
                else:
                    icploss = icploss / uncertainty[i, y_index, x_index] + torch.log(
                        uncertainty[i, y_index, x_index]) * self.uncerlograte
                uncercheck = (uncertainty[i, y_index, x_index] >= self.uncerlow) * (
                            uncertainty[i, y_index, x_index] <= self.uncerhigh)
                # icploss = icploss[uncercheck]
                if self.anglelosstype is not None:
                    # angleloss /= uncertainty[i, y_index, x_index] + torch.log(uncertainty[i, y_index, x_index])
                    if self.muluncerloss:
                        angleloss = angleloss * uncertainty[i, y_index, x_index] - torch.log(
                            uncertainty[i, y_index, x_index]) * self.uncerlograte
                    else:
                        angleloss = angleloss / uncertainty[i, y_index, x_index] + torch.log(
                            uncertainty[i, y_index, x_index]) * self.uncerlograte
                    # anglecheck = angleloss > 0
                    # angleloss = angleloss[anglecheck]

            elif self.lonetuncer:
                icploss *= uncertainty[i, y_index, x_index]
                if self.anglelosstype is not None:
                    angleloss *= uncertainty[i, y_index, x_index]
                uncercheck = uncertainty[i, y_index, x_index] == 1
            else:
                uncercheck = torch.zeros_like(icploss)
                # print("vertexmap_proj:", torch.sum(torch.isnan(vertexmap_proj[i, y_index, x_index])).item())
                # print("lastnormal:", torch.sum(torch.isnan(lastnormal[i, proj_ry, proj_rx])).item())
                # print("lastvertex:", torch.sum(torch.isnan(lastvertex[i, proj_ry, proj_rx])).item())
                # print("confidencemap:", torch.sum(torch.isnan(lastvertex[i, y_index, x_index])).item())
                # print("lastnormal", torch.sum(torch.isinf(lastnormal)))
                # print("???:", torch.sum(torch.isnan(lastnormal[i, proj_ry, proj_rx] * (vertexmap_proj[i, y_index, x_index] - lastvertex[i, proj_ry, proj_rx]))).item())
                # print("222:", torch.sum(torch.isnan(torch.sum(lastnormal[i, proj_ry, proj_rx] * (vertexmap_proj[i, y_index, x_index] - lastvertex[i, proj_ry, proj_rx]),dim=1))).item())

                # print("icploss:", torch.sum(torch.isnan(icploss)).item())
            log(type + " fovcheck:%6d maskcheck:%6d" % (torch.sum(fovcheck).item(), torch.sum(laskmaskcheck).item()),
                needgtloss)
            # self.timetest("11")
            # print(type, "maskmap:%6d last_data["maskmap"]:%6d" % (torch.sum(maskmap).item(), torch.sum(last_data["maskmap"]).item()))
            # print(type, "maskcheck:%6d laskmaskcheck:%6d" % (torch.sum(maskcheck).item(), torch.sum(laskmaskcheck).item()))
            # print(type, "sumcheck:%6d repeatcheck:%6d" % (torch.sum(sumcheck).item(), torch.sum(repeatcheck).item()))tic:
            log(type + " concheck:%6d semanticcheck:%6d" % (
                torch.sum(concheck).item(), torch.sum(semanticcheck).item()), needgtloss)
            # icploss /= self.lamda
            # icpmap[i, y_index, x_index] = \
            #     (torch.abs(torch.norm(vertexmap_proj[i, y_index, x_index] - lastvertex[i, proj_ry, proj_rx], 2, dim=1) ** 2 *
            #                confidencemap[i, y_index, x_index])) / uncertainty[i, y_index, x_index] + \
            #     torch.log(uncertainty[i, y_index, x_index])

            #
            # icploss = \
            #     torch.abs(self.dot(lastnormal[i, proj_ry, proj_rx],
            #                         (vertexmap_proj[i, y_index, x_index] - lastvertex[i, proj_ry, proj_rx])) *
            #                confidencemap[i, y_index, x_index])

            # icpmap[i, y_index, x_index] = \
            #     torch.abs(torch.norm(vertexmap_proj[i, y_index, x_index] - lastvertex[i, proj_ry, proj_rx], 2, dim=1) ** 2)

            # lastvertex_proj = torch.zeros_like(lastvertex).cuda()
            # lastnormal_proj = torch.zeros_like(lastnormal).cuda()

            # lastvertex_proj[i, y_index, x_index] = lastvertex[i, proj_ry, proj_rx]
            # lastnormal_proj[i, y_index, x_index] = lastnormal[i, proj_ry, proj_rx]

            # print(torch.sum(check).item())
            # icploss[icploss < 0] = 0
            icploss = icploss[icpvaluecheck * angvaluecheck]
            if self.anglelosstype is not None:
                angleloss = angleloss[icpvaluecheck * angvaluecheck]
            # if self.drop is not None:
            #     icptop100, _ = torch.topk(icploss, k=100)
            #     icptop10, _ = torch.topk(icploss, k=100)
            #     icpsortloss = icploss[torch.argsort(icploss)]
            #     # icpdropsortloss = icpsortloss[:int(-icploss.shape[0] * self.drop)]
            #     # self.timetest("12")
            #     log(type + " -100icpavg:%15e -10icpavg:%15e" % (
            #         torch.sum(icpsortloss[-100:]).item() / 100, torch.sum(icpsortloss[-10:]).item() / 10), needgtloss)
            #
            #     anglesortloss = angleloss[torch.argsort(icploss)]
            #     # icpdropsortloss = icpsortloss[:int(-icploss.shape[0] * self.drop)]
            #     # self.timetest("12")
            #     log(type + " -100angavg:%15e -10iangavg:%15e" % (
            #         torch.sum(anglesortloss[-100:]).item() / 100, torch.sum(anglesortloss[-10:]).item() / 10),
            #         needgtloss)
            #
            #     icpdropsortloss = icpsortloss[:int(-icploss.shape[0] * self.drop)]
            #     # self.timetest("13")
            #     log(type + " -100icpdropavg:%15e -10icpdropavg:%15e" % (
            #         torch.sum(icpdropsortloss[-100:]).item() / 100, torch.sum(icpdropsortloss[-10:]).item() / 10),
            #         needgtloss)
            #
            #     angledropsortloss = anglesortloss[:int(-icploss.shape[0] * self.drop)]
            #     # self.timetest("13")
            #     log(type + " -100angledropavg:%15e -10angledropavg:%15e" % (
            #         torch.sum(angledropsortloss[-100:]).item() / 100, torch.sum(angledropsortloss[-10:]).item() / 10),
            #         needgtloss)
            # else:
            #     # icpdropsortloss = icploss
            #     # angledropsortloss = angleloss
            #     icpdropsortloss = icploss[torch.argsort(icploss)]
            #     # self.timetest("12")
            #     log(type + " -100icpavg:%15e -10icpavg:%15e" % (
            #         torch.sum(icpdropsortloss[-100:]).item() / 100, torch.sum(icpdropsortloss[-10:]).item() / 10),
            #         needgtloss)
            #
            #     angledropsortloss = angleloss[torch.argsort(icploss)]
            #     # self.timetest("12")
            #     log(type + " -100angavg:%15e -10iangavg:%15e" % (
            #         torch.sum(angledropsortloss[-100:]).item() / 100, torch.sum(angledropsortloss[-10:]).item() / 10),
            #         needgtloss)

            icpsumloss = torch.sum(icploss)
            anglesumloss = torch.sum(angleloss)
            if self.lonetuncer:
                Lr = - torch.log(torch.sum(uncercheck) / uncercheck.shape[0]) / 3
                icpsumloss += Lr
            # elif self.useuncer:
            #     # Lr = - torch.log(torch.sum(uncercheck, dtype=torch.float32) / uncercheck.shape[0]) / self.uncerlow
            #     Lr = torch.sum(torch.log(uncertainty[i, y_index, x_index])) * self.uncerlograte
            #     icpsumloss += Lr
            else:
                Lr = torch.Tensor([0])
            log(type + " fovloss:%15e icploss:%15e icppureloss:%15e, angleloss:%15e" % (
                fovloss.item(), icpsumloss.item() / (icploss.shape[0] + 1e-5)
                * (self.proj_H * self.proj_W), icppuresumloss.item() / (icploss.shape[0] + 1e-5)
                * (self.proj_H * self.proj_W), anglesumloss.item() / (angleloss.shape[0] + 1e-5)
                * (self.proj_H * self.proj_W)), needgtloss)
            # self.timetest("14")

            fov_count_sum += torch.sum(fovcheck).item()
            icp_count_sum += torch.sum(icpvaluecheck).item()
            ang_count_sum += torch.sum(angvaluecheck).item()
            mask_count_sum += torch.sum(laskmaskcheck).item()
            conf_count_sum += torch.sum(concheck).item()
            seman_count_sum += torch.sum(semanticcheck).item()
            uncer_count_sum += torch.sum(uncercheck).item()

            if self.equalweight:
                icpsumloss = icpsumloss / (icploss.shape[0] + 1e-5) * (self.proj_H * self.proj_W)
                anglesumloss = anglesumloss / (angleloss.shape[0] + 1e-5) * (self.proj_H * self.proj_W)

            fovbatchloss += fovloss
            icpbatchloss += icpsumloss
            angbatchloss += anglesumloss
        ouput = {"fov_count_sum": fov_count_sum, "mask_count_sum": mask_count_sum, "conf_count_sum": conf_count_sum,
                 "seman_count_sum": seman_count_sum, "uncer_count_sum": uncer_count_sum, "fovloss": fovbatchloss,
                 "icploss": icpbatchloss, "angloss": angbatchloss, "icp_count_sum": icp_count_sum,
                 "ang_count_sum": ang_count_sum}
        return ouput

    def quat2mat_numpy(self, quat):
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        # print(np.shape(x))
        w2, x2, y2, z2 = w * w, x * x, y * y, z * z
        xw, yw, zw = x * w, y * w, z * w
        xy, xz, yz = y * x, z * x, z * y

        rotMat = np.array([x2 + y2 - z2 - w2, 2 * yz - 2 * xw, 2 * yw + 2 * xz,
                           2 * yz + 2 * xw, x2 - y2 + z2 - w2, 2 * zw - 2 * xy,
                           2 * yw - 2 * xz, 2 * zw + 2 * xy, x2 - y2 - z2 + w2]).reshape(3, 3)
        return rotMat

    def data_trans(self, data, quat, trans, name, rotainput):
        vertexmap = data["vertexmap"]
        normalmap = data["normalmap"]
        B, H, W, _ = vertexmap.shape
        if rotainput and name == "model":
            rota = quat
        else:
            if self.iseuler:
                rota = euler2mat(quat)
            else:
                rota = quat2mat(quat)
        ver_3 = vertexmap.permute(0, 3, 1, 2).view((B, 3, H * W))
        normalmap_proj = rota.bmm(normalmap.permute(0, 3, 1, 2).view((B, 3, H * W))).view(B, 3, H, W) \
            .permute(0, 2, 3, 1)
        vertexmap_proj = (rota.bmm(ver_3) + trans.unsqueeze(2)).view((B, 3, H, W)).permute(0, 2, 3, 1)

        return normalmap_proj, vertexmap_proj

    def getresultstr(self, B, output, prloss, type):
        if prloss is None:
            prloss = torch.zeros(1).cuda().type(self.ttype)
        datastr = type + "avg:fov-N:%6d mask-N:%6d conf-N:%6d seman-N:%6d uncer-N:%6d icp-N:%6d ang-N:%6d\n" \
                   "         fov:%15e icp:%15e ang:%15e pr:%15e" % (output["fov_count_sum"] / B,
                                                                    output["conf_count_sum"] / B,
                                                                    output["mask_count_sum"] / B,
                                                                    output["seman_count_sum"] / B,
                                                                    output["uncer_count_sum"] / B,
                                                                    output["icp_count_sum"] / B,
                                                                    output["ang_count_sum"] / B,
                                                                    (output["fovloss"] / B).item(),
                                                                    (output["icploss"] / B).item(),
                                                                    (output["angloss"] / B).item(),
                                                                    (prloss / B).item())
        return datastr


    def get_loss(self, last_data, now_data, uncertainty, quat, trans, needgtloss, rotainput):
        needgtloss = False
        B, H, W, _ = now_data["vertexmap"].shape
        if not rotainput:
            if self.onlygt or self.usegt:
                if self.backloss:
                    loss_x = F.mse_loss(trans, last_data["bkgt_translation"], reduction="sum")
                    loss_q = F.mse_loss(quat, last_data["bkgt_quaternion"], reduction="sum")
                else:
                    loss_x = F.mse_loss(trans, now_data["gt_quaternion"], reduction="sum")
                    loss_q = F.mse_loss(quat, now_data["gt_translation"], reduction="sum")
            else:
                if self.lowlimit:
                    loss_x = F.mse_loss(trans, trans.clamp(-10, 10))
                else:
                    loss_x = F.mse_loss(trans, trans.clamp(-1, 3))
                if self.iseuler:
                    if self.lowlimit:
                        loss_q1 = F.mse_loss(quat[:, :2], quat[:, :2].clamp(-15, 15))
                        loss_q2 = F.mse_loss(quat[:, 2], quat[:, 2].clamp(-15, 15))
                    else:
                        loss_q1 = F.mse_loss(quat[:, :2], quat[:, :2].clamp(-2, 2))
                        loss_q2 = F.mse_loss(quat[:, 2], quat[:, 2].clamp(-5, 5))
                else:
                    if self.lowlimit:
                        loss_q1 = F.mse_loss(quat[:, 1:], quat[:, 1:].clamp(-5e-1, 5e-1)) * 1e2
                        loss_q2 = F.mse_loss(quat[:, 0], quat[:, 0].clamp(0.9, 1)) * 1e4
                    else:
                        loss_q1 = F.mse_loss(quat[:, 1:], quat[:, 1:].clamp(-5e-2, 5e-2)) * 1e2
                        loss_q2 = F.mse_loss(quat[:, 0], quat[:, 0].clamp(0.999, 1)) * 1e4
                loss_q = loss_q1 + loss_q2

            if self.onlygt:
                # prloss = torch.exp(-self.sx) * loss_x + self.sx + torch.exp(-self.sq) * loss_q * self.beta + self.sq
                if self.lonetgt:
                    prloss = torch.exp(-self.sx) * loss_x + self.sx + torch.exp(-self.sq) * loss_q + self.sq
                else:
                    prloss = loss_x + loss_q * self.beta
            else:
                prloss = torch.exp(-self.sx) * loss_x + torch.exp(-self.sq) * loss_q
            log("prloss:%15e%15e%15e%15e%15e" % (
                loss_x.item(), loss_q.item(), prloss.item(), self.sx.item(), self.sq.item()))
        else:
            prloss = torch.zeros(1).cuda().type(self.ttype)
        # if self.useuncer and not self.useorgsetting:
        #     uncerloss = F.mse_loss(uncertainty, uncertainty.clamp(0.1, 1))
        # else:
        #     uncerloss = torch.Tensor([0]).cuda().type(self.ttype)
        # prloss *= (self.proj_W * self.proj_H * 1e4)

        if not self.onlygt:
            if self.useuncer:
                # uncertainty_proj = torch.sqrt(torch.sqrt(torch.abs(uncertainty.view(B, H, W))))
                # uncertainty_proj = torch.pow(uncertainty.view(B, H, W), 2)
                uncertainty_proj = F.sigmoid(uncertainty.view(B, H, W))
            elif self.lonetuncer:
                uncertainty_proj = torch.argmax(uncertainty, dim=3).view(B, H, W)
            else:
                uncertainty_proj = None
            # print(quat, trans)
            normalmap_proj, vertexmap_proj = self.data_trans(now_data, quat, trans, "model", rotainput)
            output = self.loss_core(vertexmap_proj, normalmap_proj, uncertainty_proj, now_data, last_data, needgtloss,
                                    "model")
            modelstr = self.getresultstr(B, output, prloss, "model")
            log(modelstr)

            if needgtloss:
                if self.backloss:
                    gt_normalmap_proj, gt_vertexmap_proj = self.data_trans(now_data, now_data["bkgt_quaternion"],
                                                                           now_data["bkgt_translation"], "gt",
                                                                           rotainput)
                else:
                    gt_normalmap_proj, gt_vertexmap_proj = self.data_trans(now_data, now_data["gt_quaternion"],
                                                                           now_data["gt_translation"], "gt", rotainput)
                gt_output = self.loss_core(gt_vertexmap_proj, gt_normalmap_proj, uncertainty_proj, now_data, last_data,
                                           needgtloss, "gt")
                gtstr = self.getresultstr(B, gt_output, None, "gt")
                log(gtstr)

                return output["angloss"] * self.anglelossrate + output["icploss"] + output["fovloss"] * self.lamda + \
                       prloss * self.prweight, gt_output["angloss"] * self.anglelossrate + gt_output["icploss"] + gt_output[
                           "fovloss"] * self.lamda, modelstr, gtstr
            else:
                return output["angloss"] * self.anglelossrate + output["icploss"] + output[
                    "fovloss"] * self.lamda + prloss * self.prweight, 0, modelstr, "None"
        else:
            modelstr = "modelavg: pr:%15e" % (prloss / B).item()
            return prloss * self.prweight, 0, modelstr, "None"

    def forward(self, last_data, now_data, uncertainty, quat, trans, needgtloss, rotainput):
        self.backloss = False
        loss, gtloss, modelstr, gtstr = self.get_loss(last_data, now_data, uncertainty, quat, trans, needgtloss,
                                                      rotainput)

        if self.doubleloss:
            self.backloss = True

            if rotainput:
                ba_quat = batch_inverse(quat)
            else:
                ba_quat = quat * -1
                if not self.iseuler:
                    ba_quat[:, 0] *= -1
            ba_trans = trans * -1
            if needgtloss:
                last_data["bkgt_translation"] = now_data["gt_translation"] * -1
                bagt_quat = now_data["gt_quaternion"] * -1
                if not self.iseuler:
                    bagt_quat[:, 0] *= -1
                last_data["bkgt_quaternion"] = bagt_quat
            back_loss, back_gtloss, back_modelstr, back_gtstr = \
                self.get_loss(now_data, last_data, uncertainty, ba_quat, ba_trans, needgtloss, rotainput)

            if needgtloss:
                return loss + back_loss, gtloss + back_gtloss, "forward-" + modelstr + "\nbackward-" + back_modelstr, \
                       "forward-" + modelstr + "\nbackward-" + back_gtstr
            else:
                return loss + back_loss, 0, "forward-" + modelstr + "\nbackward-", "None"
        else:
            if self.parallelloss:
                self.modelstr.append(modelstr)
                self.gtstr.append(gtstr)
                self.gtloss.append(gtloss)
                return loss
            else:
                return loss, gtloss, modelstr, gtstr


class AccumulateLoss(nn.Module):
    def __init__(self, continlen, ttype, accumulate_beta=50):
        super(AccumulateLoss, self).__init__()
        self.continlen = continlen
        self.accumulate_beta = accumulate_beta
        self.ttype = ttype

    def cacu_loss(self, sixdom1, sixdom2, sixdom12):
        fuse_rota = sixdom1["rota"].bmm(sixdom2["rota"])
        fuse_trans = sixdom1["rota"].bmm(sixdom1["trans"][:, :, None]).squeeze(2) + sixdom2["trans"]
        rotaloss = torch.sum(torch.pow(fuse_rota - sixdom12["rota"], 2))
        transloss = torch.sum(torch.pow(fuse_trans - sixdom12["trans"], 2))
        return rotaloss * self.accumulate_beta + transloss

    def forward(self, sixdomdir):
        accumulateloss = torch.zeros(1).cuda().type(self.ttype)
        for i in range(0, self.continlen):
            for j in range(i + 2, self.continlen):
                for k in range(i + 1, j):
                    sixdom1 = sixdomdir[i * 10 + k]
                    sixdom2 = sixdomdir[k * 10 + j]
                    sixdom12 = sixdomdir[i * 10 + j]
                    accumulateloss += self.cacu_loss(sixdom1, sixdom2, sixdom12)
        return accumulateloss