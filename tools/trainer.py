import os
import shutil

from param.summaryparam import *                # 参数配置文件设置
import numpy as np
import torch
from tqdm import tqdm

import logging
import copy


def load_parse(epoch, dataloader):
    dataloader.dataset.forwardmodel = True
    for j in range(0, epoch):
        for i, dataseq in enumerate(dataloader):
            print(j, epoch, i, len(dataloader))
    dataloader.dataset.forwardmodel = False


def save_py(savepath):
    def _ignore_copy_files(path, content):
        to_ignore = []
        for file_ in content:
            if file_ in ('__pycache__', "loam"):
                to_ignore.append(file_)
        return to_ignore
    newdir = os.path.join(savepath, "code")
    os.mkdir(newdir)
    shutil.copytree("./model/", os.path.join(newdir, "model"), ignore=_ignore_copy_files)
    shutil.copytree("./param/", os.path.join(newdir, "param"), ignore=_ignore_copy_files)
    shutil.copytree("./preprocess/", os.path.join(newdir, "preprocess"), ignore=_ignore_copy_files)
    shutil.copy("./tools/trainer.py", newdir)
    shutil.copy("./trainercom.py", newdir)


def abnormcheck(data):
    assert torch.sum(torch.isnan(data)) == 0
    assert torch.sum(torch.isinf(data)) == 0
    

def update_cfg(cfg):
    cfg["datasets"]["sequence-size"] = continlen - 1
    cfg["deeplio"]["lidar-feat-net"] = "lidar-feat-" + deepliolidarmodel
    extra_layer = 0
    if useremiss:
        extra_layer += 1
    if usedepth:
        extra_layer += 1
    cfg["datapreprocess"] = datapreprocess
    cfg["lidar-feat-" + deepliolidarmodel]["extra-layer"] = extra_layer
    cfg["vhacc"] = vhacc
    cfg["usedepth"] = usedepth
    if not useIMU:
        cfg["deeplio"]["imu-feat-net"] = None
    if cfg["lidar-feat-pointseg"]["ConvLSTMType"] == 'bid':
        cfg["lidar-feat-pointseg"]["num-layers"] = 1

    return cfg


def log(str, ptlog=True, pt=False):
    if ptlog:
        logging.info(str)
    if pt:
        print(str)


def euler2mat(euler):
    euler = euler / 180 * np.pi
    x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]
    B = euler.size(0)

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    one = torch.ones_like(cx)
    zero = torch.zeros_like(cx)

    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=1).view(B, 3, 3).cuda().type(ttype)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=1).view(B, 3, 3).cuda().type(ttype)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=1).view(B, 3, 3).cuda().type(ttype)

    rotMat = torch.bmm(torch.bmm(Rz, Ry), Rx)
    return rotMat


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    xw, yw, zw = x * w, y * w, z * w
    xy, xz, yz = y * x, z * x, z * y

    rotMat = torch.stack([x2 + y2 - z2 - w2, 2 * yz - 2 * xw, 2 * yw + 2 * xz,
                          2 * yz + 2 * xw, x2 - y2 + z2 - w2, 2 * zw - 2 * xy,
                          2 * yw - 2 * xz, 2 * zw + 2 * xy, x2 - y2 - z2 + w2], dim=1).reshape(B, 3, 3).cuda().type(
        ttype)
    return rotMat



def quat2rota(quat):
    if usesyqnet:
        rota = quat
    else:
        if iseuler:
            rota = euler2mat(quat)
        else:
            rota = quat2mat(quat)
    return rota

def cpu2cuda(dataseq):
    for i in range(0, len(dataseq)):
        data = dataseq[i]

        if i != 0:
            data["gt_translation"] = data["gt_translation"].cuda()
            data["gt_quaternion"] = data["gt_quaternion"].cuda()

            if circlelen > 1:
                data["now_rotation"] = data["now_rotation"].cuda()
                data["now_translation"] = data["now_translation"].cuda()

    return dataseq


def key_compare(orgdict, nowdict):

    result = {}
    def reshape_dict(mydict):
        for key in list(mydict.keys()):
            if isinstance(mydict[key], dict):
                for key_in in mydict[key]:
                    mydict[key + '-' + key_in] = mydict[key][key_in]
                del mydict[key]

        return mydict
    reshape_dict(orgdict)
    reshape_dict(nowdict)
    for key in list(orgdict.keys()):
        if isinstance(orgdict[key], list) or isinstance(orgdict[key], tuple):

            if set(orgdict[key]).difference(set(nowdict[key])) or set(nowdict[key]).difference(set(orgdict[key])):
                result[key] = nowdict[key]
            orgdict.pop(key)
            nowdict.pop(key)

    for key in list(nowdict.keys()):
        if isinstance(nowdict[key], list) or isinstance(nowdict[key], tuple):
            result[key] = nowdict[key]
            if key in orgdict:
                orgdict.pop(key)
            nowdict.pop(key)


    result.update((orgdict.items() - nowdict.items()))
    result.update((nowdict.items() - orgdict.items()))
    
    return result


def write_msg(epoch, batch, timeflag, optimizer, loss, modelstr, gtloss, gtstr, needloss):
    with open(os.path.join('./result/', timeflag, "msg", str(epoch) + '-msg.txt'), "a+") as f:
        if needloss:
            f.write("dataset:" + str(trainset) + "\n")
        else:
            f.write("dataset:" + str(testset) + "\n")
        f.write("   lr:%e\n" % optimizer.param_groups[0]["lr"])
        f.write(("batch:%d loss:%e gtloss%e\n" % (batch, loss.item(), gtloss)))
        f.write(modelstr + "\n")
        f.write(gtstr + "\n\n")



def proceed_two_cloud_deeplio(seq, seqlen, last_data, now_data_org, uncertainty, quat, trans, criterion, needloss, needgtloss=False, rotainput=False):
    if len(quat.shape) > 2:        #[10,3,3]
        rotainput = True
    else:
        rotainput = False
    if seqlen != 1:
        needgtloss = False


    now_data = copy.copy(now_data_org)
    # loss, gtloss, modelstr, gtstr = criterion(last_data, now_data, uncertainty, quat, trans, needgtloss, rotainput)

    if not onlypwcloss and (needloss or showtestloss):          #false true false

        # else: #NPointLoss
        loss, gtloss, modelstr, gtstr = criterion(last_data, now_data, quat, trans, needgtloss, rotainput)

        modelstr = "seq:%d-%d " % (seq, (seq + seqlen)) + modelstr
        print(modelstr)
    
    else:
        loss = torch.zeros(1).cuda().type(ttype)
        gtloss = torch.zeros(1).cuda().type(ttype)
        modelstr = "None"
        gtstr = "None"

    if seqlen == 1:  #true
        if len(quat.shape) > 2:
            rota = quat
        else:
            rota = quat2rota(quat)
        batchoutput = torch.cat((now_data["seqindex"][:, None].float(), now_data["bindex"][:, None].float() - 1,
                                 now_data["bindex"][:, None].float(), trans.cpu(), rota.cpu().view(-1, 9)), dim=1).detach().numpy()
        if len(reweight) != 1:
            batchoutput = np.concatenate((batchoutput, np.array([[reweight[seq]]]).repeat(repeats=now_data["bindex"].shape[0], axis=0)), axis=1)
    else:
        batchoutput = None

    return batchoutput, loss, gtloss, modelstr, gtstr




def proceed_one_voxelvlad(epoch, batch, timeflag, dataseq, model, optimizer, criterion, needgtloss, needloss, nowcontinlen):
    sumgtloss = torch.zeros(1).cuda().type(ttype)
    summodelstr = ""
    sumgtstr = ""

    lossratelist = cfgvoxel['lossratelist']
    modeltype = cfgvoxel['modeltype']
    if cfgvoxel["flowdirname"] is None:       #none
        nowpoints = dataseq[1]["ponintnor"].cuda()
        lastpoints = dataseq[0]["ponintnor"].cuda()

        netinput = (nowpoints, lastpoints)
        
    if cfgvoxel['multiflow']:             #True
        netout = model(list(netinput), showtestloss or needloss)


    quat, trans, flowinloss = netout     #list, len 5:  torch.Size([10, 3, 3])   list, len 5:  torch.Size([10, 3])   none
    poseoutput = {}
    abnormcheck(quat[0].data)
    abnormcheck(trans[0].data)

    lastindex = 0
    nowindex = 1
    
    print(dataseq[0]["bindex"])

    last_data = dataseq[lastindex]
    now_data = dataseq[nowindex]

    batchoutput, loss, gtloss, modelstr, gtstr = \
        proceed_two_cloud_deeplio(lastindex, nowindex - lastindex, last_data, now_data, None, quat[0], trans[0], criterion, needloss, needgtloss)

    if len(quat) > 1:

        for i in range(0, len(quat)):
            abnormcheck(quat[i].data)
            abnormcheck(trans[i].data)

            key = "layer"+str(i)
            poseoutput[key] = torch.cat((now_data["seqindex"][:, None].float(), now_data["bindex"][:, None].float() - 1,
                                 now_data["bindex"][:, None].float(), trans[i].cpu(), quat[i].cpu().view(-1, 9)), dim=1).detach().numpy()
        

    if showtestloss or needloss: #true
        loss *= lossratelist[0]
        gtloss *= lossratelist[0]

        if len(quat) > 1:
            if len(lossratelist) == 1:
                lossratelist *= len(quat)
            for i in range(1, len(quat)):
                abnormcheck(quat[i].data)
                abnormcheck(trans[i].data)
                
                _, flowoutloss,flowoutgtloss, _, _ = \
                    proceed_two_cloud_deeplio(lastindex, nowindex - lastindex, last_data, now_data, None, quat[i], trans[i], criterion, needloss, needgtloss, rotainput=True)
                
                loss += flowoutloss * lossratelist[i]
                gtloss += flowoutgtloss * lossratelist[i]

        summodelstr = summodelstr + modelstr + "\n"
        if needgtloss:
            sumgtstr = sumgtstr + gtstr + "\n"
            sumgtloss += gtloss


        write_msg(epoch, batch, timeflag, optimizer, loss, summodelstr, sumgtloss, sumgtstr, needloss)
        log("epochnet:%d  batch:%d   loss:%e   lr:%e\n\n" % (epoch, batch, loss.item(), optimizer.param_groups[0]["lr"]))
        print("\nepochnet:%d  batch:%d   loss:%e   lr:%e" % (epoch, batch, loss.item(), optimizer.param_groups[0]["lr"]))
        if needloss:

            loss.backward()
            
    return batchoutput, poseoutput, len(quat)



def get_size(model):
    def model_size(model):
        para = sum([np.prod(list(p.size())) for p in model.parameters()])
        return 'Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000)

    resutl = [model_size(model)]

    if useflowvoxel:          #zbb
        if model.senceflownet is not None:
            resutl.append(model_size(model.senceflownet))
 
        if model.flowposenet is not None:
            resutl.append(model_size(model.flowposenet))

    return resutl



def proceed_one_epoch(epoch, timeflag, dataloader, model, optimizer, criterion, needloss, vaildmode=False):
    optimizer.zero_grad()

    needgtloss = False                   # work 2 dosen't need
    
    if needloss:                                # training need loss true
        model.train()
        stepsizelist = seqstepsize         #seqstepsize = [1]  # 跳跃取数据
    else:                                    # eval dosen't need loss
        model.eval()
        stepsizelist = [1]                    # seqstepsize = [1]  # 跳跃取数据  间隔 1 ？


    for nowseqstep in stepsizelist:            #
        outputdata = None
        poses = {}
        poses["layer0"] = None                       #zbb
        poses["layer1"] = None 
        poses["layer2"] = None 
        poses["layer3"] = None 
        uniqueline_ = {}
        uniquedata_ = {}
        dataloader.dataset.seqstepsize = nowseqstep
        for i, dataseq in enumerate(tqdm(dataloader)):

            dataseq = cpu2cuda(dataseq)

            if useflowvoxel:          #true
                proceed_fun = proceed_one_voxelvlad

            if testcontin2 and not needloss:
                nowcontinlen = 2
            else:
                nowcontinlen = continlen
            #zbb  batchoutput:[32,15]   
            batchoutput, poseoutput, layernum = proceed_fun(epoch, i, timeflag, dataseq, model, optimizer, criterion, needgtloss, needloss, nowcontinlen)
            
            
            if outputdata is None:
                outputdata = batchoutput
            else:
                outputdata = np.concatenate((outputdata, batchoutput), axis=0)
                

            for i in range(0, layernum):

                key = "layer"+str(i)
                if poses[key] is None:
                    poses[key] = poseoutput[key]
                else:
                    poses[key] = np.concatenate((poses[key], poseoutput[key]), axis=0)
            

            if needloss:
                if not epochstep:
                    if (i + 1) % batchstep == 0 or i + 1 == len(dataloader):
                        # print("oK")
                        optimizer.step()
                        optimizer.zero_grad()
        if needloss and epochstep:
            optimizer.step()
            optimizer.zero_grad()

        if nowseqstep == 1:
            if len(reweight) != 1:   #===1
                orderline = np.argsort(outputdata[:, -1])[::-1]
                outputdata = outputdata[orderline, :-1]
            _, uniqueline = np.unique(outputdata[:, :2], axis=0, return_index=True)
            uniquedata = outputdata[uniqueline, :]
            uniquedata = uniquedata[uniquedata[:, 1] >= 0]

            for i in range(0, layernum):

                key = "layer"+str(i)
                _, uniqueline_[key] = np.unique(poses[key][:, :2], axis=0, return_index=True)
                uniquedata_[key] = poses[key][uniqueline_[key], :]
                uniquedata_[key] = uniquedata_[key][uniquedata_[key][:, 1] >= 0]

                np.savetxt(os.path.join('./result/', timeflag, "data", str(epoch) + key + '.txt'), uniquedata_[key], fmt='%.6e')

            if needloss:
                np.savetxt(os.path.join('./result/', timeflag, "data", str(epoch) + '.txt'), uniquedata, fmt='%.6e')
            else:
                if vaildmode:
                    np.savetxt(os.path.join('./result/', timeflag, "data", str(epoch) + 'valid.txt'), uniquedata, fmt='%.6e')
                else:
                    np.savetxt(os.path.join('./result/', timeflag, "data", str(epoch) + 'test.txt'), uniquedata, fmt='%.6e')


