import os
import socket
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    iplist = ip.split(".")
    return iplist[-2] + iplist[-1]

mac = get_host_ip()

deviceid = "-1"
useparrell = False

import time

import torch
import random
from preprocess.parser import Parser
from tools.trainer import *   
              
if useflowvoxel:                             
    from model.voxelnetvlad.voxelnetvlad import get_flow_model

from model.npointloss import NPointLoss
import torch.optim as optim
from torch.optim import lr_scheduler
import warnings
import datetime
import logging
import os

warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

np.random.seed(seednumnp)
random.seed(seednumnp)
torch.manual_seed(seednumcpu)
torch.cuda.manual_seed(seednumgpu)
torch.cuda.manual_seed_all(seednumgpu)

timeflag = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("(%Y-%m-%d %H-%M-%S)") + \
           ("-(%sx%d)" % (torch.cuda.get_device_name(0), torch.cuda.device_count()))
# ("-(%sx%d)-%s" % (devicename, torch.cuda.device_count(), mac))


if torch.cuda.device_count() == 1:
    timeflag += ("-%s" % mac)

os.mkdir("./result/" + timeflag)
os.mkdir("./result/" + timeflag + "/data")
os.mkdir("./result/" + timeflag + "/msg")
if not testmode:
    os.mkdir("./modelpkl/" + timeflag)
    save_py(os.path.join('./modelpkl/', timeflag))
if drop is not None:
    name = ("dr%d-la%.4f-be%d-th%.2f-c%d-ep%d-lr%s-sts%d-bs%d-ga%.2f" % (
        1 / drop, lamda, beta, thetac, c, num_epochs, lr, step_size, batch_size, gamma))
else:
    name = ("drNone-la%.4f-be%d-th%.2f-c%d-ep%d-lr%s-sts%d-bs%d-ga%.2f" % (
        lamda, beta, thetac, c, num_epochs, lr, step_size, batch_size, gamma))
logging.basicConfig(format='%(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                    filename=os.path.join('./result/', timeflag, 'log-' + name + '.log'), level=logging.INFO)


state = {"trainset": trainset, "testset": testset, "drop": drop, "num_epochs": num_epochs,"lr": lr, "step_size": step_size, "batch_size": batch_size,
         "adambeta": adambeta,  "icpthre": icpthre, "angthre": angthre, "angthrevalue": angthrevalue, "startepoch": startepoch, "lossalldata": lossalldata, "glothre": glothre, 
        "reweight":reweight, "modelepoch":modelepoch, "useflowvoxel":useflowvoxel,"seqstepsize":seqstepsize, "showtestloss":showtestloss, "fastpwcknn":fastpwcknn}

cfg.update(state)
cfg = update_cfg(cfg=cfg)

if useflowvoxel:                   
    cfg['newnet'] = cfgvoxel
if not testmode:       
    torch.save(cfg, os.path.join('./modelpkl/', timeflag, 'state.pth'))

if useflowvoxel:
    model = get_flow_model(cfg=cfgvoxel).cuda()     

with open(os.path.join('./result/', timeflag, 'para-msg.txt'), "w+") as f:

    for msg in cfg:
        f.write(str(msg) + "--" + str(cfg[msg]) + "\n")

with open(os.path.join('./param/common.yaml')) as f:
    cfg_default = yaml.safe_load(f)
    if not useflowvoxel:
        cfg_default.pop('newnet')
    cfg_differ = key_compare(cfg_default, cfg.copy())
    cfg["model-msg"] = get_size(model)                     #FLOPs
    print(cfg["model-msg"])
    cfg_differ["model-msg"] = cfg["model-msg"]
    # cfg_differ = sorted(cfg_differ.items(),key=lambda x:x[0],reverse=False)
    with open(os.path.join('./result/', timeflag, 'para-differ.yaml'), "w+") as f:
        yaml.safe_dump(cfg_differ, f, encoding='utf-8', allow_unicode=True, sort_keys=True)

with open(os.path.join('./result/', timeflag, 'para-msg.yaml'), "w+") as f:
    yaml.safe_dump(cfg, f, encoding='utf-8', allow_unicode=True, sort_keys=True)

if useflowvoxel:              #true
    bestyaml = './param/nowbest_flow.yaml'

with open(os.path.join(bestyaml)) as f:
    cfg_best = yaml.safe_load(f)
    if not useflowvoxel:
        cfg_best.pop('newnet')
    cfg_differ_best = key_compare(cfg_best, cfg.copy())


with open(os.path.join('./result/', timeflag, 'para-bestdiffer.yaml'), "w+") as f:
    yaml.safe_dump(cfg_differ_best, f, encoding='utf-8', allow_unicode=True, sort_keys=True)

########3
if lossalldata is not None or useflowvoxel:       #lossalldata=10240      true
    if useflowvoxel:
        lossfun = NPointLoss                  #N个点的loss

    criterion = lossfun(beta=beta, ttype=ttype, iseuler=iseuler, lowlimit=lowlimit, prweight=prweight, fastknn=fastknn).cuda().type(ttype)


if loadmodel:
    model.load_state_dict(torch.load(os.path.join("./modelpkl/", modelname, str(startepoch) + "-model.pth")))


params = [{'params': model.parameters()}, {'params': [criterion.sx, criterion.sq]}]


if opttype == "adam":
    optimizer = optim.Adam(params, lr=lr, betas=adambeta, weight_decay=1e-5, amsgrad=amsgrad)
else:
    optimizer = optim.SGD(params, lr=lr, momentum=sgdmomentum, weight_decay=1e-5)


if loadmodel:             
    optimizer.load_state_dict(torch.load(os.path.join("./modelpkl/", modelname, str(startepoch) + "-optimizer.pth")))
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=modelepoch)

else:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

torch.cuda.empty_cache()


if continnetlen > 1 and continnetequal:
    loaderlen = continlen + continnetlen - 1
else:
    loaderlen = continlen

#load dataset

parse = Parser(root=dataroot, iseuler=iseuler, train_sequences=trainset, test_sequences=testset, valid_sequences=vaildset,
               batch_size=batch_size, workers=workers,  ntype=ntype,               
               shuffle_train=shuffle_train,
               usesavedata=usesavedata,                
               continlen=loaderlen,               
               lossalldata=lossalldata, 
               testcontin2=testcontin2, testcompletion=testcompletion, traincompletion=traincompletion, testbatch=testbatch,
               newnetname=cfgvoxel['modeltype'],
               cfgvoxel=cfgvoxel,showtestloss=showtestloss)

if loadmodel and not onlyneedtest:      

    load_parse(modelepoch, parse.trainloader)          


timelog = []
timestart = []
if onlyneedtest:          #test
    epoch = modelepoch

    if usevaild:

        if epoch - startepoch == 2 and NPYpreload == "online":
            parse.validloader.dataset.npyreadyflag = True
        with torch.no_grad():
            proceed_one_epoch(epoch, timeflag, parse.validloader, model, optimizer, criterion, needloss=False, vaildmode=True)


    teststart = time.time()
    if len(testset) > 0:
        if epoch - startepoch == 2 and NPYpreload == "online":
            parse.testloader.dataset.npyreadyflag = True
        with torch.no_grad():
            proceed_one_epoch(epoch, timeflag, parse.testloader, model, optimizer, criterion, needloss=False)
    testover = time.time()

    if epoch % 10 == 0:
        with open(os.path.join('./result/', timeflag, str(epoch) + 'epochflag.txt'), "w") as f:
            for i in range(len(timelog)):
                f.write("test:%f %s\n" % (timelog[i], timestart[i]))
            timelog = []
            timestart = []
    nowtime = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H-%M-%S")
    log("test:%f nowtime:%s" % ((testover - teststart), nowtime), pt=True)
    timelog.append((testover - teststart))
    timestart.append(nowtime)

else:        #train

    for epoch in range(modelepoch + 1, num_epochs + 1):      
        torch.cuda.empty_cache()
        if epoch - startepoch == 2 and NPYpreload == "online":    
            parse.trainloader.dataset.npyreadyflag = True

        if epochthre != 0:    
            criterion.nowepoch = epoch

        start = time.time()
        scheduler.step()


        if usevaild:
            if epoch - startepoch == 2 and NPYpreload == "online":
                parse.validloader.dataset.npyreadyflag = True
            with torch.no_grad():
                proceed_one_epoch(epoch - 1, timeflag, parse.validloader, model, optimizer, criterion, needloss=False, vaildmode=True)

        if len(trainset) > 0:
            proceed_one_epoch(epoch, timeflag, parse.trainloader, model, optimizer, criterion, needloss=True)
            trainover = time.time()
            if torch.cuda.device_count() > 1 and useparrell:
                modelsave = model.module
            else:
                modelsave = model

            if trainset[0] >= 40:
                savestep = 1
            elif useflowvoxel:
                savestep = 1
            else:
                savestep = 20

            if not testmode:

                torch.save(modelsave.state_dict(), os.path.join('./modelpkl/', timeflag, str(epoch) + '-model.pth'))
                torch.save(scheduler.state_dict(), os.path.join('./modelpkl/', timeflag, str(epoch) + '-scheduler.pth'))
                torch.save(optimizer.state_dict(), os.path.join('./modelpkl/', timeflag, str(epoch) + '-optimizer.pth'))


        teststart = time.time()
        if len(testset) > 0:
            if epoch - startepoch == 2 and NPYpreload == "online":
                parse.testloader.dataset.npyreadyflag = True
            with torch.no_grad():
                proceed_one_epoch(epoch, timeflag, parse.testloader, model, optimizer, criterion, needloss=False)
        testover = time.time()


        if epoch % 10 == 0:
            with open(os.path.join('./result/', timeflag, str(epoch) + 'epochflag.txt'), "w") as f:
                for i in range(len(timelog)):
                    f.write("all:%f train:%f test:%f %s\n" % (timelog[i][0], timelog[i][1], timelog[i][2], timestart[i]))
                timelog = []
                timestart = []
        nowtime = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H-%M-%S")
        log("alltime:%f train:%f test:%f nowtime:%s" % ((time.time() - start), (trainover - start), (testover - teststart), nowtime), pt=True)
        timelog.append([(time.time() - start), (trainover - start), (testover - teststart)])
        timestart.append(nowtime)
