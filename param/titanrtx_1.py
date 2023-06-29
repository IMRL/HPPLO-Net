import yaml
import os

dataroot = "/data/semantickitti/dataset"

if os.path.exists("./param/config.yaml"):
    with open("./param/config.yaml") as f:
        cfg = yaml.safe_load(f, )

if os.path.exists("./param/configvoxel.yaml"):
    with open("./param/configvoxel.yaml") as f:
        cfgvoxel = yaml.safe_load(f, )
        
        
#############经常修改############################

cfgvoxel["svdnettype"] = "weight_po2pl"

# onlyneedtest = True
onlyneedtest = False
loadmodel = False  # 是否加载已保存模型  继续训练或测试
modelname = "(2022-11-24 23-00-25)-(TITAN RTXx1)-19577-flow-full-hard"
startepoch = 40   #loadmode=true   loadmodel epoch

showtestloss = True # 显示测试集的loss
batch_size = 10  # titan*2 190

testmode = False  # 测试模式（小batch+小数据集）
testbatch = 1
testset = [9, 10]
trainset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

lr = 1e-3  

#####################不常修改###################

lossalldata = "10240"
useflowvoxel = True

iseuler = True

fastknn = True
fastpwcknn = True

Distributed = False  # 多卡模式，暂时不会用（多机多卡，单机多卡）
# torch.set_printoptions(profile="full")

# 论文除了useuncer都为False（lowlmit理论也为false相当于加强的fovloss）
useorgsetting = False  # 是否使用论文原始参数，已废弃
# iseuler = False  # 欧拉角还是四元数
comquat = None  # 如果使用四元数，是否采用1+3拼接形式 grad相当于以前的true，

onlygt = False  # 只有gt
lonetgt = False  # lonet的gt形式
usegt = False  # 加入gt
lowlimit = True  # 弱化位移和旋转限制（范围大小）

useuncer = False
bypass = False  # lonet和Pointseg使用
lonetuncer = False  # lonet的uncertainmap， 因为梯度无法传播暂时无法实验
muluncerloss = False
# pointnetnor = False
doubleloss = False  # src-->tar + tar-->src
x55norm = False  # 5*5 or 上下左右 normalmap
l2icp = False  # 计算icp中，三个方向的移动是否加上平方
l2icptheta = 10000.0  # l2的权重最大值
# rotainput = False  # 计算loss输入的为rota

vhacc = False  # 64*720 or 52*720
loamheight = False  # loam保留0-50线，这里方便保留0-51，高度为52
lonetwidth = False  # lonet标准宽度为1800

avgpool = False  # 同时使用全局均值池化和全连接
fullgap = False  # 完全使用全局均值池化代替全连接（除最后输出位姿的全连接）
useLSTM = False  # 是否在GAp和Fc之间加入LSTM
useselfattention = False  # 是否使用self-attention机制
useIMU = False  # 加入IMU的加速度和加速旋转信息，使用后覆盖useLSTM    # 是否在GAp和Fc之间加入LSTM
# IMUtype = None  # unsync1 按照timestamp为基准，序列8可能有偏移  sync+imulen=1 相当于直接使用高精度IMU数据
IMUtype = "unsync2all-af-vf"
IMUnorm = False
IMUnormtype = "org"
IMUlen = 15


depthnorm = False  # 将深度归一化到[0,255]间
predepthnorm = False  # 将深度归一化到[0,255]间，预处理完成
usedepth = False  # 是否使用深度信息作为网络输入
# useGyroNet = False
rnntype = "lstm"
usesemanticinput = False  # 是否使用语义信息（同时用于网络输入和loss计算，还未细分）
usesemanticloss = False  # 是否使用语义信息（同时用于网络输入和loss计算，还未细分）
anglelosstype = None  # angleloss计算角度差还是向量差
useremiss = False
normalinput = False  # 单独使用normal作为输入

continlen = 2  # 1一次使用连续i帧数据，并通过loss消除累计误差2deeplio seq的长度
continlosslen = 1 # deeplio专用，计算k帧到k+1..k+i的误差，通过拟合位姿的累乘
continnetlen = 1 # deeplio专用, 通过网络直接拟合计算多帧误差

circlelen = 1  # 计算两帧点云时，多次输入网络，计算位姿，用位姿变换新的顶点图再输入网络中
circlebase = 50  # 迭代i次稳定以后再用多次迭代的变换
##########  网络种类选取

uselonet = False
usedivpointseg = False  # 位移和旋转分别输出，权值不共享（可以加入uncertainmap）
usepointnet = False
usedeeppco = False
usesharesiamsenet = False  # 两帧数据分别输入网络，权值共享
useunsharesiamsenet = False  # 两帧数据分别输入网络，权值不共享
usesvdnet = False  # svd分解作为网络输出
usesyqnet = False
num_features = 512
head = "mlp"

##########  这些不太会变
usebnmaxpool = True  # 保留网络的maxpool和batchnorm，一般保留
shuffle_train = True  # 打乱训练集，必做
binpreproess = False  # 将预处理内容加入内存，内存不够已经废弃
flaot64 = False  # 精度
usesavedata = True  # 读取已经算好的预处理数据而不多次计算，一般必用加快训练速度
#################  加载老模型需要考虑的
# loadmodel = False  # 是否加载已保存模型
# modelname = None
changelr = False  # 改变保存模型的学习率
# startepoch = 0
##############
# minepoch = 10
c = 6  # 论文为6
# batch_size = 1

lamda = 1e-2  # 论文未给出大小, fovloss比重
beta = 100  # prloss比重(用gt就是gtloss，不用就是强化的fovloss，限制旋转位移范围)
prweight = 1.0  # gtloss比重
thetac = 0.4  # 论文为0.4
anglelossrate = 1.0

opttype = "adam"  # 优化器选择
sgdmomentum = 0.9
adambeta = (0.9, 0.99)
weight_decay = 1e-5
amsgrad = False # 使用adam的情况下是否使用amsgrad（adam+sgd）

epochstep = False   # 执行完一次epoch才修改网络权重
batchstep = 1  # 多少个batchsuze执行一次梯度


uncerlow = 0.01  # uncermap的阈值，不在阈值内的点会被统计
uncerhigh = 0.5
uncerlograte = 1.0  # uncertainmap log 调节loss的比例（形偏向lonet）（lonet为 1/3  uncertainmap为 1)
drop = None  # 1e-2, 1e-1  # 论文无该参数，1e-4只舍去icploss最大的2-3个点，相当于没有
step_size = 20  # 论文为20
dynastep = False # 根据序列长度动态调整step_size
num_epochs = 1200
# lr = 1.1602906250000002e-05
dynalr = False
gamma = 0.5  # 论文为0.5
seqlen = 2048

lastdrop = 0.5  # pointseg最后输出网络dropout的p值
icpthre = [0.0, 10000.0]  # 纯粹angleloss和icploss的最大阈值
angthre = [0.0, 90.0]      #  如果使用angle模式，则为角度制


usedeeplio = False
deepliolidarmodel = "pointseg"
useloampoints = False
usehiddenstate = False  # 将以前的隐藏层输出用于一次迭代的输入
loaderstepsize = 1  # 使用多帧数据时，间隔取数据
divfeat = False  # 旋转和位移的特征是否分开计算
needlongterm = False # 0~i计算Loss
equalweight = False # icp/ang loss大小权重控制为均等，和数量无关
randseed = False # 固定随机数种子
epochthre = 0 # 大于一定epoch再改变loss策略 对drop angthre icpthre l2icp有效
stri_epochthre = 0 # 大于一定epoch再使用needlongterm,continnetlen,continlosslen
continnetequal = False # 使用continnet的时候，跳帧是否和连续帧网络等长，即0-1,1-2,2-3对应0-2,1-3还是0-2,1-3,2-4
imufusetype = None  # 使用continnet的时候, imu合并方式，"cat"或"avg
IMUpreload = False  # imu数据预加载
datapreprocess = False

# lossalldata = None  # 所有可用的点都考虑loss
globallossrate = 1.0    # 该loss表示一对匹配点和原点形成夹角的大小
glothre = [0.0, 90.0]
globallosstype = False
NPYpreload = None   # unsup数据预加载

parallelloss = False   # 对loss使用DataParallel
reweight = [1] # 使用conlennet的时候，每个值按照权重最为最后结果

seednumnp = 0
seednumcpu = 0
seednumgpu = 0
open3dnor = False
fordnewheight = False
modelepoch = -1 # 采用之前的模型但是epoch和保存的模型不同

testcontin2 = False  # 测试集上序列长度为2
# testbatch = -1  # 测试集上batchsize为1
testcompletion = False # 测试集补全, [0,1]变[0,1][0,1][0,1]  [0,1][1,2]变[0,1][0,1][1,2]
traincompletion = False
# onlyneedtest = False # 只需要进行测试
usevaild = False # 让训练集通过和测试集相同的环境, 使用hide的时候才会起作用
newtestfun = False # 新的测试方案, 使用hide的时候才会起作用, usevaild为true是，必为true

useapex = False # 降低复杂度
seqstepsize = [1]  # 跳跃取数据
mapnoground = None
onlylastloss = False # 长序列只计算最后一个位姿

###################
# voxeldataname = None
# useflowvoxel = False
selfvoxelloss = False  # 使用光流估计的点来计算loss
icpprlossname = None
icpprrota = False
svdresidual = None
epochsave = [-1, -1]


