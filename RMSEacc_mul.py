import datetime
import os, sys
lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
lib_path2 = '/test/flow_motion/unsupodo_refine2'
sys.path.append(lib_path2)
import multiprocessing
import os
import numpy as np
from tools.evatools.local2word import localsepara, get_world, gtget_world, localsepara_fast, relative_error
from tools.evatools.evaluation import evaluation, gtlocalevaluation
import pandas as pd


root = "/test/flow_motion/simple_flowuncer_seg/"

flagtime = datetime.datetime.strptime("2021-11-23 00-00-00", '%Y-%m-%d %H-%M-%S')
datasize = np.array([4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201,
                     921, 1061, 3281, 631, 1091, 1731, 491, 1081, 4981, 831, 2721,
                     -1, -1, -1, -1, -1, -1, -1, -1,
                     3817, 6103, -1, 8278, 11424, 9248, 5351, 4705, 5233, -1,
                     6443, 20801, 4701, 11846, 16596, 7538, 14014, 5677, 10903, 9765, 9868, 41529], dtype=np.int)


def mul_process(timedir):


    print(timedir)
    filepath = os.path.join(root + "data/summary", timedir + ".xlsx")
    localpath = os.path.join(root + "data/summarylocal", timedir + ".xlsx")
    reader = None

    print(filepath)
    if os.path.exists(filepath):
        writer = pd.ExcelWriter(filepath, mode="a", engine='openpyxl')
    else:
        data_df = pd.DataFrame()
        writer = pd.ExcelWriter(filepath, mode="w")
        data_df.to_excel(writer)
        writer.save()


    if os.path.exists(localpath):
        writer_re = pd.ExcelWriter(localpath, mode="a", engine='openpyxl')
    else:
        data_df = pd.DataFrame()
        writer_re = pd.ExcelWriter(localpath, mode="w")
        data_df.to_excel(writer_re)
        writer_re.save()

    if "apollo"in timedir:
        start = 1
    elif "flow" in timedir:
        start = 1
    else:
        start = 1

    for e in range(0, 600, 1):
    # for e in [42, 47]:
        result = None
        if os.path.exists(filepath):
            if reader is None:
                reader = pd.read_excel(filepath, sheet_name=None)
            if str(e) in reader.keys():
                continue

        choose = ["test",""]

        relaresult = None
        for dir in choose:
            eresult = None
            dir = str(e) + dir
            SQUE = localsepara_fast(root, timedir, dir)
            if SQUE is None:
                continue
            print(dir)
            totallen = 0.0
            for sque in SQUE:
                get_world(root, timedir, dir, int(sque))
                trans, rota = evaluation(root, timedir, dir, int(sque))
                relative = relative_error(root, timedir, dir, int(sque))
                rota *= 100
                if eresult is None:
                    eresult = np.array([sque, trans, rota, trans*datasize[sque], rota*datasize[sque]], dtype=np.float64).reshape(1, -1)
                else:
                    temp = np.array([sque, trans, rota, trans * datasize[sque], rota * datasize[sque]], dtype=np.float64).reshape(1, -1)
                    eresult = np.concatenate((eresult, temp), axis=0)
                totallen += datasize[sque]
                if relaresult is None:
                    relaresult = relative
                else:
                    relaresult = np.concatenate((relaresult, relative), axis=1)
            averag = np.sum(eresult, axis=0).reshape(1, -1) / totallen
            averag[:, 1] = averag[:, 3]
            averag[:, 2] = averag[:, 4]
            trueavg = np.mean(eresult, axis=0)
            print(dir,"avgtran:",trueavg[1],"avgrota:",trueavg[2])
            eresult = np.concatenate((eresult, averag), axis=0)

            if result is None:
                result = eresult
            else:
                result = np.concatenate((result, eresult), axis=0)
        if SQUE is None:
            continue
        data = pd.DataFrame(result)
        data_rae = pd.DataFrame(relaresult, index=["tmse", "trmse", "tmae", "tlen", "trate",
                                                 "rmse", "rrmse", "rmae", "rlen", "rrate"], columns=["x", "y", "z", "tlen"] * (relaresult.shape[1] // 4))

        if os.path.exists(filepath):
            writer = pd.ExcelWriter(filepath, mode="a", engine='openpyxl')
        else:
            writer = pd.ExcelWriter(filepath, mode="w")
        data.to_excel(writer, str(e), float_format='%.5f', index=False, header=False)
        writer.save()


        if os.path.exists(localpath):
            writer_re = pd.ExcelWriter(localpath, mode="a", engine='openpyxl')
        else:
            # writer_re = pd.ExcelWriter(localpath, mode="w")
            data_df = pd.DataFrame()
            writer_re = pd.ExcelWriter(localpath, mode="w")
            data_df.to_excel(writer)
            writer_re.save()
        data_rae.to_excel(writer_re, str(e), float_format='%.5f', index=True, header=True)
        writer_re.save()
    writer.close()
    writer_re.close()


if __name__ == '__main__':

    dirpath = os.path.join(root, "result")
    # dirpath = root
    timelist = os.listdir(dirpath)
    timelist = [
        # "/experiment/kitti_ford_apollo/aloam/data"
        "(2023-04-06 17-33-45)-(TITAN RTXx1)-19535-flow-full"
    
    ]
    for timeflag in timelist:
        # mul_process(timeflag)
        p = multiprocessing.Process(target=mul_process, args=(timeflag,))  # 创建一个进程，args传参 必须是元组
        p.start()
