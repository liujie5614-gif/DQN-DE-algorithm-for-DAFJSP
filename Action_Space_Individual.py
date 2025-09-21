import random
import numpy as np
import copy
import DE_utils


def Select_DE_crossover(Products, gen, record, a):
    newgen = []
    if a == 0:
        newgen = rule1(gen, record)
    elif a == 1:
        newgen = rule2(gen, record)
    elif a == 2:
        newgen = rule3(gen, record)
    elif a == 3:
        newgen = rule4(gen, record)
    elif a == 4:
        newgen = rule5(gen, record, Products)
    elif a == 5:
        newgen = rule6(gen, record, Products)
    elif a == 6:
        newgen = rule7(gen, record)
    return newgen


# 针对个体的变异动作,将工厂最后加工的那个零件进行交换
def rule1(gen, record):
    gen, record = copy.deepcopy(gen), copy.deepcopy(record)
    distribute, Pcode = gen[0], gen[1]
    Onum = int(len(Pcode[0]) / len(distribute[0]))
    jnum = sum([len(distribute[i]) for i in range(len(distribute))])
    mnum = len(record[0]) - 1
    Fnum = len(record)
    jobft = [[i, 0] for i in range(jnum)]
    for i in range(Fnum):
        for j in range(mnum):
            for k in range(len(record[i][j][2])):
                if record[i][j][1][k] > jobft[record[i][j][2][k]][1]:
                    jobft[record[i][j][2][k]][1] = record[i][j][1][k]
    jobft.sort(key=lambda x: x[1])
    jobft_f = [[] for i in range(Fnum)]
    for i in range(jnum):
        for j in range(Fnum):
            if jobft[i][0] in distribute[j]:
                jobft_f[j].append(jobft[i])
    exchange = [jobft_f[i][-1][0]for i in range(Fnum)]
    for i in range(Fnum):
        distribute[i].remove(exchange[i])
        if i == Fnum - 1:
            distribute[i].append(exchange[0])
        else:
            distribute[i].append(exchange[i + 1])
    pcode = DE_utils.cros_addh(Pcode)
    pcode.sort(key=lambda x: x[1], reverse=True)
    newcode = [[] for i in range(Fnum)]
    for j in range(len(pcode)):
        for k in range(len(distribute)):
            if pcode[j][0] in distribute[k]:
                newcode[k].append(pcode[j][0])
    newgen = [distribute, newcode]
    return newgen


# 针对个体的变异动作,将工厂最后加工的零件和最先加工的零件进行交换
def rule2(gen, record):
    gen, record = copy.deepcopy(gen), copy.deepcopy(record)
    distribute, Pcode = gen[0], gen[1]
    Onum = int(len(Pcode[0]) / len(distribute[0]))
    jnum = sum([len(distribute[i]) for i in range(len(distribute))])
    mnum = len(record[0]) - 1
    Fnum = len(record)
    jobft = [[i, 0] for i in range(jnum)]
    for i in range(Fnum):
        for j in range(mnum):
            for k in range(len(record[i][j][2])):
                if record[i][j][1][k] > jobft[record[i][j][2][k]][1]:
                    jobft[record[i][j][2][k]][1] = record[i][j][1][k]
    jobft.sort(key=lambda x: x[1])
    jobft_f = [[] for i in range(Fnum)]
    for i in range(jnum):
        for j in range(Fnum):
            if jobft[i][0] in distribute[j]:
                jobft_f[j].append(jobft[i])
    exchange_first = [jobft_f[i][0][0] for i in range(Fnum)]
    exchange_last = [jobft_f[i][-1][0]for i in range(Fnum)]
    pcode = DE_utils.cros_addh(Pcode)
    h, code = [i[1] for i in pcode], [i[0] for i in pcode]
    for i in range(Fnum):
        h[exchange_first[i]*Onum:(exchange_first[i]+1)*Onum], h[exchange_last[i]*Onum:(exchange_last[i]+1)*Onum] \
            = h[exchange_last[i]*Onum:(exchange_last[i]+1)*Onum], h[exchange_first[i]*Onum:(exchange_first[i]+1)*Onum]
    res = [[code[i], h[i]] for i in range(len(h))]
    res.sort(key=lambda x: x[1], reverse=True)
    newcode = [[] for i in range(Fnum)]
    for j in range(len(res)):
        for k in range(len(distribute)):
            if res[j][0] in distribute[k]:
                newcode[k].append(res[j][0])
    newgen = [distribute, newcode]
    return newgen


# 针对个体的变异动作,将工厂最先加工完的那个零件进行交换
def rule3(gen, record):
    gen, record = copy.deepcopy(gen), copy.deepcopy(record)
    distribute, Pcode = gen[0], gen[1]
    Onum = int(len(Pcode[0]) / len(distribute[0]))
    jnum = sum([len(distribute[i])for i in range(len(distribute))])
    Fnum = len(record)
    mnum = len(record[0])-1
    jobft = [[i, 0]for i in range(jnum)]
    for i in range(Fnum):
        for j in range(mnum):
            for k in range(len(record[i][j][2])):
                if record[i][j][1][k] > jobft[record[i][j][2][k]][1]:
                    jobft[record[i][j][2][k]][1] = record[i][j][1][k]
    jobft.sort(key=lambda x: x[1])
    jobft_f = [[]for i in range(Fnum)]
    for i in range(jnum):
        for j in range(Fnum):
            if jobft[i][0] in distribute[j]:
                jobft_f[j].append(jobft[i])
    exchange = [jobft_f[i][0][0]for i in range(Fnum)]
    for i in range(Fnum):
        distribute[i].remove(exchange[i])
        if i == Fnum - 1:
            distribute[i].append(exchange[0])
        else:
            distribute[i].append(exchange[i + 1])
    pcode = DE_utils.cros_addh(Pcode)
    pcode.sort(key=lambda x: x[1], reverse=True)
    newcode = [[] for i in range(Fnum)]
    for j in range(len(pcode)):
        for k in range(len(distribute)):
            if pcode[j][0] in distribute[k]:
                newcode[k].append(pcode[j][0])
    newgen = [distribute, newcode]
    return newgen


# 针对个体的变异动作,将最晚加工完零件的工厂中最后加工的那个零件转移到最先加工完零件的工厂
def rule4(gen, record):
    gen, record = copy.deepcopy(gen), copy.deepcopy(record)
    distribute, Pcode = gen[0], gen[1]
    Onum = int(len(Pcode[0]) / len(distribute[0]))
    jnum = sum([len(distribute[i])for i in range(len(distribute))])
    Fnum = len(record)
    mnum = len(record[0])-1
    jobft = [[i, 0]for i in range(jnum)]
    for i in range(Fnum):
        for j in range(mnum):
            for k in range(len(record[i][j][2])):
                if record[i][j][1][k] > jobft[record[i][j][2][k]][1]:
                    jobft[record[i][j][2][k]][1] = record[i][j][1][k]
    jobft.sort(key=lambda x: x[1])
    jobft_f = [[]for i in range(Fnum)]
    for i in range(jnum):
        for j in range(Fnum):
            if jobft[i][0] in distribute[j]:
                jobft_f[j].append(jobft[i])
    jobft_f = [[i, jobft_f[i][-1][0], jobft_f[i][-1][1]]for i in range(Fnum)]
    jobft_f.sort(key=lambda x: x[2])
    job_index = [jobft_f[-1][1], jobft_f[-1][0]]
    F_index = jobft_f[0][0]
    distribute[F_index].append(job_index[0])
    distribute[job_index[1]].remove(job_index[0])
    pcode = DE_utils.cros_addh(Pcode)
    pcode.sort(key=lambda x: x[1], reverse=True)
    newcode = [[] for i in range(Fnum)]
    for j in range(len(pcode)):
        for k in range(len(distribute)):
            if pcode[j][0] in distribute[k]:
                newcode[k].append(pcode[j][0])
    newgen = [distribute, newcode]
    return newgen


# 将工厂中最后装配的那个产品中最后加工完的零件的最后一道工序提前
def rule5(gen, record, Products):
    gen, record, Products = copy.deepcopy(gen), copy.deepcopy(record), copy.deepcopy(Products)
    distribute, Pcode = gen[0], gen[1]
    newgen = copy.deepcopy(gen)
    Onum = int(len(Pcode[0]) / len(distribute[0]))
    jnum = sum([len(distribute[i]) for i in range(len(distribute))])
    Fnum = len(record)
    mnum = len(record[0]) - 1
    jobft = [[i, 0] for i in range(jnum)]
    for i in range(Fnum):
        for j in range(mnum):
            for k in range(len(record[i][j][2])):
                if record[i][j][1][k] > jobft[record[i][j][2][k]][1]:
                    jobft[record[i][j][2][k]][1] = record[i][j][1][k]
    firstP = [record[i][-1][2][-1] if len(record[i][-1][2]) > 0 else random.randint(0, len(Products)-1)for i in range(Fnum)]
    firstP = list(set(firstP))
    while len(firstP) < Fnum:
        tempt = random.randint(0, len(Products)-1)
        if tempt not in firstP:
            firstP.append(tempt)
    firstP_jobs = [Products[firstP[i]]for i in range(Fnum)]
    firstP_jobs_late = [[]for i in range(Fnum)]
    for i in range(Fnum):
        for j in range(len(firstP_jobs[i])):
            firstP_jobs_late[i].append([firstP_jobs[i][j], jobft[firstP_jobs[i][j]][1]])
    for i in range(Fnum):
        firstP_jobs_late[i].sort(key=lambda x: x[1], reverse=True)
    moveup_jobs = [firstP_jobs_late[i][0][0]for i in range(len(firstP))]
    moveup_jobs = [[moveup_jobs[i], 0]for i in range(len(firstP))]
    for i in range(len(firstP)):
        for j in range(Fnum):
            if moveup_jobs[i][0] in distribute[j]:
                moveup_jobs[i][1] = j
    indices = [[]for i in range(len(firstP))]
    for i in range(Fnum):
        indices[i] = [index for (index, item) in enumerate(gen[1][moveup_jobs[i][1]]) if item == moveup_jobs[i][0]]
    up_indices = [random.randint(0, indices[i][-1])for i in range(len(indices))]
    for i in range(Fnum):
        newgen[1][moveup_jobs[i][1]][indices[i][-1]], newgen[1][moveup_jobs[i][1]][up_indices[i]] = newgen[1][moveup_jobs[i][1]][up_indices[i]], newgen[1][moveup_jobs[i][1]][indices[i][-1]]
    newgen = gen
    return newgen


# 将工厂中随机一个产品中最后加工完的零件的最后一道工序提前
def rule6(gen, record, Products):
    gen, record, Products = copy.deepcopy(gen), copy.deepcopy(record), copy.deepcopy(Products)
    distribute, Pcode = gen[0], gen[1]
    newgen = copy.deepcopy(gen)
    Onum = int(len(Pcode[0]) / len(distribute[0]))
    jnum = sum([len(distribute[i]) for i in range(len(distribute))])
    Fnum = len(record)
    mnum = len(record[0]) - 1
    jobft = [[i, 0] for i in range(jnum)]
    for i in range(Fnum):
        for j in range(mnum):
            for k in range(len(record[i][j][2])):
                if record[i][j][1][k] > jobft[record[i][j][2][k]][1]:
                    jobft[record[i][j][2][k]][1] = record[i][j][1][k]
    pro = [i for i in range(len(Products))]
    random.shuffle(pro)
    firstP = pro[:Fnum]
    firstP_jobs = [Products[firstP[i]]for i in range(Fnum)]
    firstP_jobs_late = [[]for i in range(Fnum)]
    for i in range(Fnum):
        for j in range(len(firstP_jobs[i])):
            firstP_jobs_late[i].append([firstP_jobs[i][j], jobft[firstP_jobs[i][j]][1]])
    for i in range(Fnum):
        firstP_jobs_late[i].sort(key=lambda x: x[1], reverse=True)
    moveup_jobs = [firstP_jobs_late[i][0][0]for i in range(Fnum)]
    moveup_jobs = [[moveup_jobs[i], 0]for i in range(Fnum)]
    for i in range(Fnum):
        for j in range(Fnum):
            if moveup_jobs[i][0] in Pcode[j]:
                moveup_jobs[i][1] = j
    indices = [[] for i in range(Fnum)]
    for i in range(Fnum):
        indices[i] = [index for (index, item) in enumerate(gen[1][moveup_jobs[i][1]]) if item == moveup_jobs[i][0]]
    up_indices = [random.randint(0, indices[i][-1]) for i in range(len(indices))]
    for i in range(Fnum):
        newgen[1][moveup_jobs[i][1]][indices[i][-1]], newgen[1][moveup_jobs[i][1]][up_indices[i]] = \
        newgen[1][moveup_jobs[i][1]][up_indices[i]], newgen[1][moveup_jobs[i][1]][indices[i][-1]]
    newgen = gen
    return newgen


# 针对个体的变异动作,将工厂随机零件进行交换
def rule7(gen, record):
    gen, record = copy.deepcopy(gen), copy.deepcopy(record)
    distribute, Pcode = gen[0], gen[1]
    Onum = int(len(Pcode[0]) / len(distribute[0]))
    jnum = sum([len(distribute[i])for i in range(len(distribute))])
    Fnum = len(record)
    mnum = len(record[0])-1
    jobft = [[i, 0]for i in range(jnum)]
    for i in range(Fnum):
        for j in range(mnum):
            for k in range(len(record[i][j][2])):
                if record[i][j][1][k] > jobft[record[i][j][2][k]][1]:
                    jobft[record[i][j][2][k]][1] = record[i][j][1][k]
    jobft.sort(key=lambda x: x[1])
    jobft_f = [[]for i in range(Fnum)]
    for i in range(jnum):
        for j in range(Fnum):
            if jobft[i][0] in distribute[j]:
                jobft_f[j].append(jobft[i])
    exchange = [jobft_f[i][0][0]for i in range(Fnum)]
    for i in range(Fnum):
        distribute[i].remove(exchange[i])
        if i == Fnum - 1:
            distribute[i].append(exchange[0])
        else:
            distribute[i].append(exchange[i + 1])
    pcode = DE_utils.cros_addh(Pcode)
    pcode.sort(key=lambda x: x[1], reverse=True)
    newcode = [[] for i in range(Fnum)]
    for j in range(len(pcode)):
        for k in range(len(distribute)):
            if pcode[j][0] in distribute[k]:
                newcode[k].append(pcode[j][0])
    newgen = [distribute, newcode]
    return newgen

