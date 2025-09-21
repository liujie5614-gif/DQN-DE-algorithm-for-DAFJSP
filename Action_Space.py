import random
import numpy as np
import copy
import DE_utils


def Select_DE_crossover(env, pop1, pop2, CR, a):
    newpop1, newpop2 = [], []
    if a == 0:
        newpop1, newpop2 = rule1(env, pop1, pop2, CR)
    elif a == 1:
        newpop1, newpop2 = rule2(pop1, pop2, CR)
    elif a == 2:
        newpop1, newpop2 = rule3(pop1, pop2, CR)
    elif a == 3:
        newpop1, newpop2 = rule4(env, pop1, pop2, CR)
    elif a == 4:
        newpop1, newpop2 = rule5(pop1, pop2, CR)
    return newpop1, newpop2


# 个体内产品分配交换
def rule1(env, pop1, pop2, CR):
    newpop1, newpop2 = copy.deepcopy(pop1), copy.deepcopy(pop2)
    pop1, pop2 = copy.deepcopy(pop1), copy.deepcopy(pop2)
    for i in range(len(pop1)):
        if random.random() < CR:
            distribute1, distribute2 = copy.deepcopy(pop1[i][0]), copy.deepcopy(pop2[i][0])
            Pcode1, Pcode2 = copy.deepcopy(pop1[i][1]), copy.deepcopy(pop2[i][1])
            m_position1 = np.random.randint(0, min([len(distribute1[i])for i in range(len(distribute1))]), int(len(distribute1[0])/4)).tolist()
            m_position2 = np.random.randint(0, min([len(distribute2[i])for i in range(len(distribute2))]), int(len(distribute2[0])/4)).tolist()
            m_job1 = [[]for i in range(len(distribute1))]
            m_job2 = [[]for i in range(len(distribute2))]
            for j in range(len(distribute1)-1):
                for k in m_position1:
                    m_job1[j].append(pop1[i][0][j][k])
                    if j == len(distribute1)-2:
                        m_job1[j+1].append(pop1[i][0][j+1][k])
                    distribute1[j][k], distribute1[j+1][k] = distribute1[j+1][k], distribute1[j][k]
            for j in range(len(distribute2)-1):
                for k in m_position2:
                    m_job2[j].append(pop2[i][0][j][k])
                    if j == len(distribute2)-2:
                        m_job2[j+1].append(pop2[i][0][j+1][k])
                    distribute2[j][k], distribute2[j+1][k] = distribute2[j+1][k], distribute2[j][k]
            pcode1, pcode2 = DE_utils.cros_addh(Pcode1), DE_utils.cros_addh(Pcode2)
            h1, h2 = [i[1] for i in pcode1], [i[1] for i in pcode2]
            code1, code2 = [i[0] for i in pcode1], [i[0] for i in pcode2]
            pnum = int(len(Pcode1[0])/len(distribute1[0]))
            for j in range(len(m_job1)):
                for k in range(len(m_job1[j])):
                    if j == len(m_job1)-1:
                        code1[m_job1[j][k]*pnum:(m_job1[j][k]+1)*pnum] = [m_job1[0][k]]*pnum
                    else:
                        code1[m_job1[j][k]*pnum:(m_job1[j][k]+1)*pnum] = [m_job1[j+1][k]]*pnum
            for j in range(len(m_job2)):
                for k in range(len(m_job2[j])):
                    if j == len(m_job2)-1:
                        code2[m_job2[j][k]*pnum:(m_job2[j][k]+1)*pnum] = [m_job2[0][k]]*pnum
                    else:
                        code2[m_job2[j][k]*pnum:(m_job2[j][k]+1)*pnum] = [m_job2[j+1][k]]*pnum
            res1 = [[code1[i], h1[i]] for i in range(len(h1))]
            res2 = [[code2[i], h2[i]] for i in range(len(h2))]
            res1.sort(key=lambda x: x[1], reverse=True)
            res2.sort(key=lambda x: x[1], reverse=True)
            newcode1 = [[] for i in range(len(distribute1))]
            newcode2 = [[] for i in range(len(distribute2))]
            for j in range(len(res1)):
                for k in range(len(distribute1)):
                    if res1[j][0] in distribute1[k]:
                        newcode1[k].append(res1[j][0])
                    if res2[j][0] in distribute2[k]:
                        newcode2[k].append(res2[j][0])
            gen1, gen2 = [distribute1, newcode1], [distribute2, newcode2]
            newpop1[i], newpop2[i] = gen1, gen2
    return newpop1, newpop2


# 同种群个体间产品分配交换
def rule2(pop1, pop2, CR):
    newpop1, newpop2 = copy.deepcopy(pop1), copy.deepcopy(pop2)
    for i in range(len(pop1)):
        if random.random() < CR:
            distribute1, distribute2 = copy.deepcopy(newpop1[random.randint(0, len(pop1)-1)][0]), copy.deepcopy(newpop2[random.randint(0, len(pop1)-1)][0])
            newpop1[i][0], newpop2[i][0] = distribute1, distribute2
            Pcode1, Pcode2 = newpop1[i][1], newpop2[i][1]
            pcode1, pcode2 = DE_utils.cros_addh(Pcode1), DE_utils.cros_addh(Pcode2)
            pcode1.sort(key=lambda x: x[1], reverse=True)
            pcode2.sort(key=lambda x: x[1], reverse=True)
            newcode1 = [[] for i in range(len(distribute1))]
            newcode2 = [[] for i in range(len(distribute2))]
            for j in range(len(pcode1)):
                for k in range(len(distribute1)):
                    if pcode1[j][0] in distribute1[k]:
                        newcode1[k].append(pcode1[j][0])
                    if pcode2[j][0] in distribute2[k]:
                        newcode2[k].append(pcode2[j][0])
            newpop1[i], newpop2[i] = [distribute1, newcode1], [distribute2, newcode2]
    return newpop1, newpop2


# 不同种群个体间产品分配交换
def rule3(pop1, pop2, CR):
    newpop1, newpop2 = copy.deepcopy(pop1), copy.deepcopy(pop2)
    for i in range(len(pop1)):
        if random.random() < CR:
            distribute1, distribute2 = copy.deepcopy(newpop2[random.randint(0, len(pop1) - 1)][0]), copy.deepcopy(
                newpop1[random.randint(0, len(pop1) - 1)][0])
            newpop1[i][0], newpop2[i][0] = distribute1, distribute2
            Pcode1, Pcode2 = newpop1[i][1], newpop2[i][1]
            pcode1, pcode2 = DE_utils.cros_addh(Pcode1), DE_utils.cros_addh(Pcode2)
            pcode1.sort(key=lambda x: x[1], reverse=True)
            pcode2.sort(key=lambda x: x[1], reverse=True)
            newcode1 = [[] for i in range(len(distribute1))]
            newcode2 = [[] for i in range(len(distribute2))]
            for j in range(len(pcode1)):
                for k in range(len(distribute1)):
                    if pcode1[j][0] in distribute1[k]:
                        newcode1[k].append(pcode1[j][0])
                    if pcode2[j][0] in distribute2[k]:
                        newcode2[k].append(pcode2[j][0])
            newpop1[i], newpop2[i] = [distribute1, newcode1], [distribute2, newcode2]
    return newpop1, newpop2


# 采用初始化方法
def rule4(env, pop1, pop2, CR):
    pop1, pop2 = copy.deepcopy(pop1), copy.deepcopy(pop2)
    newpop1 = DE_utils.DE_init_rule1(env, len(pop1))
    newpop2 = DE_utils.DE_init_rule2(env, len(pop1))
    for i in range(len(pop2)):
        if random.random() > CR:
            newpop1[i] = pop1[i]
            newpop2[i] = pop2[i]
    return newpop1, newpop2


# 全随机
def rule5(pop1, pop2, CR):
    newpop1, newpop2 = copy.deepcopy(pop1), copy.deepcopy(pop2)
    for i in range(len(pop1)):
        if random.random() < CR:
            distribute1, distribute2 = copy.deepcopy(newpop1[i][0]), copy.deepcopy(newpop2[i][0])
            Pcode1, Pcode2 = copy.deepcopy(newpop1[i][1]), copy.deepcopy(newpop2[i][1])
            Onum = int(len(Pcode1[0])/len(distribute1[0]))
            newdistribute1, newdistribute2 = [[]for i in range(len(distribute1))], [[]for i in range(len(distribute2))]
            distribute1_ex, distribute2_ex = [i for j in distribute1 for i in j], [i for j in distribute2 for i in j]
            random.shuffle(distribute1_ex)
            random.shuffle(distribute2_ex)
            newcode1, newcode2 = [], []
            flag, f = 0, len(newdistribute1)
            for j in range(len(distribute1_ex)):
                newdistribute1[flag].append(distribute1_ex[j])
                newdistribute2[flag].append(distribute2_ex[j])
                if flag == f - 1:
                    flag = 0
                else:
                    flag += 1
            for j in range(f):
                p1 = newdistribute1[j] * Onum
                p2 = newdistribute2[j] * Onum
                random.shuffle(p1)
                random.shuffle(p2)
                newcode1.append(p1)
                newcode2.append(p2)
            newpop1[i], newpop2[i] = [newdistribute1, newcode1], [newdistribute2, newcode2]
    return newpop1, newpop2

