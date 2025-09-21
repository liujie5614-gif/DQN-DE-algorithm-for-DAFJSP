import copy
import random
import Action_Space
import Action_Space_Individual
import ENV
import numpy as np


def DE_init(env, popsize):
    pop1 = DE_init_rule1(env, popsize)
    pop2 = DE_init_rule2(env, popsize)
    fitness = [0 for _ in range(popsize)]
    return pop1, pop2, fitness


def DE_mutation(f, pop1, pop2, best1code, best2code, fitness1, fitness2):
    newpop1, newpop2 = copy.deepcopy(pop1), copy.deepcopy(pop2)
    code1_best, code2_best = cros_addh(best1code), cros_addh(best2code)
    for i in range(len(newpop1)):
        gen1 = newpop1[i]
        gen2 = newpop2[i]
        # # 概率择优
        # if fitness1[i] < fitness2[i]:
        #     if random.random() < 0.7:
        #         distribute = gen1[0]
        #     else:
        #         distribute = gen2[0]
        # else:
        #     if random.random() < 0.3:
        #         distribute = gen1[0]
        #     else:
        #         distribute = gen2[0]
        # 点位交叉
        distribute1 = gen1[0]
        distribute2 = gen2[0]
        d_code1, d_code2 = [], []
        for j in range(len(distribute1)):
            for l in range(len(distribute1[j])):
                d_code1.append([j, distribute1[j][l]])
        for j in range(len(distribute2)):
            for l in range(len(distribute2[j])):
                d_code2.append([j, distribute2[j][l]])
        d_code1.sort(key=lambda x:x[1])
        d_code2.sort(key=lambda x:x[1])
        position = random.randint(0,len(d_code1)-1)
        newd = d_code1[:position] + d_code2[position:]
        distribute = [[]for i in range(len(distribute1))]
        for factory, job in newd:
            distribute[factory].append(job)
        # 作差（有缺陷，完全相同或完全不同时，工件都被分配到同一工厂）
        # distribute1 = gen1[0]
        # distribute2 = gen2[0]
        # d_code1, d_code2 = [], []
        # for j in range(len(distribute1)):
        #     for l in range(len(distribute1[j])):
        #         d_code1.append([j, distribute1[j][l]])
        # for j in range(len(distribute2)):
        #     for l in range(len(distribute2[j])):
        #         d_code2.append([j, distribute2[j][l]])
        # d_code1.sort(key=lambda x:x[1])
        # d_code2.sort(key=lambda x:x[1])
        # d_diff = []
        # for j in range(len(d_code1)):
        #     temp = d_code1[j][0] - d_code2[j][0]
        #     # d_diff.append([abs(temp) if temp != 0 else d_code1[j][0], j])
        #     d_diff.append([abs(temp), j])
        # distribute = [[]for i in range(len(distribute1))]
        # for factory, job in d_diff:
        #     distribute[factory].append(job)
        ProcessCode1 = gen1[1]
        ProcessCode2 = gen2[1]
        r = random.randint(0, len(newpop1)-1)
        rgen1, rgen2 = newpop1[r], newpop2[r]
        cros_r1, cros_r2 = cros_addh(rgen1[1]), cros_addh(rgen2[1])
        cros_p1, cros_p2 = cros_addh(ProcessCode1), cros_addh(ProcessCode2)
        diff11 = [[cros_p1[i][0], code1_best[i][1] - cros_p1[i][1]]for i in range(len(cros_r1))]
        diff21 = [[cros_p2[i][0], code2_best[i][1] - cros_p2[i][1]]for i in range(len(cros_r2))]
        diff2 = [[cros_p1[i][0], cros_r1[i][1] - cros_r2[i][1]]for i in range(len(cros_r1))]
        res1 = [[cros_p2[i][0], cros_p1[i][1] + f*diff11[i][1] + f*diff2[i][1]]for i in range(len(cros_r1))]
        res2 = [[cros_p2[i][0], cros_p2[i][1] + f*diff21[i][1] + f*diff2[i][1]]for i in range(len(cros_r2))]
        res1.sort(key=lambda x: x[1], reverse=True)
        res2.sort(key=lambda x: x[1], reverse=True)
        newcode1 = [[] for i in range(len(distribute))]
        newcode2 = [[] for i in range(len(distribute))]
        for j in range(len(res1)):
            for k in range(len(distribute)):
                if(res1[j][0] in distribute[k]):
                    newcode1[k].append(res1[j][0])
                if (res2[j][0] in distribute[k]):
                    newcode2[k].append(res2[j][0])
        newgen1 = [distribute, newcode1]
        newgen2 = [distribute, newcode2]
        newpop1[i], newpop2[i] = newgen1, newgen2
    return newpop1, newpop2


def cros_addh(p1):
    h_list = [i for i in range(max([len(p1[i])for i in range(len(p1))]))]
    h_list.reverse()
    cros_p1 = []
    for i in range(len(p1)):
        cros_p1.append([[p1[i][j], h_list[j]] for j in range(len(p1[i]))])
    cros_p1 = [i for j in cros_p1 for i in j]
    cros_p1.sort(key=lambda x: x[0])
    return cros_p1


def DE_crossover(env, pop1, pop2, CR, a):
    newpop1, newpop2 = Action_Space.Select_DE_crossover(env, pop1, pop2, CR, a)
    return newpop1, newpop2


def DE_crossover_individual(Products, gen, record, a):
    newgen = Action_Space_Individual.Select_DE_crossover(Products, gen, record, a)
    return newgen


def DE_crossover_search(gen):
    newgen, gen = copy.deepcopy(gen), copy.deepcopy(gen)
    distribute, Pcode = gen[0], gen[1]
    # Onum = int(len(Pcode[0]) / len(distribute[0]))
    # jnum = sum([len(distribute[i]) for i in range(len(distribute))])
    Fnum = len(gen[0])
    # ex_position = [random.sample(range(0, len(gen[1][i])), 1 if len(gen[1][i]) < 4 else 4)for i in range(Fnum)]
    ex_position = [random.sample(range(0, len(gen[1][i])), len(gen[1][i])//10)for i in range(Fnum)]
    ex_job = [[]for i in range(Fnum)]
    for i in range(Fnum):
        for j in ex_position[i]:
            ex_job[i].append(gen[1][i][j])
    for i in range(Fnum):
        random.shuffle(ex_job[i])
    for i in range(Fnum):
        for j in range(len(ex_position[i])):
            newgen[1][i][ex_position[i][j]] = ex_job[i][j]
    return newgen


def DE_select(newpop1, newpop2, pop1, pop2, newfitness1, newfitness2, fitness1, fitness2, record1, record2, newrecord1, newrecord2, number_times, env):
    newpop1, newpop2, pop1, pop2, record1, record2 = copy.deepcopy(newpop1), copy.deepcopy(newpop2), copy.deepcopy(pop1), copy.deepcopy(pop2), copy.deepcopy(record1), copy.deepcopy(record2)
    spop1, spop2, fitness1, fitness2, newrecord1, newrecord2 = copy.deepcopy(pop1), copy.deepcopy(pop2), copy.deepcopy(fitness1), copy.deepcopy(fitness2), copy.deepcopy(newrecord1), copy.deepcopy(newrecord2)
    p1, p2, fitness = DE_init(env, len(newpop1))
    fit1, _, rec1 = DE_calfitness(env, p1, fitness)
    fit2, _, rec2 = DE_calfitness(env, p2, fitness)
    for i in range(len(pop1)):
        if newfitness1[i] < fitness1[i]:
            spop1[i] = newpop1[i]
            fitness1[i] = newfitness1[i]
            record1[i] = newrecord1[i]
        else:
            number_times[0][i] += 1
            if number_times[0][i] >= 20:
                spop1[i] = p1.pop()
                fitness1[i] = fit1.pop()
                record1[i] = rec1.pop()
            number_times[0][i] = 0
        if newfitness2[i] < fitness2[i]:
            spop2[i] = newpop2[i]
            fitness2[i] = newfitness2[i]
            record2[i] = newrecord2[i]
        else:
            number_times[1][i] += 1
            if number_times[1][i] >= 20:
                spop1[i] = p2.pop()
                fitness1[i] = fit2.pop()
                record1[i] = rec2.pop()
            number_times[1][i] = 0
    # spop1, spop2 = [], []
    # tempt1, tempt2 = pop1 + newpop1, pop2 + newpop2
    # f1, f2 = fitness1 + newfitness1, fitness2 + newfitness2
    # tempt1, tempt2 = [[f1[i], tempt1[i]]for i in range(len(tempt1))], [[f2[i], tempt2[i]]for i in range(len(tempt2))]
    # tempt1.sort(key=lambda x: x[0])
    # tempt2.sort(key=lambda x: x[0])
    # for i in range(len(pop1)):
    #     if i < 10:
    #         spop1.append(tempt1[i][1])
    #         spop2.append(tempt2[i][1])
    #     else:
    #         spop1.append(tempt1[random.randint(10, len(tempt1)-1)][1])
    #         spop2.append(tempt2[random.randint(10, len(tempt1)-1)][1])
    return spop1, spop2, fitness1, fitness2, record1, record2


# 计算适应度
def DE_calfitness(env, lpop, fit):
    fitness = [0 for _ in range(len(lpop))]
    pop = copy.deepcopy(lpop)
    # fitness = copy.copy(fit)
    record = [[[[] for k in range(env.Mnum + 2)] for j in range(env.Fnum)] for i in range(len(pop))]
    fitness_p = []
    assemble_st = []
    for i in range(len(pop)):
        env.reset()
        PT = env.jobT
        Machine = env.Machine
        jobs = env.jobs
        machines = env.machines
        gen = pop[i]
        distribute = gen[0]
        ProcessCode = gen[1]
        for j in range(env.Fnum):
            for k in distribute[j]:  # 分配的工件
                jobs[k].belongToF = j
        # 计算每一个个体的适应度
        ProcessNum = [-1 for _ in range(env.Jnum)]
        for j in range(env.Fnum):
            for k in ProcessCode[j]:
                job = jobs[k]
                groupmachines = machines[j]
                onprocess = job.op
                job.op += 1
                ptimes = PT[k][onprocess]
                machine = Machine[k][onprocess]
                mnum = len(ptimes)
                if mnum == 1:
                    t = ptimes[0]
                    m = groupmachines[machine[0]]
                    met = m.endt
                    jet = job.endt
                    st = max(met, jet)
                    et = st + t
                    env.update(st, et, k, j, machine[0])
                else:
                    sts = []
                    for x in range(mnum):
                        m = groupmachines[machine[x]]
                        sts.append(job.endt if job.endt >= m.endt else m.endt)
                    a, b = np.array(sts), np.array(ptimes)
                    c = (a + b).tolist()
                    index = 0
                    for x in range(mnum):
                        if c[index] >= c[x]:
                            index = x
                    st = a.tolist()[index]
                    t = b.tolist()[index]
                    et = st + t
                    env.update(st, et, k, j, machine[index])
        machines_end = []
        for k in range(env.Fnum):
            for x in range(env.Mnum):
                machines_end.append(machines[k][x].endt)
        # 加工的最大完工时间
        C_max_p = max(machines_end)
        fitness_p.append(C_max_p)
        # 装配后的最大完工时间
        TT = env.Tansporttime
        AT = env.AssembleTime
        job_endt = [[] for _ in range(env.Pnum)]
        job_F = [[] for _ in range(env.Pnum)]
        for j in range(env.Pnum):
            for k in env.Products[j]:
                job_endt[j].append(jobs[k].endt)
                job_F[j].append(jobs[k].belongToF)
        # job_endt_copy = copy.copy(job_endt)
        pro_et = [[] for _ in range(env.Pnum)]
        TT_record = [[[]for _ in range(env.Fnum)] for _ in range(env.Pnum)]
        for j in range(env.Pnum):
            pros = env.Products[j]
            for k in range(env.Fnum):
                job_endt_copy = copy.deepcopy(job_endt)
                for li in range(len(pros)):
                    if job_F[j][li] != k:
                        # job_endt_copy[j][li] = job_endt[j][li] + TT[j][li]
                        job_endt_copy[j][li] = [job_endt[j][li], job_endt[j][li] + TT[j][li], pros[li]]
                        TT_record[j][k].append([job_endt[j][li], job_endt[j][li] + TT[j][li], pros[li]])
                    else:
                        job_endt_copy[j][li] = [job_endt[j][li], job_endt[j][li], pros[li]]
                job_endt_copy[j].sort(key=lambda y: y[1],reverse=True)
                # TT_record[j][k]=[job_endt_copy[j][0][0], job_endt_copy[j][0][1], job_endt_copy[j][0][2]]
                pro_et[j].append(job_endt_copy[j][0][1])
        P_et_min = [[i] for i in range(env.Pnum)]
        for j in range(env.Pnum):
            P_et_min[j].append(min(pro_et[j]))
        P_et_min.sort(key=lambda y: y[1])
        acode = list(map(lambda y: y[0], P_et_min))
        assemble_sts = []
        T_record = [[]for _ in range(env.Fnum)]
        for j in range(env.Pnum):
            asts = [[i] for i in range(env.Fnum)]
            ap = acode[j]
            for k in range(env.Fnum):
                amachine = env.A_machines[k]
                asts[k].append(max(pro_et[ap][k], amachine.endt))
            asts.sort(key=lambda y: y[1])
            af = asts[0][0]
            ast = asts[0][1]
            aet = ast + AT[ap]
            assemble_sts.append(ast)
            env.A_machines[af].endt = aet
            env.A_machines[af].Start_Processing.append(ast)
            env.A_machines[af].End_Processing.append(aet)
            env.A_machines[af].jobs.append(ap)
            T_record[af].append(TT_record[ap][af])
        assemble_end = []
        for j in range(env.Fnum):
            assemble_end.append(env.A_machines[j].endt)
        C_max_A = max(assemble_end)
        assemble_st.append(min(assemble_sts))
        fitness[i] = C_max_A
        for j in range(env.Fnum):
            for k in range(env.Mnum):
                record[i][j][k].append(machines[j][k].Start_Processing)
                record[i][j][k].append(machines[j][k].End_Processing)
                record[i][j][k].append(machines[j][k].jobs)
            T_st, T_et, T_job = [], [], []
            for n in range(len(T_record[j])):
                for each_t in range(len(T_record[j][n])):
                    T_st.append(T_record[j][n][each_t][0])
                    T_et.append(T_record[j][n][each_t][1])
                    T_job.append(T_record[j][n][each_t][2])
            record[i][j][env.Mnum].append(T_st)
            record[i][j][env.Mnum].append(T_et)
            record[i][j][env.Mnum].append(T_job)
            record[i][j][env.Mnum+1].append(env.A_machines[j].Start_Processing)
            record[i][j][env.Mnum+1].append(env.A_machines[j].End_Processing)
            record[i][j][env.Mnum+1].append(env.A_machines[j].jobs)
        # record[i].append(TT_record)
    best = []

    fitness_copy = [[fitness[i], i]for i in range(len(fitness))]
    fitness_copy.sort(key=lambda y: y[0])
    best_gen = pop[fitness_copy[0][1]]
    best_C = fitness_copy[0][0]
    worst_C = fitness_copy[len(fitness_copy)-1][0]

    best_record = record[fitness_copy[0][1]]
    best.append(best_C)
    best.append(best_gen)
    best.append(best_record)
    # 更新状态
    fitness_A = [fitness[i] - assemble_st[i] for i in range(len(fitness))]
    env.set_state(best_C, np.var(fitness), np.var(fitness_p), np.var(fitness_A), np.var(assemble_st), worst_C - best_C)
    return fitness, best, record


def DE_record_state(record):
    record = copy.deepcopy(record)
    s, e, job = [], [], []
    for i in record:
        for j in i:
            s.append(j[0])
            e.append(j[1])
            job.append(j[2])
    return s+e+job


# 将工件尽可能分散在不同工厂
def DE_init_rule1(env, popsize):
    pop = []
    for j in range(popsize):
        gen = []
        F = env.Fnum
        distribution = [[]for _ in range(F)]
        products = []
        pindex = [_ for _ in range(env.Pnum)]
        random.shuffle(pindex)
        for i in range(env.Pnum):
            products.append(env.Products[pindex[i]])
        # job_list = np.array(products).flatten().tolist()
        job_list = [j for i in products for j in i]
        flag = 0
        for i in range(env.Jnum):
            distribution[flag].append(job_list[i])
            flag += 1
            if flag >= F:
                flag = 0
        gen.append(distribution)
        p = []
        for i in range(F):
            p1 = distribution[i]*env.n
            random.shuffle(p1)
            p.append(p1)
        gen.append(p)
        pop.append(gen)
    return pop


# 将工件尽可能分散在相同工厂
def DE_init_rule2(env, popsize):
    pop = []
    for j in range(popsize):
        gen = []
        F = env.Fnum
        n = int(env.Jnum/F)
        distribution = [[]for _ in range(F)]
        products = []
        pindex = [_ for _ in range(env.Pnum)]
        random.shuffle(pindex)
        for i in range(env.Pnum):
            products.append(env.Products[pindex[i]])
        # job_list = np.array(products).flatten().tolist()
        job_list = [j for i in products for j in i]
        for i in range(F):
            if i < F-1:
                distribution[i].extend(job_list[n*i:n*(i+1)])
            else:
                distribution[i].extend(job_list[n*i:])
        gen.append(distribution)
        p = []
        for i in range(F):
            p1 = distribution[i]*env.n
            random.shuffle(p1)
            p.append(p1)
        gen.append(p)
        pop.append(gen)
    return pop


# 按工序排序分配
def init_rule1(env, popsize):
    pop = []
    for i in range(popsize):
        operation_list = []
        for j in range(env.Jnum):
            for k in range(env.n):
                operation_list.append([j, min(env.jobT[j][k])])
        operation_list.sort(key=lambda x: x[1])
        job_lists = [operation_list[i][0] for i in range(len(operation_list))]
        job_list = []
        for j in job_lists:
            if j not in job_list:
                job_list.append(j)
        distribution = [[]for i in range(env.Fnum)]
        for j in range(env.Fnum):
            if j == env.Fnum - 1:
                distribution[j] = job_list[j * (env.Jnum // env.Fnum):]
            else:
                distribution[j] = job_list[j*(env.Jnum//env.Fnum):(j+1)*(env.Jnum//env.Fnum)]
        p = []
        for j in range(env.Fnum):
            p1 = distribution[j] * env.n
            random.shuffle(p1)
            p.append(p1)
        pop.append([distribution, p])
    return pop


# 按工件排序分配
def init_rule2(env, popsize):
    pop = []
    for i in range(popsize):
        operation_list = [[]for i in range(env.Jnum)]
        for j in range(env.Jnum):
            for k in range(env.n):
                operation_list[j].append(min(env.jobT[j][k]))
        operation_list = [[i, sum(operation_list[i])] for i in range(len(operation_list))]
        operation_list.sort(key=lambda x: x[1])
        job_list = [operation_list[i][0] for i in range(len(operation_list))]
        distribution = [[] for i in range(env.Fnum)]
        for j in range(env.Fnum):
            if j == env.Fnum - 1:
                distribution[j] = job_list[j * (env.Jnum // env.Fnum):]
            else:
                distribution[j] = job_list[j * (env.Jnum // env.Fnum):(j + 1) * (env.Jnum // env.Fnum)]
        p = []
        for j in range(env.Fnum):
            p1 = distribution[j] * env.n
            random.shuffle(p1)
            p.append(p1)
        pop.append([distribution, p])
    return pop

