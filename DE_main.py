import copy
import pickle
import time
import ENV
import DE_utils
import draw
import matplotlib.pyplot as plt


def Solve_DAFJSP_DE(Jnum, Onum, Pnum, Fnum, Mnum):
    instance = ENV.Schedule_Instance(Jnum, Onum, Pnum, Fnum, Mnum)
    Schedule_path = 'D:\\学校\\学习\\python\\Paper\\DAFSP\\data\\' + str(Jnum) + '_' + str(Pnum) + '.pkl'
    f_read = open(Schedule_path, 'rb')
    schedule_instance = pickle.load(f_read)
    instance.jobT = schedule_instance['jobT']
    instance.Products = schedule_instance['Products']
    instance.Machine = schedule_instance['Machine']
    instance.AssembleTime = schedule_instance['AssembleTime']
    instance.Tansporttime = schedule_instance['Tansporttime']
    env = ENV.Schedule_Env(instance, Jnum, Onum, Pnum, Fnum, Mnum)
    # print(env.Products)
    C_record = []
    action, best = [], []
    epoch, popsize = 200, 200
    pop1, pop2, fitness = DE_utils.DE_init(env, popsize)
    pop1, pop2 = DE_utils.DE_crossover(env, pop1, pop2, 1, 4)
    Fmax, Fmin, ep_r, r = 1.5, 0.1, 0, 0
    CRmax, CRmin = 0.9, 0.6
    F, CR = Fmin, CRmin

    fitness1, best1, record1 = DE_utils.DE_calfitness(env, pop1, fitness)
    fitness2, best2, record2 = DE_utils.DE_calfitness(env, pop2, fitness)
    number_times = [[0 for i in range(popsize)] for j in range(2)]
    for i in range(epoch):
        F = Fmax + ((i) / (epoch - 1)) * (Fmin - Fmax)
        CR = CRmin + ((i) / (epoch - 1)) * (CRmax - CRmin)
        # 变异
        newpop1, newpop2 = DE_utils.DE_mutation(F, pop1, pop2, best1[1][1], best2[1][1], fitness1, fitness2)
        # 交叉
        for j in range(popsize):
            newpop1[j] = DE_utils.DE_crossover_search(newpop1[j])
            newpop2[j] = DE_utils.DE_crossover_search(newpop2[j])
        # 选择
        newfitness1, newbest1, newrecord1 = DE_utils.DE_calfitness(env, newpop1, fitness)
        newfitness2, newbest2, newrecord2 = DE_utils.DE_calfitness(env, newpop2, fitness)
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2, number_times, env)
        # 保存最优解
        _, best1, _ = DE_utils.DE_calfitness(env, pop1, fitness)
        _, best2, _ = DE_utils.DE_calfitness(env, pop2, fitness)
        if best1[0] <= best2[0]:
            best = best1
        else:
            best = best2
        C_record.append(best[0])
    return best, C_record


if __name__ == '__main__':
    start = time.time()

    # Jnum, Onum, Pnum, Fnum, Mnum = 15, 4, 5, 2, 6
    # Jnum, Onum, Pnum, Fnum, Mnum = 50, 4, 7, 2, 6
    Jnum, Onum, Pnum, Fnum, Mnum = 100, 4, 10, 2, 6
    best, C_record = Solve_DAFJSP_DE(Jnum, Onum, Pnum, Fnum, Mnum)

    end = time.time()
    print(end - start)

    draw.draw(best[2])
    print(C_record)
    print(best)

