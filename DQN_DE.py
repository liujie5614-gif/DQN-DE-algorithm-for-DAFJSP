import copy
import pickle
import random
import time
import ENV
import DE_utils
from DQN import DQN
import torch
import draw
import matplotlib.pyplot as plt


def Solve_DAFJSP_DQN_DE(Jnum, Onum, Pnum, Fnum, Mnum):
    instance = ENV.Schedule_Instance(Jnum, Onum, Pnum, Fnum, Mnum)
    Schedule_path = 'D:\\学校\\学习\\python\\Paper\\DAFSP\\data\\' + str(Jnum) + '_' + str(Pnum) + '.pkl'
    model_Path = 'D:\\学校\\学习\\python\\Paper\\DAFSP\\results\\' + str(Jnum) + '_' + str(Pnum) + '\\'
    f_read = open(Schedule_path, 'rb')
    schedule_instance = pickle.load(f_read)
    f_read.close()
    instance.jobT = schedule_instance['jobT']
    instance.Products = schedule_instance['Products']
    instance.Machine = schedule_instance['Machine']
    instance.AssembleTime = schedule_instance['AssembleTime']
    instance.Tansporttime = schedule_instance['Tansporttime']
    env = ENV.Schedule_Env(instance, Jnum, Onum, Pnum, Fnum, Mnum)
    individual_action = 7
    dqn = DQN((Jnum * Onum + Pnum) * 3, individual_action)
    dqn_pop = DQN(12, 5)
    loss_r = []
    try:
        dqn.eval_net.load_state_dict(torch.load(model_Path + 'eval_net_params_in.pkl'))
        dqn_pop.eval_net.load_state_dict(torch.load(model_Path + 'eval_net_params_p.pkl'))
        dqn.target_net.load_state_dict(torch.load(model_Path + 'target_net_params_in.pkl'))
        dqn_pop.target_net.load_state_dict(torch.load(model_Path + 'target_net_params_p.pkl'))
    except:
        print('--')
    # print(env.Products)
    C_record = []
    action, best = [], []
    epoch, popsize = 200, 200
    pop1, pop2, fitness = DE_utils.DE_init(env, popsize)
    Fmax, Fmin, ep_r, r = 1.0, 0.2, 0, 0
    CRmax, CRmin = 0.9, 0.6
    fitness1, best1, record1 = DE_utils.DE_calfitness(env, pop1, fitness)
    state1_pop = env.state
    fitness2, best2, record2 = DE_utils.DE_calfitness(env, pop2, fitness)
    state2_pop = env.state
    state_pop = [[state1_pop[i], state2_pop[i]] for i in range(len(state1_pop))]
    state_pop = [i for j in state_pop for i in j]
    best = best1 if best1[0] < best2[0] else best2
    number_times = [[0 for i in range(popsize)]for j in range(2)]
    for i in range(epoch):
        F = Fmax + ((i) / (epoch - 1)) * (Fmin - Fmax)
        CR = CRmin + ((i) / (epoch - 1)) * (CRmax - CRmin)
        # 变异
        newpop1, newpop2 = DE_utils.DE_mutation(F, pop1, pop2, best1[1][1], best2[1][1], fitness1, fitness2)
        newfitness1, best1, newrecord1 = DE_utils.DE_calfitness(env, newpop1, fitness)
        newfitness2, best2, newrecord2 = DE_utils.DE_calfitness(env, newpop2, fitness)
        if best1[0] <= best2[0]:
            if best1[0] < best[0]:
                best = best1
        else:
            if best2[0] < best[0]:
                best = best2
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                        newfitness2, fitness1,
                                                                        fitness2, record1, record2, newrecord1,
                                                                        newrecord2, number_times, env)
        # 针对种群的dqn交叉
        a_pop = dqn_pop.choose_action(state_pop)
        newpop1, newpop2 = DE_utils.DE_crossover(env, newpop1, newpop2, CR, a_pop)
        newfitness1, best1, newrecord1 = DE_utils.DE_calfitness(env, newpop1, fitness)
        newfitness2, best2, newrecord2 = DE_utils.DE_calfitness(env, newpop2, fitness)
        if best1[0] <= best2[0]:
            if best1[0] < best[0]:
                best = best1
        else:
            if best2[0] < best[0]:
                best = best2
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2, number_times, env)
        # 针对个体的dqn交叉
        memory_action = [[], []]
        for j in range(popsize):
            if random.random() > CR:
                state1, state2 = copy.deepcopy(record1[j]), copy.deepcopy(record2[j])
                for k in range(Fnum):
                    state1[k].pop(-2)
                    state2[k].pop(-2)
                state1 = [j for i in DE_utils.DE_record_state(state1) for j in i]
                state2 = [j for i in DE_utils.DE_record_state(state2) for j in i]
                a1 = dqn.choose_action(state1)
                a2 = dqn.choose_action(state2)

                action.append(a1)
                action.append(a2)
                newpop1[j] = DE_utils.DE_crossover_individual(env.Products, pop1[j], record1[j], a1)
                newpop2[j] = DE_utils.DE_crossover_individual(env.Products, pop2[j], record2[j], a2)
                newfitness1_, best1, record1_ = DE_utils.DE_calfitness(env, [newpop1[j]], fitness)
                newfitness2_, best2, record2_ = DE_utils.DE_calfitness(env, [newpop2[j]], fitness)
                newrecord1[j] = copy.deepcopy(record1_[0])
                newrecord2[j] = copy.deepcopy(record2_[0])
                newfitness1[j] = newfitness1_[0]
                newfitness2[j] = newfitness2_[0]
                if best1[0] <= best2[0]:
                    if best1[0] < best[0]:
                        best = best1
                else:
                    if best2[0] < best[0]:
                        best = best2
            else:
                newpop1[j] = pop1[j]
                newpop2[j] = pop2[j]
                newfitness1[j] = fitness1[j]
                newfitness2[j] = fitness2[j]
                newrecord1[j] = record1[j]
                newrecord2[j] = record2[j]
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2, number_times, env)

        # 搜寻工序排列最优
        for j in range(popsize):
            newpop1[j] = DE_utils.DE_crossover_search(pop1[j])
            newpop2[j] = DE_utils.DE_crossover_search(pop2[j])
        newfitness1, best1, newrecord1 = DE_utils.DE_calfitness(env, newpop1, fitness)
        newfitness2, best2, newrecord2 = DE_utils.DE_calfitness(env, newpop2, fitness)
        if best1[0] <= best2[0]:
            if best1[0] < best[0]:
                best = best1
        else:
            if best2[0] < best[0]:
                best = best2
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2, number_times, env)
        # 保存最优解
        _, best1, _ = DE_utils.DE_calfitness(env, pop1, fitness)
        state1_pop = env.state
        _, best2, _ = DE_utils.DE_calfitness(env, pop2, fitness)
        state2_pop = env.state
        state_pop = [[state1_pop[i], state2_pop[i]] for i in range(len(state1_pop))]
        state_pop = [i for j in state_pop for i in j]

        if best1[0] <= best2[0]:
            if best1[0] < best[0]:
                best = best1
        else:
            if best2[0] < best[0]:
                best = best2
        C_record.append(best[0])
    return best, C_record, action


if __name__ == '__main__':
    start = time.time()
    best, C_record = [99999], []
    Jnum, Onum, Pnum, Fnum, Mnum = 15, 4, 5, 2, 6
    # Jnum, Onum, Pnum, Fnum, Mnum = 50, 4, 7, 2, 6
    # Jnum, Onum, Pnum, Fnum, Mnum = 100, 4, 10, 2, 6
    # Jnum, Onum, Pnum, Fnum, Mnum = 60, 4, 6, 2, 6
    # best, C_record, action = Solve_DAFJSP_DQN_DE(Jnum, Onum, Pnum, Fnum, Mnum)

    end = time.time()
    schedule_instance_name = 'D:\\学校\\学习\\python\\Paper\\DAFSP\\' + str(Jnum) + '_' + str(Pnum) + '_' + str(
        Fnum) + '\\' + str(Jnum) + '_' + str(Pnum) + '.pkl'
    try:
        f_read = open(schedule_instance_name, 'rb')
        res_dict = pickle.load(f_read)
        C_record_pre, best_pre = res_dict['C_record'], res_dict['best']
        if best[0] >= best_pre[0]:
            best = best_pre
            C_record = C_record_pre
        f_read.close()
    except:
        print("--")
    print(end-start)
    print(C_record)
    print(best)
    # print(action)

    res = {'C_record': C_record, 'best': best}
    # schedule_instance_name = 'D:\\学校\\学习\\python\\Paper\\DAFSP\\' + str(Jnum) + '_' + str(Pnum) + '.pkl'
    f_save = open(schedule_instance_name, 'wb')
    pickle.dump(res, f_save)
    f_save.close()

    draw.draw(best[2], str(Jnum) + '-' + str(Pnum) + '-' + str(Fnum))
    fig1 = plt.figure()
    ax = fig1.add_subplot()
    ax.set(
        ylabel='Cmax', xlabel='epoch')
    ax.set(title=str(Jnum) + '_' + str(Pnum) + '_' + str(Fnum))
    plt.plot(C_record)
    plt.savefig(str(Jnum) + '_' + str(Pnum) + '_' + str(Fnum), dpi=800)


