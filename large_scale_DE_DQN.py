import copy
import random
import ENV
import DE_utils
from DQN import DQN
import torch
import draw
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Jnum, Onum, Pnum, Fnum, Mnum = 800, 10, 80, 2, 10
    instance = ENV.Schedule_Instance(Jnum, Onum, Pnum, Fnum, Mnum)
    env = ENV.Schedule_Env(instance, Jnum, Onum, Pnum, Fnum, Mnum)
    individual_action = 7
    dqn = DQN((Jnum*Onum + Pnum)*3, individual_action)
    dqn.eval_net.load_state_dict(torch.load('eval_net_params_p.pkl'))
    dqn.target_net.load_state_dict(torch.load('target_net_params_p.pkl'))
    dqn.memory = torch.load('memory_p.pkl')
    # print(instance)
    C_record = [[], []]
    best = []
    action, loss_r = [], []
    epoch, popsize = 100, 100
    pop1, pop2, fitness = DE_utils.DE_init(env, popsize)
    Fmax, Fmin, ep_r, r = 0.9, 0.1, 0, 0
    CRmax, CRmin = 0.9, 0.6
    F, CR = Fmin, CRmin
    fitness1, best1, record1 = DE_utils.DE_calfitness(env, pop1, fitness)
    state1_pop = env.state
    fitness2, best2, record2 = DE_utils.DE_calfitness(env, pop2, fitness)
    state2_pop = env.state
    state_pop = [[state1_pop[i], state2_pop[i]] for i in range(len(state1_pop))]
    state_pop = [i for j in state_pop for i in j]
    for i in range(epoch):
        F = Fmin + ((i + 1) / epoch) * (Fmax - Fmin)
        CR = CRmin + ((i + 1) / epoch) * (CRmax - CRmin)
        # 变异
        newpop1, newpop2 = DE_utils.DE_mutation(F, pop1, pop2, best1[1][1], best2[1][1], fitness1, fitness2)
        newfitness1, _, newrecord1 = DE_utils.DE_calfitness(env, newpop1, fitness)
        newfitness2, _, newrecord2 = DE_utils.DE_calfitness(env, newpop2, fitness)
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2)
        # 针对种群的dqn交叉
        newpop1, newpop2 = DE_utils.DE_crossover(env, newpop1, newpop2, CR, random.randint(0, 4))
        newfitness1, _, newrecord1 = DE_utils.DE_calfitness(env, newpop1, fitness)
        newfitness2, _, newrecord2 = DE_utils.DE_calfitness(env, newpop2, fitness)
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2)

        # 针对个体的dqn交叉
        memory_state = [[], []]
        memory_action = [[], []]
        for j in range(popsize):
            state1, state2 = copy.deepcopy(record1[j]), copy.deepcopy(record2[j])
            memory_state[0].append(state1)
            memory_state[1].append(state2)
            state1 = [j for i in DE_utils.DE_record_state(state1) for j in i]
            state2 = [j for i in DE_utils.DE_record_state(state2) for j in i]
            if i < epoch / 2:
                a1 = random.randint(0, individual_action - 1)
                a2 = random.randint(0, individual_action - 1)
            else:
                a1 = dqn.choose_action(state1)
                a2 = dqn.choose_action(state2)
            action.append(a1)
            action.append(a2)
            memory_action[0].append(a1)
            memory_action[1].append(a2)

            newpop1[j] = DE_utils.DE_crossover_individual(env.Products, pop1[j], record1[j], a1)
            newpop2[j] = DE_utils.DE_crossover_individual(env.Products, pop2[j], record2[j], a2)

            newfitness1_, newbest1, record1_ = DE_utils.DE_calfitness(env, [newpop1[j]], fitness)
            newfitness2_, newbest2, record2_ = DE_utils.DE_calfitness(env, [newpop2[j]], fitness)
            newrecord1[j] = copy.deepcopy(record1_[0])
            newrecord2[j] = copy.deepcopy(record2_[0])
            newfitness1[j] = newfitness1_[0]
            newfitness2[j] = newfitness2_[0]
            r1 = fitness1[j] - newfitness1_[0]
            r2 = fitness2[j] - newfitness2_[0]
            r1, r2 = 0 if r1 < 0 else r1, 0 if r1 < 0 else r1
            state1_ = [j for i in (DE_utils.DE_record_state(record1_[0])) for j in i]
            state2_ = [j for i in (DE_utils.DE_record_state(record2_[0])) for j in i]
            dqn.store_transition(state1, a1, float(r1), state1_)
            dqn.store_transition(state2, a2, float(r2), state2_)
            loss_v = float(dqn.learn())
            loss_r.append(float(loss_v))
            if j % 100 == 0:
                print('epoch:', i, ', loss', loss_v)
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2)
        # 搜寻工序排列最优
        for j in range(popsize):
            newpop1[j] = DE_utils.DE_crossover_search(pop1[j])
            newpop2[j] = DE_utils.DE_crossover_search(pop2[j])
        newfitness1, newbest1, newrecord1 = DE_utils.DE_calfitness(env, newpop1, fitness)
        newfitness2, newbest2, newrecord2 = DE_utils.DE_calfitness(env, newpop2, fitness)
        pop1, pop2, fitness1, fitness2, record1, record2 = DE_utils.DE_select(newpop1, newpop2, pop1, pop2, newfitness1,
                                                                              newfitness2, fitness1,
                                                                              fitness2, record1, record2, newrecord1,
                                                                              newrecord2)
        # 保存最优解
        _, best1, _ = DE_utils.DE_calfitness(env, pop1, fitness)
        _, best2, _ = DE_utils.DE_calfitness(env, pop2, fitness)
        C_record[0].append(best1[0])
        C_record[1].append(best2[0])
    if best1[0] <= best2[0]:
        best = best1
    else:
        best = best2
    print(C_record)
    print(best)
    print(action)
    torch.save(dqn.eval_net.state_dict(), 'eval_net_params_p.pkl')
    torch.save(dqn.target_net.state_dict(), 'target_net_params_p.pkl')
    torch.save(dqn.memory, 'memory_p.pkl')
    # draw.draw(best[2])

    fig1 = plt.figure()
    ax = fig1.add_subplot()
    ax.set(
        ylabel='loss', xlabel='epoch')
    plt.plot(loss_r)
    plt.show()
