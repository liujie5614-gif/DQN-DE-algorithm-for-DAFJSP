import copy
import random
import numpy as np


class job:
    def __init__(self, index):
        self.index = index
        self.endt = 0
        self.belongToF = 0
        self.belongToPro = 0
        self.op = 0


class Machine:
    def __init__(self, index):
        self.index = index
        self.endt = 0
        self.Start_Processing = []
        self.End_Processing = []
        self.jobs = []


class Schedule_Env:
    def __init__(self, Instance, Jnum, n, Pnum, Fnum, Mnum):
        self.Jnum = Jnum
        self.n = n
        self.Pnum = Pnum
        self.Fnum = Fnum
        self.Mnum = Mnum
        self.jobT = copy.copy(Instance.jobT)
        self.Products = copy.copy(Instance.Products)
        self.Machine = copy.copy(Instance.Machine)
        self.Tansporttime = copy.copy(Instance.Tansporttime)
        self.AssembleTime = copy.copy(Instance.AssembleTime)

    def reset(self):
        self.state = [0 for _ in range(6)]
        self.jobs = []
        self.machines = [[] for _ in range(self.Fnum)]
        self.A_machines = [Machine(0) for _ in range(self.Fnum)]
        for i in range(self.Jnum):
            wi = job(i)
            self.jobs.append(wi)
        for i in range(self.Pnum):
            for j in self.Products[i]:
                self.jobs[j].belongToPro = i
        for j in range(self.Fnum):
            for i in range(self.Mnum):
                Mi = Machine(i)
                self.machines[j].append(Mi)

    def set_state(self, C, varc, varp, vara, varas, mr):
        self.state[0] = C
        self.state[1] = varc
        self.state[2] = varp
        self.state[3] = vara
        self.state[4] = varas
        self.state[5] = mr

    def update(self, st, et, j, F, m):
        job_ = self.jobs[j]
        machine = self.machines[F][m]
        job_.endt = et
        machine.endt = et
        machine.Start_Processing.append(st)
        machine.End_Processing.append(et)
        machine.jobs.append(j)


class Schedule_Instance:
    def __init__(self, Jnum, n, Pnum, Fnum, Mnum):
        self.Jnum = Jnum    # 工件
        self.n = n          # 工序
        self.Pnum = Pnum    # 产品数量
        self.Fnum = Fnum    # 工厂数量
        self.Mnum = Mnum    # 机器数量
        self.jobT = [[] for _ in range(Jnum)]
        self.Products = [[] for _ in range(Pnum)]
        self.Tansporttime = [[] for _ in range(Pnum)]
        self.Machine = [[] for _ in range(Jnum)]
        self.AssembleTime = []
        for i in range(Jnum):
            for j in range(n):
                selected_m = 1
                if random.random() < 0.4:
                    selected_m = random.randint(1, 2)
                # self.jobT[i].append(np.random.randint(10, 50, size=selected_m).tolist())
                self.jobT[i].append(np.random.randint(10, 20, size=selected_m).tolist())     # real-life

        for i in range(Jnum):
            Machine_list = [i for i in range(Mnum)]
            random.shuffle(Machine_list)
            for j in range(n):
                l = len(self.jobT[i][j])
                if len(Machine_list) < l:
                    self.Machine[i].append(random.sample(range(0, Mnum), l))
                else:
                    self.Machine[i].append(Machine_list[0:l])
                    for k in range(l):
                        Machine_list.pop(0)
                # self.Machine[i].append(random.sample(range(0, Mnum), len(self.jobT[i][j])))

        job_list = [_ for _ in range(Jnum)]
        random.shuffle(job_list)
        for i in range(2):
            for j in range(self.Pnum):
                self.Products[j].append(job_list[j+i*self.Pnum])
                # self.Tansporttime[j].append(random.randint(10, 30))
                self.Tansporttime[j].append(random.randint(20, 40))   # real-life
        for job_p in job_list[2*self.Pnum:]:
            random_p = random.randint(0, self.Pnum-1)
            self.Products[random_p].append(job_p)
            # self.Tansporttime[random_p].append(random.randint(10, 30))
            self.Tansporttime[random_p].append(random.randint(30, 50))   # real-life
        # products_list = [_ for _ in range(Pnum)]
        # random.shuffle(products_list)
        # index = 0
        # for i in range(Jnum):
        #     if index >= Pnum:
        #         index = 0
        #     self.Products[products_list[products_list[index]]].append(job_list[i])
        #     self.Tansporttime[products_list[products_list[index]]].append(random.randint(10, 50))
        #     index += 1
        for i in range(Pnum):
            # self.AssembleTime.append(random.randint(10, 50))
            self.AssembleTime.append(random.randint(30, 50))   # real-life

    def __str__(self):
        return f'{self.jobT}\n' \
               f'{self.Products}\n' \
               f'{self.Machine}\n' \
               f'{self.AssembleTime}\n' \
               f'{self.Tansporttime}\n'
