import collections
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_num_threads(100)

BATCH_SIZE = 64
MEMORY_CAPACITY = 2000
LR = 0.0001                   # learning rate
EPSILON = 0.95              # greedy policy
GAMMA = 0.95                   # reward discount
TARGET_REPLACE_ITER = 50   # target update frequency
ES = 5000


class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.001)   # initialization
        self.fc2 = nn.Linear(512, 32)
        self.fc2.weight.data.normal_(0, 0.001)   # initialization

        self.fc3 = nn.Linear(32, 64)
        self.fc3.weight.data.normal_(0, 0.001)  # initialization
        self.fc4 = nn.Linear(64, 128)
        self.fc4.weight.data.normal_(0, 0.001)  # initialization

        self.out = nn.Linear(128, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.001)   # initialization

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=3,  # input shape (3,J_num,M_num)
        #         out_channels=6,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,  # 使得出来的图片大小不变P=（3-1）/2,
        #     ),  # output shape (3,J_num,M_num)
        #     nn.ReLU()  # output shape:  (6,int(J_num/2),int(M_num/2))
        # )
        # self.val_hidden = nn.Linear(3*3 * int(J_num / 2) * int(M_num / 2), 256)
        # self.val_hidden.weight.data.normal_(0, 0.01)
        # self.adv_hidden = nn.Linear(3*3 * int(J_num / 2) * int(M_num / 2), 256)
        # self.adv_hidden.weight.data.normal_(0, 0.01)
        # self.val = nn.Linear(256, 1)
        # self.val.weight.data.normal_(0, 0.01)
        # self.adv = nn.Linear(256, N_ACTIONS)
        # self.adv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        x = torch.relu(x)

        actions_value = F.relu(self.out(x))
        # actions_value = torch.sigmoid(self.out(x))
        return actions_value

        # x = self.conv1(x)
        # x = x.view(x.size(0), -1)
        # val_hidden = self.val_hidden(x)
        # val_hidden = F.relu(val_hidden)
        # adv_hidden = self.adv_hidden(x)
        # adv_hidden = F.relu(adv_hidden)
        # val = self.val(val_hidden)
        # adv = self.adv(adv_hidden)
        # adv_ave = torch.mean(adv, dim=1, keepdim=True)
        # x = adv + val - adv_ave
        # return x


class DQN(object):
    def __init__(self, N_STATES, N_ACTIONS):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)
        self.eval_net.to(device)
        self.target_net.to(device)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.Min_EPSILON = 0.1
        self.Max_EPSILON = 0.95
        # self.EPSILON = self.Min_EPSILON
        self.EPSILON = 0.7
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.CrossEntropyLoss()
        self.record = 0
        self.es = ES

    def choose_action(self, x):
        # x = [(xx-min(x))/(max(x)-min(x)) for xx in x]
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() <= self.EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            # actions_value = self.target_net.forward(x)
            # action = torch.max(actions_value, 1)[1].data.numpy()
            # action = torch.argmax(actions_value, 1).data.numpy()
            action = torch.argmax(actions_value, 1).data.cpu().numpy()
            action = action[0]   # return the argmax index
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action
        # self.record += 1
        # self.EPSILON = self.Min_EPSILON + (self.Max_EPSILON - self.Min_EPSILON) * min(1.0, (self.record / self.es))
        return action

    def store_transition(self, s, a, r, s_):
        # transition = np.hstack((s, [a, r], s_))
        # # replace the old memory with new memory
        # index = self.memory_counter % MEMORY_CAPACITY
        # self.memory[index, :] = transition
        self.memory.append((s, a, r, s_))
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        batch = copy.deepcopy(batch)
        # sample batch from memory
        batch_state = np.array([o[0] for o in batch])
        batch_next_state = np.array([o[3] for o in batch])
        batch_action = np.array([o[1] for o in batch])
        batch_reward = np.array([o[2] for o in batch])
        batch_action = torch.LongTensor(np.reshape(batch_action, (-1, len(batch_action)))).to(device)
        batch_reward = torch.LongTensor(np.reshape(batch_reward, (-1, len(batch_reward)))).to(device)
        # batch_state = torch.FloatTensor(np.reshape(batch_state, (-1, 3, J_num, M_num))).to(device)
        batch_state = torch.FloatTensor(np.reshape(batch_state, (-1, self.N_STATES))).to(device)
        # batch_next_state = torch.FloatTensor(np.reshape(batch_next_state, (-1, 3, J_num, M_num))).to(device)
        batch_next_state = torch.FloatTensor(np.reshape(batch_next_state, (-1, self.N_STATES))).to(device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        # q_eval = self.eval_net(batch_state)
        # q_eval = q_eval.gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()

        # q_target = batch_reward + GAMMA * q_next.argmax(1)[0].view(1, min(BATCH_SIZE, len(self.memory)))
        q_target = batch_reward + GAMMA * q_next.argmax(1)[0]

        # q_eval = self.eval_net(batch_state).gather(1, batch_action)
        # q_next_eval = self.eval_net(batch_next_state).detach()
        # q_next = self.target_net(batch_next_state).detach()
        # q_a = q_next_eval.argmax(dim=1)
        # q_a = torch.reshape(q_a, (-1, len(q_a)))
        # q_target = batch_reward + GAMMA * q_next.gather(1, q_a)

        print(q_eval,q_target)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
