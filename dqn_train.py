# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import copy
import platform,multiprocessing
import alpha_dqn_engine
import mcts as mcts
from ModelNet import Net2,VGG


ACTIONS = [0, 1, 2, 3]
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MEMORY_CAPACITY = 100000
SUCCESS_CAPACITY = 128
N_ACTIONS = 4
N_STATES = 16*4*4
TARGET_REPLACE_ITER = 100
BATCH_SIZE = 128
LR = 0.001

LOGGER = logging.getLogger(os.path.basename(__file__))

cuda_gpu = torch.cuda.is_available()

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs




class DqnDataSet(torch.utils.data.Dataset):
    def __init__(self,memory,memory_counter):
        self.memory = memory
        self.memory_counter = memory_counter

    def __getitem__(self, index):

        data = self.memory[index]

        state = data[0:N_STATES]

        ar = data[N_STATES:N_STATES + 4]

        return state,ar

    def __len__(self):
        date_num = MEMORY_CAPACITY if self.memory_counter >= MEMORY_CAPACITY else self.memory_counter
        return int(date_num)

class DQN(object):
    def __init__(self):
        if(os.path.exists('my_model.pkl')):
            LOGGER.info("load model file:%s"%('my_model.pkl'))
            self.eval_net = torch.load('my_model.pkl')
            self.load_from_model = True
        else:
            self.eval_net = Net2()
            self.load_from_model = False

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.large_memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY + 1, N_STATES + 4))  #保存当前状态，下一步状态，当前奖励，下一步动作,最后一行保存额外信息
        self.large_memory = np.zeros((10*MEMORY_CAPACITY + 1, N_STATES + 4))
        self.loadmemory()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def savenet(self):
        torch.save(self.eval_net,"my_model.pkl")

    def savememory(self):
        np.save("memory.npy",self.memory)
        np.save("large_memory.npy", self.large_memory)

    def loadmemory(self):
        if(os.path.exists('memory.npy')):
            self.memory = np.load('memory.npy')
            self.memory_counter = np.int(self.memory[-1][0])
            LOGGER.info("memory load memory.npy success,memory_counter:%d"%(self.memory_counter))

        if (os.path.exists('large_memory.npy')):
            self.large_memory = np.load('large_memory.npy')
            self.large_memory_counter = np.int(self.large_memory[-1][0])
            LOGGER.info("memory load large_memory.npy success,memory_counter:%d" % (self.memory_counter))

    def refresh_lr(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def choose_action(self, x):
        x = x.view(1,-1)
        #x = torch.unsqueeze(torch.FloatTensor(x), 0)
        #一开始全靠随机走，先不用神经网络预测，因为网络参数随机初始化，用网络预测很有可能一直原地踏步
        if np.random.uniform() < EPSILON and (self.memory_counter >= MEMORY_CAPACITY or self.load_from_model):  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)

        return action

    def store_transition(self, s, ar):
        transition = np.hstack((s, ar))
        # replace the old memory with new memory


        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        self.memory[-1][0] = self.memory_counter

        index = self.large_memory_counter % (10*MEMORY_CAPACITY)
        self.large_memory[index, :] = transition
        self.large_memory_counter += 1
        self.large_memory[-1][0] = self.large_memory_counter

    def clear_memory(self):
        self.memory_counter = 0
        self.memory[-1][0] = self.memory_counter

    def learn(self):
        date_num = MEMORY_CAPACITY if self.memory_counter >= MEMORY_CAPACITY else self.memory_counter

        sample_index = np.random.choice(date_num, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_ar = torch.FloatTensor(b_memory[:, N_STATES:N_STATES + 4])

        if (cuda_gpu):
            b_s = Variable(b_s).cuda()
            b_ar = Variable(b_ar).cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s)  # shape (batch, 1)

        train_correct_act = torch.true_divide((torch.argmax(q_eval, dim=1) == torch.argmax(b_ar, dim=1)).sum(),BATCH_SIZE) * 100

        loss = self.loss_func(q_eval, b_ar)

        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.eval_net.parameters(),1)
        self.optimizer.step()
        return loss.item(),train_correct_act
    def learn2(self):
        train_dataset = DqnDataSet(self.memory,self.memory_counter)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        running_loss = 0.0
        train_correct_act = 0
        train_total = 0

        for i, data in enumerate(train_loader, 0):
            b_s, b_ar = data
            if cuda_gpu:
                b_s = Variable(b_s).type(torch.FloatTensor).cuda()
                b_ar = Variable(b_ar).type(torch.FloatTensor).cuda()
            else:
                b_s = Variable(b_s).type(torch.FloatTensor)
                b_ar = Variable(b_ar).type(torch.FloatTensor)

            q_eval = self.eval_net(b_s)  # shape (batch, 1)

            train_correct_act += (torch.argmax(q_eval, dim=1) == torch.argmax(b_ar, dim=1)).sum()

            loss = F.mse_loss(q_eval, b_ar)

            self.optimizer.zero_grad()
            # loss.backward()
            loss.backward()
            # loss_value.backward()
            self.optimizer.step()


            running_loss += loss.item()
            # running_loss += (loss_value + loss_eval).item()

            train_total += b_ar.size(0)

        return torch.true_divide(running_loss, train_total)*BATCH_SIZE, torch.true_divide(100 * train_correct_act, train_total)

    def to_gpu(self):
        if(cuda_gpu):
            self.eval_net = self.eval_net.cuda()

    def to_cpu(self):
        if (cuda_gpu):
            self.eval_net = self.eval_net.cpu()


    def policy_value_fn(self, state):
        """
        input: board
        output: a list of (action, probability) tuples for each available
            action and the score of the board state
        """

        state = torch.FloatTensor(map_channel16(state))
        act_value = self.eval_net.forward(state)
        act_value = F.relu(act_value)
        act_value = act_value.flatten().data.numpy()
        act_value = 2**(act_value*16) - 1
        #act_value = 2**(act_value*16) - 1
        return act_value

def log2_shaping(s,divide=16):
    s = np.log2(1+s)/divide
    return s

def init_logger():
    fmt = '%(asctime)s:%(message)s'
    format_str = logging.Formatter(fmt)
    LOGGER.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = logging.FileHandler(os.path.basename(__file__) + '.log')
    th.setFormatter(format_str)
    LOGGER.addHandler(sh)
    LOGGER.addHandler(th)

def get_action(x,net):
    x = x.view(1, -1)
    if (cuda_gpu):
        x = x.cuda()

    actions_value = net.forward(x)
    if (cuda_gpu):
        actions_value = actions_value.cpu()
    action = torch.max(actions_value, 1)[1].data.numpy()
    action = action[0]

    return action

def map_channel16(map):

    tensor_map = np.zeros((16,4,4),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(map[i][j] == 0):
                level = 0
            else:
                level = int(np.log2(map[i][j]))

            tensor_map[level][i][j] = 1

    return tensor_map


def extend_data(data,ar):

    ret = []
    ret.append([data,ar])


    #旋转90度
    rot90_data = np.rot90(data,1)
    rot90_ar = copy.deepcopy(ar)
    rot90_ar = rot90_ar[[3,2,0,1]]
    ret.append([rot90_data,rot90_ar])

    #旋转180度
    rot180_data = np.rot90(data,2)
    rot180_ar = copy.deepcopy(ar)
    rot180_ar = rot180_ar[[1,0,3,2]]
    ret.append([rot180_data,rot180_ar])

    #旋转270度
    rot270_data = np.rot90(data,3)
    rot270_ar = copy.deepcopy(ar)
    rot270_ar = rot270_ar[[2,3,1,0]]
    ret.append([rot270_data,rot270_ar])

    return ret

def policy_value_fn_t(state):
    return np.zeros(4)

def main():
    init_logger()
    global EPSILON,LR
    dqn = DQN()

    dqn.savenet()

    LOGGER.info('\nCollecting experience...MEMORY_CAPACITY:%d'%(MEMORY_CAPACITY))

    i_episode = 0

    alpha_dqn_engine.GameEnv.init_cache()
    game_engine = alpha_dqn_engine.GameEnv(render=True)


    mcts_player = mcts.MCTSPlayer(dqn.policy_value_fn, n_playout=1000)

    LOGGER.info(dqn.eval_net)

    if(not os.path.exists("./episode_save")):
        os.mkdir("./episode_save")

    back_nums = 0


    dqn.eval_net.train()
    lear_lr = LR
    while True:
        game_engine.initGame()


        done = False

        step_counter = 0
        reward = 0

        state = copy.deepcopy(game_engine.map_score)

        mcts_player.reset_player(state)
        mcts_player.set_n_play_out(1000)


        running_loss = 0.0
        train_act_acc = 0
        predi_acc = 0
        predi_loss = 0
        train_num = 0
        while not done:
            #a = dqn.choose_action(s)

            predi_act = dqn.eval_net.forward(torch.FloatTensor(map_channel16(state).flatten()))

            move,ar = mcts_player.get_action(state)

            predi_acc += (torch.argmax(predi_act.flatten()) == np.argmax(ar.flatten())).sum().item()


            r,k = game_engine.move(move)

            r = log2_shaping(r,divide=1)

            state_ = copy.deepcopy(game_engine.map_score)
            mcts_info = mcts_player.get_mcts_info()
            mcts_player.update_with_move(move, k, state_)

            if (game_engine.game_over()):
                done = True

            after_score = game_engine.get_score()

            #ar = log2_shaping(ar,divide=16)
            ar = F.relu(torch.FloatTensor(ar)).data.numpy()
            ar = log2_shaping(ar, divide=16)
            move_datas = extend_data(state,ar)
            for data in move_datas:
                dqn.store_transition(map_channel16(data[0]).reshape(1,-1)[0],data[1])

            predi_loss += F.mse_loss(torch.FloatTensor(predi_act.flatten()), torch.FloatTensor(ar.flatten())).item()

            if(after_score[1] >= 2048):
                mcts_player.set_n_play_out(2000)

            #if (after_score[1] >= 4096):
             #   mcts_player.set_n_play_out(3000)
            if(step_counter>=3800 and after_score[1] <8192):
                mcts_player.set_n_play_out(10000)

            reward += r
            state = state_
            step_counter += 1
            #LOGGER.info("episode:%d step:%d reward:%d max_num:%d memory_counter:%d" % (i_episode, step_counter, reward, after_score[1],dqn.memory_counter))
            print("\r episode:%d step:%d reward:%f step_r:%f max_num:%d mcts_depth:%d mcts_maxcol:%d memory_counter:%d large_counter:%d lr:%.6f loss:%.6f act_acc:%.3f  predi_loss:%.6f predi_acc:%.3f" % (
                i_episode, step_counter, reward,
                reward/step_counter,after_score[1],mcts_info["max_depth"],
                mcts_info["max_col"],dqn.memory_counter,dqn.large_memory_counter,
                LR,
                0 if train_num == 0 else running_loss/train_num,
                0 if train_num == 0 else train_act_acc/train_num,
                predi_loss/step_counter,
                predi_acc*100/step_counter),end="")

        dqn.savememory()


        if dqn.memory_counter >= BATCH_SIZE:
            dqn.to_gpu()
            learn_num = 100 if dqn.memory_counter/BATCH_SIZE >=100 else int(dqn.memory_counter/BATCH_SIZE)
            lear_lr = LR
            for i in range(learn_num):
                rl, act_acc = dqn.learn()
                running_loss += rl
                train_act_acc += act_acc
                train_num += 1
            dqn.to_cpu()
            if(running_loss/train_num <= 0.008):
                LR = 0.0005
            else:
                LR = 0.001
            dqn.refresh_lr(LR)




        dqn.savenet()
        print("\n")
        LOGGER.info("episode:%d step:%d reward:%f step_r:%f max_num:%d mcts_depth:%d mcts_maxcol:%d memory_counter:%d large_counter:%d lr:%.6f loss:%.6f act_acc:%.3f  predi_loss:%.6f  predi_acc:%.3f" % (
            i_episode, step_counter, reward,
            reward/step_counter, after_score[1],mcts_info["max_depth"],
            mcts_info["max_col"],dqn.memory_counter,dqn.large_memory_counter,
            lear_lr,
            0 if train_num == 0 else running_loss/train_num,
            0 if train_num == 0 else train_act_acc/train_num ,
            predi_loss/step_counter,
            predi_acc*100/step_counter))
        i_episode += 1


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    main()

