# -*- coding:utf-8 -*-
import numpy as np
import copy
import platform,multiprocessing
from multiprocessing import Process,Queue
import gameui
# 游戏棋盘 4*4大小 左上角坐标是(0,0)，X轴向右+1，Y轴向下+1，如下
# (0,1) (0,1) (0,2) (0,3)
# (1,1) (1,1) (1,2) (1,3)
# (2,1) (2,1) (2,2) (2,3)
# (3,1) (3,1) (3,2) (3,3)

class GameEnv(object):

    VALID_CACHE = None

    def __init__(self,render = False):
        #np.random.seed(1)
        self.invalied_num = 0
        self.render = render

        if(self.render):
            self.render_process = SubProcess()
            self.render_process.start()

        self.initGame()


    def initGame(self):
        self.map_score = np.zeros((4, 4), dtype=np.int)

        # init_score_array = np.random.randint(0,4,size=(2,2))
        #
        # if((init_score_array[0] == init_score_array[1]).all()):
        #     # 如果random出来的值相等，就重新random一次
        #     init_score_array = np.random.randint(0, 4, size=(2, 2))
        #
        # self.map_score[init_score_array[0][0]][init_score_array[0][1]] = 2
        # self.map_score[init_score_array[1][0]][init_score_array[1][1]] = 2
        self.invalied_num = 0
        self.create_new_element()
        self.create_new_element()
        if(self.render):
            self.render_process.put(self.map_score.tolist())

    @classmethod
    def init_cache(cls):
        GameEnv.VALID_CACHE = np.zeros((65536,2))
        print("start init_cache")
        MAX_NUM = 16
        temp_engine = GameEnv()
        for a in range(MAX_NUM):
            for b in range(MAX_NUM):
                for c in range(MAX_NUM):
                    for d in range(MAX_NUM):
                        key = (a<<12) + (b<<8) + (c<<4) + d
                        key_ret = [False,False]
                        v_d = np.array([a, b, c, d])
                        merged_array,_ = temp_engine.handle_array(copy.deepcopy(v_d))
                        if (not (merged_array == v_d).all()):
                            key_ret[0] = True

                        merged_array,_ = temp_engine.handle_array(copy.deepcopy(v_d),reverse=True)
                        if (not (merged_array == v_d).all()):
                            key_ret[1] = True
                        GameEnv.VALID_CACHE[key] = key_ret


        print("finish init_cache")

    def create_new_element(self):

        filter_array = []
        for i in range(4):
            for j in range(4):
                if(self.map_score[i][j] == 0):
                    filter_array.append((i,j))

        if(len(filter_array) == 0):
            return

        init_score_array = np.random.randint(0, len(filter_array))
        x = filter_array[init_score_array][0]
        y = filter_array[init_score_array][1]

        if(np.random.uniform() >= 0.9):
            new_num = 4
        else:
            new_num = 2
        #new_num = 2

        self.map_score[x][y] = new_num

        return (x*4 + y)*10 + new_num

    def game_over(self):

        if(self.invalied_num >=5):
            return True

        for i in range(4):
            array = self.map_score[i]

            merged_array,_ = self.merge_array(array)
            if (len(merged_array) < 4):
                return False

            array = self.map_score[:,i]
            merged_array,_ = self.merge_array(array)
            if (len(merged_array) < 4):
                return False

        return True

    def valid_actioins2(self):
        ret = np.zeros(4,dtype=np.bool)

        for i in range(4):
            array = copy.deepcopy(self.map_score[i])
            merged_array,_ = self.handle_array(copy.deepcopy(array))
            if(not (merged_array == array).all()):
                ret[0] = True
            merged_array, _ = self.handle_array(copy.deepcopy(array),reverse=True)
            if (not (merged_array == array).all()):
                ret[1] = True

            array = copy.deepcopy(self.map_score[:, i])
            merged_array, _ = self.handle_array(copy.deepcopy(array))
            if (not (merged_array == array).all()):
                ret[2] = True
            merged_array, _ = self.handle_array(copy.deepcopy(array), reverse=True)
            if (not (merged_array == array).all()):
                ret[3] = True

            if(ret.all()):
                return ret
        return ret

    def valid_actioins(self):
        ret = np.zeros(4,dtype=np.bool)

        for i in range(4):
            array = copy.deepcopy(self.map_score[i])
            array = array + (array == 0)
            array = np.log2(array).astype(int).tolist()
            key = (array[0]<<12) + (array[1]<<8) + (array[2]<<4) + array[3]
            ret[0:2] += GameEnv.VALID_CACHE[key].astype(bool)

            array = copy.deepcopy(self.map_score[:, i])
            array = array + (array == 0)
            array = np.log2(array).astype(int).tolist()
            key = (array[0]<<12) + (array[1]<<8) + (array[2]<<4) + array[3]
            ret[2:4] += GameEnv.VALID_CACHE[key].astype(bool)

            if(ret.all()):
                return ret
        return ret


    def set_state(self,state):
        self.map_score = np.array(state)
        if (self.render):
            self.render_process.put(self.map_score.tolist())

    def get_value(self):
        value = 0
        for i in range(4):
            for j in range(4):
                value += int(self.map_score[i][j] / 16)

        return value

    def merge_array(self,array,reverse = False):
        # 第一步 去除中间的0
        # 第二步，合并相邻的相等的
        reward = 0
        filter_array = []
        if(reverse):
            array = array[::-1]
        just_merged = False
        for element in array:
            if (element > 0):
                if (len(filter_array) == 0 or filter_array[len(filter_array) - 1] != element or just_merged):
                    filter_array.append(element)
                    just_merged = False
                else:
                    reward += filter_array[len(filter_array) - 1]
                    filter_array[len(filter_array) - 1] *= 2
                    just_merged = True

        filter_array = filter_array if not reverse else filter_array[::-1]

        # if(reward == 0):
        #     return filter_array,reward
        # else:
        #     filter_array,rt = self.merge_array(filter_array,reverse=reverse)
        #     reward += rt

        return filter_array,reward



    def handle_array(self,array, reverse = False):

        filter_array,reward = self.merge_array(array,reverse=reverse)

        if(not reverse):
            array[0:len(filter_array)] = filter_array
            array[len(filter_array):len(array)] = np.zeros(len(array) - len(filter_array))
        else:
            array[len(array) - len(filter_array):len(array)] = filter_array
            array[0:len(array) - len(filter_array)] = np.zeros(len(array) - len(filter_array))

        return array,reward



    #action: 0-左 1-右 2-上 3-下
    def move(self,action,create_new_element=True):

        pre_map = np.array(self.map_score)
        reward = 0
        for i in range(4):
            if (action in [0, 2]):
                reverse = False
            else:
                reverse = True

            if(action in [0,1]):
                array = self.map_score[i]
            else:
                array = self.map_score[:,i]

            array,rt = self.handle_array(array,reverse)

            if(action in [0,1]):
                self.map_score[i] = array
            else:
                self.map_score[:,i] = array

            reward += rt

        if(self.game_over()):
            reward = reward

        if(not (pre_map == self.map_score).all()):
            if(create_new_element):
                new_num = self.create_new_element()
            else:
                new_num = None
            self.invalied_num = 0
        else:
            new_num = None
            reward = 0
            self.invalied_num += 1

        if(self.render):
            self.render_process.put(self.map_score.tolist())

        return reward,new_num

    def print_map(self):
        for i in range(4):
            str_tmp = ""
            for j in range(4):
                str_tmp += (" " + str(self.map_score[i][j].item())).ljust(5)
            print(str_tmp)

    # return:(地图所有分值,最大块分值)
    def get_score(self):
        all_score = 0
        max_score = 0
        not_zero_num = 0
        for i in range(4):
            for j in range(4):
                tmp_score = self.map_score[i][j]
                not_zero_num = not_zero_num + 1 if tmp_score > 0 else not_zero_num
                all_score += tmp_score
                if(tmp_score > max_score):
                    max_score = tmp_score

        return all_score,max_score,not_zero_num

    def destroy(self):
        pass



class SubProcess(object):
    def __init__(self):
        self.q = Queue()
        self.p = Process(target=self.f)

    def f(self):
        gameui.start_render(self.q)

    def put(self,item):
        self.q.put(item)

    def start(self):
        self.p.start()

if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    GameEnv.init_cache()
    g = GameEnv(render=True)
    g.map_score[0] = np.array([2, 16, 8, 32])
    g.map_score[1] = np.array([16, 4, 4, 8])
    g.map_score[2] = np.array([4, 32, 64, 128])
    g.map_score[3] = np.array([8, 4, 8, 256])
    g.print_map()

    print(g.valid_actioins())

    # import time
    # t1 = time.time()
    # v1 = g.valid_actioins()
    # t2 = time.time()
    # v2 = g.valid_actioins2()
    # t3 = time.time()
    # print(v1,t2 - t1)
    # print(v2,t3 - t2)



    while True:
        getc = input("请输入：")
        reward = g.move(int(getc))
        print("reward",reward)
        if(g.game_over()):
            print("游戏结束")
            g.print_map()
            break
        g.print_map()

