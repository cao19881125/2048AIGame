# -*- coding:utf-8 -*-
import numpy as np
import copy
import gc
#import time
#import alpha_dqn_engine_cpp as alpha_dqn_engine
import alpha_dqn_engine
from alpha_dqn_utils import timethis,timeblock

N_ACTIONS = 4
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.9


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def log2_shaping(s,divide=16):
    s = np.log2(1+s)/divide
    return s

class TreeNode(object):
    def __init__(self,parent,state):
        self.child_action = [None,None,None,None] #四个ActionNode
        self.parent = parent
        self.state = state

        temp_engine = alpha_dqn_engine.GameEnv()
        temp_engine.set_state(state)
        self.valid_action = temp_engine.valid_actioins()

    def add_action_node(self,action,reward):
        if(self.child_action[action] == None):
            self.child_action[action] = ActionNode(self,reward)

    def get_action_reward(self):
        return np.array([action_node.reward if action_node else 0 for action_node in self.child_action])

    def select(self):
        valid_actions = self.valid_action
        if(np.random.uniform() < EPSILON):
            action_rewards = self.get_action_reward()

            if(action_rewards.var() == 0):
                #return np.random.randint(0, N_ACTIONS)
                return np.random.choice(np.argwhere(valid_actions==True).flatten())

            valid_value = action_rewards[np.argwhere(valid_actions==True).flatten()]
            arg_value = np.argwhere(valid_actions==True).flatten()
            return arg_value[np.argmax(valid_value)]
        else:
            return np.random.choice(np.argwhere(valid_actions==True).flatten())

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self.child_action == [None,None,None,None]

    def update_recursive(self):
        if(self.parent):
            self.parent.update_recursive()


class ActionNode(object):
    def __init__(self,parent,reward):
        self.parent = parent
        self.child_tree = {} #action下对应的TreeNode  key是新走的点的位置*10 + 随机初始化的数字 位置范围0-15
        self.reward = reward
        self.move_r = 0


    def add_tree_node(self,key,state):
        if(key not in self.child_tree):
            self.child_tree[key] = TreeNode(self,state)

    def calc_reward(self):
        children_max = 0
        for tree_node in self.child_tree.values():
            children_max += np.max(tree_node.get_action_reward())

        children_max = children_max/len(self.child_tree) if len(self.child_tree)>0 else np.float(0)

        target = self.move_r + GAMMA*children_max
        #self.reward = self.reward + ALPHA*(target - self.reward)
        self.reward = target

    def update_recursive(self):
        self.calc_reward()
        if(self.parent):
            self.parent.update_recursive()

class MCTS(object):
    def __init__(self, policy_value_fn,n_playout=10000,n_max_depth = 30):
        self.root = None
        self._policy = policy_value_fn
        self._n_playout = n_playout
        self.n_max_depth = n_max_depth
        self.max_col = 0
        self.depth_base = 0
        self.max_depth = 0

    def set_n_playout(self,n_playout):
        self._n_playout = n_playout

    def _playout2(self, state):
        game_engine = alpha_dqn_engine.GameEnv()
        game_engine.set_state(state)
        node = self.root

        done = False

        depth = 0

        while True:
            if node.is_leaf():
                break
            a = node.select()
            r, key = game_engine.move(a)
            node.child_action[a].move_r = r
            state = copy.deepcopy(game_engine.map_score)
            node.child_action[a].add_tree_node(key, state)
            node = node.child_action[a].child_tree[key]
            depth += 1

        if (game_engine.game_over()):
            done = True

        game_engine.destroy()

        if(not done):
            action_rewards = self._policy(state)
            for i in range(len(action_rewards)):
                node.add_action_node(i, action_rewards[i])

        node.update_recursive()
        if (np.max(node.state) > self.max_col):
            self.max_col = np.max(node.state)

        if (depth + self.depth_base > self.max_depth):
            self.max_depth = depth + self.depth_base

    def _playout(self, state):

        game_engine = alpha_dqn_engine.GameEnv()
        game_engine.set_state(state)
        node = self.root

        done = False

        depth = 0

        while not done and depth <= self.n_max_depth:
            if node.is_leaf():
                action_rewards = self._policy(state)
                for i in range(len(action_rewards)):
                    node.add_action_node(i, action_rewards[i])

            a = node.select()

            r,key = game_engine.move(a)
            #r = log2_shaping(r,divide=1)

            #node.child_action[a].calc_reward(r)
            node.child_action[a].move_r = r

            if(key != None):
                state = copy.deepcopy(game_engine.map_score)
                node.child_action[a].add_tree_node(key,state)

                node = node.child_action[a].child_tree[key]
                depth += 1

            if(game_engine.game_over()):
                done = True

        game_engine.destroy()

        node.update_recursive()
        if(np.max(node.state) > self.max_col):
            self.max_col = np.max(node.state)

        if(depth + self.depth_base > self.max_depth):
            self.max_depth = depth + self.depth_base


    def get_mode_values(self, state):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout2(state_copy)

        return self.root.get_action_reward()


    def update_with_move(self, last_move,key,state):

        if(last_move == -10):
            self.depth_base = 0


        if(last_move not in range(4)):
            self.root = TreeNode(None,state)
            return

        if key in self.root.child_action[last_move].child_tree:
            self.root = self.root.child_action[last_move].child_tree[key]
            self.root.parent = None
        else:
            self.root = TreeNode(None,state)
        self.depth_base += 1
        self.max_col = 0
        self.max_depth = 0

        gc.collect()

class MCTSPlayer(object):
    def __init__(self, policy_value_function, n_playout=2000):
        self.mcts = MCTS(policy_value_function, n_playout)

    def reset_player(self,state):
        self.mcts.update_with_move(-10,None,state)

    def update_with_move(self,move,key,state):
        self.mcts.update_with_move(move,key,state)

    def get_mcts_info(self):
        info = {"max_col":self.mcts.max_col,"max_depth":self.mcts.max_depth}
        return info

    def set_n_play_out(self,n_playout):
        self.mcts.set_n_playout(n_playout)

    def get_action(self, state,return_value = True):

        # probs = self.mcts.get_move_probs(state)
        move_values = self.mcts.get_mode_values(state)

        game_engine = alpha_dqn_engine.GameEnv()
        game_engine.set_state(state)
        valid_action = game_engine.valid_actioins()
        game_engine.destroy()

        move_values[np.argwhere(valid_action==False).flatten()] = 0
        arg_value = np.argwhere(valid_action==True).flatten()
        move_value = move_values[arg_value]

        # probs = softmax(move_value)
        #
        # move = np.random.choice(
        #     arg_value,
        #     p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
        # )
        move = arg_value[np.argmax(move_value)]

        if(return_value):
            return move,move_values
        else:
            return move

def main():

    def temp_function(state):
        return np.array([0,0,0,0],dtype=np.float)

    mcts = MCTSPlayer(temp_function,n_playout=200)

    game_engine = alpha_dqn_engine.GameEnv(render=True)

    i_episode = 0

    while True:
        game_engine.initGame()

        end = False

        step_counter = 0
        reward = 0

        state = copy.deepcopy(game_engine.map_score)
        mcts.reset_player(state)

        #import pdb;pdb.set_trace()
        while not end:

            move = mcts.get_action(state,return_value=False)

            r,k = game_engine.move(move)
            state_ = copy.deepcopy(game_engine.map_score)

            mcts.update_with_move(move,k,state_)

            if (game_engine.game_over()):
                end = True

            after_score = game_engine.get_score()
            reward += r

            step_counter += 1
            state = state_
            print("episode:%d step:%d reward:%d max_num:%d" % (i_episode, step_counter, reward, after_score[1]))
            game_engine.print_map()

        i_episode += 1

        print("episode:%d step:%d reward:%d max_num:%d" % (i_episode, step_counter, reward, after_score[1]))

if __name__ == "__main__":
    main()