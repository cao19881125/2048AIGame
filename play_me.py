
import alpha_dqn_engine
from pynput import keyboard
from pynput.keyboard import Key
import platform,multiprocessing
import numpy as np
import time
import copy

game_over = False
reward = 0

def log2_shaping(s,divide=16):
    s = np.log2(1+s)/divide
    return s

def log2_shaping2(s,divide=16):
    s = np.log2(s + (s == 0))/divide
    return s

def main():
    alpha_dqn_engine.GameEnv.init_cache()

    g = alpha_dqn_engine.GameEnv(render=True)
    #g.print_map()


    g.print_map()
    def handler_key(key):
        global game_over,reward
        if (key == Key.up):
            action = 2
        elif (key == Key.down):
            action = 3
        elif (key == Key.left):
            action = 0
        elif (key == Key.right):
            action = 1
        else:
            return

        if(action in range(4)):
            valid_actions = g.valid_actioins()
            if(not valid_actions[action]):
                print("invalid action",action)
                return

            r,_ = g.move(action)
            reward += r

        print("")
        g.print_map()
        print("reward ",reward)
        if (g.game_over()):
            print("game over")
            game_over = True

    listener = keyboard.Listener(on_release=handler_key)
    listener.start()

    while not game_over:
        time.sleep(1)


    # while not game_over:
    #     print("game_over:",game_over)
    #     with keyboard.Listener(on_release=handler_key) as listener:
    #         listener.join()

if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    main()