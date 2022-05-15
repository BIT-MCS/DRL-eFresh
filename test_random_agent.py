from main_setting import Params
from environment.uav_collection.env import Env
from environment.uav_collection.log import Log
from environment.uav_collection.env_setting import Setting
import numpy as np
import time
import os

params = Params()
sg = Setting()
if params.trainable is True:
    print("please set params.trainable is False!!!")
    exit(0)


local_time = str(time.strftime("%Y %m-%d %H-%M-%S", time.localtime()))
env = Env(Log(0, params.max_time_steps, params.root_path, local_time))

# log.log(ARGUMENTS)
start_env = env.reset()

print(env.action_space)
print('Starting a new TEST iterations...')
print('Log_dir:', env.log_dir)

iteration = 0
time_step = 0

while iteration < params.max_test_episode:

    action_n = np.array([[np.random.uniform(low=-1,high=1)*sg.V['MAP_X'],
                          np.random.uniform(low=-1,high=1)*sg.V['MAP_Y'],
                          np.random.uniform(low=0,high=1)]
                         for _ in range(env.sg.V["NUM_UAV"])])

    ob, r, d, _, _, _,_ = env.step(action=action_n, current_step=time_step)

    time_step += 1
    terminal = (time_step >= params.max_time_steps)
    # env.log.draw_path(env, iteration)

    if d or terminal:
        print('\n%d th episode:\n' % iteration)

        env.log.draw_path(env, iteration)
        iteration += 1
        time_step = 0
        obs_n = env.reset()
