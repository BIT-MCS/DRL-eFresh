from envs import Make_Env
from main_setting import Params
from methods.model import Model
from utils.base_utils import Util
import numpy as np
import csv
import os
import torch

params = Params()


def test(local_time, model_path, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    util = Util(device)
    torch.manual_seed(params.seed)
    torch.set_num_threads(4)
    worker_id = 0

    # ----------------make environment----------------------
    env = Make_Env(device, params.max_time_steps, local_time, worker_id)
    # -----------------load parameters----------------------
    obs_shape = env.observ_shape
    uav_num = params.uav_num

    # ---------------create local model---------------------
    local_ppo_model = Model(obs_shape, uav_num, device, trainable=False)
    local_ppo_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    local_ppo_model.to(device)

    episode_length = 0
    interact_time = 0
    # --------------define file writer-----------------------
    file_root_path = os.path.join(params.root_path, str(local_time) + '/' + str(+worker_id) + '/file')
    os.makedirs(file_root_path)

    print('Starting a new TEST iterations...')
    print("Log_dir:",file_root_path)

    reward_file = open(os.path.join(file_root_path, 'test_reward.csv'), 'w', newline='')
    reward_writer = csv.writer(reward_file)
    while True:
        if episode_length >= params.max_test_episode:
            print('testing over')
            break
        print('---------------in episode ', episode_length, '-----------------------')
        step = 0
        av_reward = 0
        cur_obs, uav_aoi, uav_snr, uav_tuse, uav_effort = env.reset()
        temporal_hidden_states = torch.zeros(params.temporal_hidden_size).unsqueeze(0)
        spatial_hidden_state=torch.zeros(params.spatial_hidden_size,8, 8).unsqueeze(0)
        masks = torch.ones(1)

        while step < params.max_time_steps:
            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                if params.use_rnn:
                    if params.use_spatial_att:
                        value, action, action_log_probs, temporal_hidden_states,spatial_hidden_state = local_ppo_model.act(cur_obs, uav_aoi,
                                                                                                       uav_snr, uav_tuse,
                                                                                                       uav_effort,
                                                                                                       temporal_hidden_states,
                                                                                                       masks,
                                                                                                      spatial_hidden_state)
                    else:
                        value, action, action_log_probs, temporal_hidden_states = local_ppo_model.act(cur_obs, uav_aoi,
                                                                                                      uav_snr, uav_tuse,
                                                                                                      uav_effort,
                                                                                                      temporal_hidden_states,
                                                                                                      masks)
                else:
                    value, action, action_log_probs = local_ppo_model.act(cur_obs, uav_aoi, uav_snr, uav_tuse,
                                                                          uav_effort)

                next_obs, reward, done, uav_aoi, uav_snr, uav_tuse, uav_effort = env.step(
                    util.to_numpy(action), current_step=step,
                    current_episode=episode_length)

            av_reward += reward

            step = step + 1
            cur_obs = next_obs

            if params.debug_mode is True:
                env.draw_path(step)

        if params.debug_mode is False:
            env.draw_path(episode_length)

        # ---------------- average reward -----------------------------
        av_reward = av_reward.cpu().mean().numpy()
        reward_writer.writerow([np.mean(av_reward)])
        episode_length = episode_length + 1

    reward_file.close()
