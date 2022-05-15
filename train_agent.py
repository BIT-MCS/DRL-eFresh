from envs import Make_Env  # TODO:interfere
from main_setting import Params
from storage import RolloutStorage
from methods.model import Model
from utils.base_utils import Util, plot_line
import os
import torch
import pandas as pd

params = Params()


def train(worker_id, traffic_light, counter, shared_model, shared_grad_buffers, local_time, son_process_counter,
          device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.cuda_device)

    util = Util(device)
    torch.manual_seed(params.seed + worker_id)
    torch.set_num_threads(4)
    # ----------------make environment----------------------
    env = Make_Env(device, params.max_time_steps, local_time, worker_id)
    # -----------------load parameters----------------------
    obs_shape = env.observ_shape
    uav_num = params.uav_num
    clip = params.clip
    use_gae = params.use_gae
    ent_coeff = params.ent_coeff
    value_coeff = params.value_coeff
    clip_coeff = params.clip_coeff
    gamma = params.gamma
    gae_lambda = params.gae_lambda
    use_adv_norm = params.use_adv_norm
    # --------------create name---------------------------
    method_name = 'FixTauPpo'
    if use_adv_norm:
        method_name += 'AdvNorm'
    # if worker_id == 0:
    #     # visdom cannot work in ssh!!!
    #     vis = visdom.Visdom(env=method_name)

    # -----------------create storage---------------------
    rollout = RolloutStorage(params.max_time_steps, params.mini_batch_num, obs_shape, uav_num)
    rollout.to(device)

    # ---------------create local model---------------------
    local_ppo_model = Model(obs_shape, uav_num, device)
    local_ppo_model.to(device)

    episode_length = 0
    interact_time = 0
    # --------------define file writer-----------------------
    root_path = os.path.join(params.root_path, str(local_time))

    if worker_id ==0:
        file_root_path = os.path.join(params.root_path, str(local_time) + '/' + str(worker_id) + '/file')
        os.makedirs(file_root_path)

    # loss_file = open(os.path.join(file_root_path, 'loss.csv'), 'w', newline='')
    # loss_writer = csv.writer(loss_file)
    # reward_file = open(os.path.join(file_root_path, 'reward.csv'), 'w', newline='')
    # reward_writer = csv.writer(reward_file)
    # action_file = open(os.path.join(file_root_path, 'action.csv'), 'w', newline='')
    # action_writer = csv.writer(action_file)

    av_reward_list = []
    av_value_loss_list = []
    av_policy_loss_list = []
    av_ent_loss_list = []

    # load local model parameters
    local_ppo_model.load_state_dict(shared_model.state_dict())
    done = False
    init_tau = 1.
    end_tau = 0.1
    tau = init_tau
    while True:
        if episode_length >= params.max_train_episode:
            print('training over')
            break
        if worker_id == 0:
            print('---------------in episode ', episode_length, '-----------------------')

        tau -= (init_tau - end_tau) / params.max_train_episode

        if worker_id == 0:
            # print('clip', clip)
            print('tau', tau)

        step = 0
        av_reward = 0
        cur_obs, uav_aoi, uav_snr, uav_tuse, uav_effort = env.reset()
        rollout.after_update(cur_obs, uav_aoi, uav_snr, uav_tuse, uav_effort)
        # action_writer.writerow(['episode', episode_length])

        while step < params.max_time_steps:
            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            returns = torch.zeros(1, 1)
            with torch.no_grad():
                if params.use_rnn:
                    if params.use_spatial_att:
                        value, action, action_log_probs, temporal_hidden_states, spatial_hidden_states = local_ppo_model.act(
                            rollout.obs[step].unsqueeze(0), rollout.uav_aoi[step].unsqueeze(0),
                            rollout.uav_snr[step].unsqueeze(0), rollout.uav_tuse[step].unsqueeze(0),
                            rollout.uav_effort[step].unsqueeze(0), rollout.temporal_hidden_states[step].unsqueeze(0),
                            rollout.masks[step], rollout.spatial_hidden_states[step].unsqueeze(0))
                    else:
                        value, action, action_log_probs, temporal_hidden_states = local_ppo_model.act(
                            rollout.obs[step].unsqueeze(0), rollout.uav_aoi[step].unsqueeze(0),
                            rollout.uav_snr[step].unsqueeze(0), rollout.uav_tuse[step].unsqueeze(0),
                            rollout.uav_effort[step].unsqueeze(0), rollout.temporal_hidden_states[step].unsqueeze(0),
                            rollout.masks[step])

                else:
                    value, action, action_log_probs = local_ppo_model.act(rollout.obs[step].unsqueeze(0),
                                                                          rollout.uav_aoi[step].unsqueeze(0),
                                                                          rollout.uav_snr[step].unsqueeze(0),
                                                                          rollout.uav_tuse[step].unsqueeze(0),
                                                                          rollout.uav_effort[step].unsqueeze(0))

                next_obs, reward, done, uav_aoi, uav_snr, uav_tuse, uav_effort = env.step(
                    util.to_numpy(action),
                    current_step=step,
                    current_episode=episode_length,
                    current_worker=worker_id)

            # action_writer.writerow([step, action.squeeze().cpu().numpy()])
            av_reward += reward
            # ---------judge if game over --------------------
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done])
            # ----------add to memory ---------------------------
            if params.use_rnn:
                if params.use_spatial_att:
                    rollout.insert(next_obs.detach(), uav_aoi.detach(), uav_snr.detach(), uav_tuse.detach(),
                                   uav_effort.detach(), action.detach(), action_log_probs.detach(), value.detach(),
                                   reward.detach(), masks.detach(), returns, temporal_hidden_states.detach(),
                                   spatial_hidden_states.detach())
                else:
                    rollout.insert(next_obs.detach(), uav_aoi.detach(), uav_snr.detach(), uav_tuse.detach(),
                                   uav_effort.detach(), action.detach(), action_log_probs.detach(), value.detach(),
                                   reward.detach(), masks.detach(), returns, temporal_hidden_states.detach())

            else:
                rollout.insert(next_obs.detach(), uav_aoi.detach(), uav_snr.detach(), uav_tuse.detach(),
                               uav_effort.detach(), action.detach(), action_log_probs.detach(), value.detach(),
                               reward.detach(), masks.detach(), returns)
                # if episode_length % 10 == 0 and rank == 0:
                #     env.render()
            step = step + 1

        # --------------update---------------------------
        done = done[0]
        with torch.no_grad():
            if done:
                next_value = torch.zeros(1)
            else:
                if params.use_rnn:
                    if params.use_spatial_att:
                        next_value = local_ppo_model.get_value(rollout.obs[-1:], rollout.uav_aoi[-1:],
                                                               rollout.uav_snr[-1:],
                                                               rollout.uav_tuse[-1:],
                                                               rollout.uav_effort[-1:],
                                                               rollout.temporal_hidden_states[-1:],
                                                               rollout.masks[-1:], rollout.spatial_hidden_states[-1:])
                    else:
                        next_value = local_ppo_model.get_value(rollout.obs[-1:], rollout.uav_aoi[-1:],
                                                               rollout.uav_snr[-1:],
                                                               rollout.uav_tuse[-1:],
                                                               rollout.uav_effort[-1:],
                                                               rollout.temporal_hidden_states[-1:],
                                                               rollout.masks[-1:])
                else:
                    next_value = local_ppo_model.get_value(rollout.obs[-1:], rollout.uav_aoi[-1:],
                                                           rollout.uav_snr[-1:],
                                                           rollout.uav_tuse[-1:],
                                                           rollout.uav_effort[-1:])

        rollout.compute_returns(next_value.detach(), use_gae, gamma, gae_lambda)
        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        if use_adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        av_value_loss = 0
        av_policy_loss = 0
        av_ent_loss = 0
        loss_cnt = 0
        for _ in range(params.ppo_epoch):
            if params.use_rnn:
                if params.use_spatial_att:
                    data_generator = rollout.spatial_att_feed_forward_generator(advantages)
                else:
                    data_generator = rollout.rnn_feed_forward_generator(advantages)

            else:
                data_generator = rollout.feed_forward_generator(advantages)
            for samples in data_generator:
                signal_init = traffic_light.get()

                if params.use_rnn:  # TODO
                    if params.use_spatial_att:
                        obs_batch, uav_aoi_batch, uav_snr_batch, uav_tuse_batch, uav_effort_batch, action_batch, \
                        old_values, return_batch, masks_batch, old_action_log_probs, advantages_batch, \
                        temporal_hidden_states, spatial_hidden_states = samples

                        cur_values, cur_action_log_probs, dist_entropy = local_ppo_model.evaluate_actions(obs_batch,
                                                                                                          uav_aoi_batch,
                                                                                                          uav_snr_batch,
                                                                                                          uav_tuse_batch,
                                                                                                          uav_effort_batch,
                                                                                                          action_batch,
                                                                                                          temporal_hidden_states,
                                                                                                          masks_batch,
                                                                                                          spatial_hidden_states)
                    else:
                        obs_batch, uav_aoi_batch, uav_snr_batch, uav_tuse_batch, uav_effort_batch, action_batch, \
                        old_values, return_batch, masks_batch, old_action_log_probs, advantages_batch, \
                        temporal_hidden_states = samples

                        cur_values, cur_action_log_probs, dist_entropy = local_ppo_model.evaluate_actions(obs_batch,
                                                                                                          uav_aoi_batch,
                                                                                                          uav_snr_batch,
                                                                                                          uav_tuse_batch,
                                                                                                          uav_effort_batch,
                                                                                                          action_batch,
                                                                                                          temporal_hidden_states,
                                                                                                          masks_batch)
                else:
                    obs_batch, uav_aoi_batch, uav_snr_batch, uav_tuse_batch, uav_effort_batch, action_batch, \
                    old_values, return_batch, masks_batch, old_action_log_probs, advantages_batch = samples

                    cur_values, cur_action_log_probs, dist_entropy = local_ppo_model.evaluate_actions(obs_batch,
                                                                                                      uav_aoi_batch,
                                                                                                      uav_snr_batch,
                                                                                                      uav_tuse_batch,
                                                                                                      uav_effort_batch,
                                                                                                      action_batch)

                # ----------use ppo clip to compute loss------------------------
                ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages_batch
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_values + (cur_values - old_values).clamp(-clip, clip)
                value_losses = (cur_values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                value_loss = value_loss * value_coeff
                action_loss = action_loss * clip_coeff
                ent_loss = dist_entropy * ent_coeff

                total_loss = value_loss + action_loss - ent_loss
                local_ppo_model.zero_grad()

                total_loss.backward()

                # ----------------- add model gradient ----------------------------
                shared_grad_buffers.add_gradient(local_ppo_model)
                av_value_loss += value_loss
                av_policy_loss += action_loss
                av_ent_loss += ent_loss
                loss_cnt += 1

                # ---------wait for update----------------------
                counter.increment()
                while traffic_light.get() == signal_init:
                    pass
                # update local_ppo_model
                local_ppo_model.load_state_dict(shared_model.state_dict())

        av_value_loss /= loss_cnt
        av_policy_loss /= loss_cnt
        av_ent_loss /= loss_cnt
        # --------------- draw & log -----------------------------
        if worker_id == 0:
            env.draw_path(episode_length)

        # ---------------- average reward -----------------------------
        av_reward_np = av_reward.cpu().mean().numpy()
        # reward_writer.writerow([episode_length,av_reward_np])

        if worker_id == 0:
            av_reward_list.append(av_reward_np)
            av_value_loss_list.append(av_value_loss.item())
            av_policy_loss_list.append(av_policy_loss.item())
            av_ent_loss_list.append(av_ent_loss.item())

            plot_line("Accumulated reward", av_reward_list, root_path)
            plot_line("Average critic loss", av_value_loss_list, root_path)
            plot_line("Average actor loss", av_policy_loss_list, root_path)
            plot_line("Average entropy loss", av_ent_loss_list, root_path)

            ListCSV=pd.DataFrame(columns=["reward"],data=av_reward_list)
            ListCSV.to_csv(file_root_path+"/reward_list.csv",encoding='utf-8')

            print('average reward: ', av_reward_list[-1])
            print('value_loss: ', av_value_loss_list[-1], 'policy_loss:', av_policy_loss_list[-1], 'entropy loss:',
                  av_ent_loss_list[-1])

        # loss_writer.writerow(
        #     [episode_length, av_value_loss, av_policy_loss, av_ent_loss])

        if worker_id == 0 and (episode_length+1) % params.save_interval == 0:
            model_root_path = os.path.join(params.root_path, str(local_time) + '/ckpt/')
            os.makedirs(model_root_path, exist_ok=True)
            model_root_path = os.path.join(model_root_path, 'model_%d.pt' % episode_length)
            torch.save(local_ppo_model.state_dict(), model_root_path)

        episode_length = episode_length + 1
    # loss_file.close()
    # reward_file.close()

    son_process_counter.increment()
