import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from main_setting import Params

params = Params()


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, mini_batch_num, obs_shape, uav_num,
                 temporal_hidden_state_size=params.temporal_hidden_size,
                 spatial_hidden_state_size=params.spatial_hidden_size):
        self.mini_batch_num = mini_batch_num
        self.obs = torch.zeros(num_steps + 1, *obs_shape)
        self.uav_aoi = torch.zeros(num_steps + 1, params.uav_num)
        self.uav_snr = torch.zeros(num_steps + 1, params.uav_num)
        self.uav_tuse = torch.zeros(num_steps + 1, params.uav_num)
        self.uav_effort = torch.zeros(num_steps + 1, params.uav_num)
        # self.next_obs = torch.zeros(num_steps + 1, *obs_shape)
        self.rewards = torch.zeros(num_steps, 1)

        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        self.action_dia = torch.zeros(num_steps, int(uav_num * params.uav_action_dim))
        self.masks = torch.ones(num_steps + 1, 1)
        self.num_steps = num_steps
        self.step = 0
        if params.use_rnn:
            self.temporal_hidden_states = torch.zeros(num_steps + 1, temporal_hidden_state_size)
            self.temporal_hidden_state_size = temporal_hidden_state_size
            if params.use_spatial_att:
                self.spatial_hidden_states=torch.zeros(num_steps + 1, spatial_hidden_state_size,8, 8)
                self.spatial_hidden_state_size = spatial_hidden_state_size

    def to(self, device):
        self.obs = self.obs.to(device)
        self.uav_aoi = self.uav_aoi.to(device)
        self.uav_snr = self.uav_snr.to(device)
        self.uav_tuse = self.uav_tuse.to(device)
        self.uav_effort = self.uav_effort.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_dia = self.action_dia.to(device)
        self.masks = self.masks.to(device)
        if params.use_rnn:
            self.temporal_hidden_states = self.temporal_hidden_states.to(device)
            if params.use_spatial_att:
                self.spatial_hidden_states=self.spatial_hidden_states.to(device)

    def insert(self, obs, uav_aoi, uav_snr, uav_tuse, uav_effort, actions, action_log_probs, value_preds, rewards,
               masks, returns, temporal_hidden_states=None,spatial_hidden_states=None):
        self.action_dia[self.step].copy_(actions.squeeze())
        self.action_log_probs[self.step].copy_(action_log_probs.squeeze())
        self.value_preds[self.step].copy_(value_preds.squeeze())
        self.rewards[self.step].copy_(rewards.squeeze())
        self.obs[self.step + 1].copy_(obs.squeeze())
        self.masks[self.step + 1].copy_(masks.squeeze())
        self.uav_aoi[self.step + 1].copy_(uav_aoi.squeeze())
        self.uav_snr[self.step + 1].copy_(uav_snr.squeeze())
        self.uav_tuse[self.step + 1].copy_(uav_tuse.squeeze())
        self.uav_effort[self.step + 1].copy_(uav_effort.squeeze())

        # self.next_obs[self.step].copy_(next_obs.squeeze())
        self.returns[self.step].copy_(returns.squeeze())

        if params.use_rnn:
            self.temporal_hidden_states[self.step + 1].copy_(temporal_hidden_states.squeeze())
            if params.use_spatial_att:
                self.spatial_hidden_states[self.step+1].copy_(spatial_hidden_states.squeeze())

        self.step = self.step + 1
        # self.step = (self.step + 1) % self.num_steps

    def after_update(self, obs, uav_aoi, uav_snr, uav_tuse, uav_effort):
        self.step = 0
        self.obs[0].copy_(obs.squeeze())
        self.uav_aoi[0].copy_(uav_aoi.squeeze())
        self.uav_snr[0].copy_(uav_snr.squeeze())
        self.uav_tuse[0].copy_(uav_tuse.squeeze())
        self.uav_effort[0].copy_(uav_effort.squeeze())

        self.masks[0].copy_(torch.ones(1))
        if params.use_rnn:
            self.temporal_hidden_states[0].copy_(self.temporal_hidden_states[-1])
            if params.use_spatial_att:
                self.spatial_hidden_states[0].copy_(self.spatial_hidden_states[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages):
        mini_batch_size = self.num_steps // self.mini_batch_num  # 500/4
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps)), mini_batch_size, drop_last=False)
        for indices in sampler:
            next_indices = [indice + 1 for indice in indices]
            # [index, dim]
            obs_batch = self.obs[indices]
            uav_aoi_batch = self.uav_aoi[indices]
            uav_snr_batch = self.uav_snr[indices]
            uav_tuse_batch = self.uav_tuse[indices]
            uav_effort_batch = self.uav_effort[indices]
            action_dia_batch = self.action_dia[indices]
            value_pred_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            advantages_batch = advantages[indices]
            masks_batch = self.masks[indices]
            yield obs_batch, uav_aoi_batch, uav_snr_batch, uav_tuse_batch, uav_effort_batch, action_dia_batch, \
                  value_pred_batch, return_batch, masks_batch, old_action_log_probs_batch, advantages_batch,

    def rnn_feed_forward_generator(self, advantages):
        # mini_batch_size = (self.num_steps- params.rnn_seq_len) // self.mini_batch_num  # 500-5/4
        mini_batch_size = params.batch_size
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps - params.rnn_seq_len)), mini_batch_size,
                               drop_last=True)
        T = params.rnn_seq_len
        N = mini_batch_size
        for indices in sampler:
            obs_batch = self.obs[indices]
            uav_aoi_batch = self.uav_aoi[indices]
            uav_snr_batch = self.uav_snr[indices]
            uav_tuse_batch = self.uav_tuse[indices]
            uav_effort_batch = self.uav_effort[indices]
            action_dia_batch = self.action_dia[indices]
            value_pred_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            advantages_batch = advantages[indices]
            masks_batch = self.masks[indices]
            temporal_hidden_states = self.temporal_hidden_states[indices]
            sample_index = np.array(indices)
            for t in range(1, params.rnn_seq_len):
                cur_index = sample_index + t
                cur_index = cur_index.tolist()
                obs_batch_i = self.obs[cur_index]
                uav_aoi_batch_i = self.uav_aoi[cur_index]
                uav_snr_batch_i = self.uav_snr[cur_index]
                uav_tuse_batch_i = self.uav_tuse[cur_index]
                uav_effort_batch_i = self.uav_effort[cur_index]
                action_dia_batch_i = self.action_dia[cur_index]
                value_pred_batch_i = self.value_preds[cur_index]
                return_batch_i = self.returns[cur_index]
                old_action_log_probs_batch_i = self.action_log_probs[cur_index]
                advantages_batch_i = advantages[cur_index]
                masks_batch_i = self.masks[cur_index]
                if t == 1:
                    obs_batch = torch.stack((obs_batch, obs_batch_i))
                    uav_aoi_batch = torch.stack((uav_aoi_batch, uav_aoi_batch_i))
                    uav_snr_batch = torch.stack((uav_snr_batch, uav_snr_batch_i))
                    uav_tuse_batch = torch.stack((uav_tuse_batch, uav_tuse_batch_i))
                    uav_effort_batch = torch.stack((uav_effort_batch, uav_effort_batch_i))
                    action_dia_batch = torch.stack((action_dia_batch, action_dia_batch_i))
                    value_pred_batch = torch.stack((value_pred_batch, value_pred_batch_i))
                    return_batch = torch.stack((return_batch, return_batch_i))
                    old_action_log_probs_batch = torch.stack((old_action_log_probs_batch, old_action_log_probs_batch_i))
                    advantages_batch = torch.stack((advantages_batch, advantages_batch_i))
                    masks_batch = torch.stack((masks_batch, masks_batch_i))
                else:
                    obs_batch = torch.cat((obs_batch, obs_batch_i.unsqueeze(0)), dim=0)
                    uav_aoi_batch = torch.cat((uav_aoi_batch, uav_aoi_batch_i.unsqueeze(0)), dim=0)
                    uav_snr_batch = torch.cat((uav_snr_batch, uav_snr_batch_i.unsqueeze(0)), dim=0)
                    uav_tuse_batch = torch.cat((uav_tuse_batch, uav_tuse_batch_i.unsqueeze(0)), dim=0)
                    uav_effort_batch = torch.cat((uav_effort_batch, uav_effort_batch_i.unsqueeze(0)), dim=0)
                    action_dia_batch = torch.cat((action_dia_batch, action_dia_batch_i.unsqueeze(0)), dim=0)
                    value_pred_batch = torch.cat((value_pred_batch, value_pred_batch_i.unsqueeze(0)), dim=0)
                    return_batch = torch.cat((return_batch, return_batch_i.unsqueeze(0)), dim=0)
                    old_action_log_probs_batch = torch.cat(
                        (old_action_log_probs_batch, old_action_log_probs_batch_i.unsqueeze(0)), dim=0)
                    advantages_batch = torch.cat((advantages_batch, advantages_batch_i.unsqueeze(0)), dim=0)
                    masks_batch = torch.cat((masks_batch, masks_batch_i.unsqueeze(0)), dim=0)

            obs_batch = _flatten_helper(T, N, obs_batch)
            uav_aoi_batch = _flatten_helper(T, N, uav_aoi_batch)
            uav_snr_batch = _flatten_helper(T, N, uav_snr_batch)
            uav_tuse_batch = _flatten_helper(T, N, uav_tuse_batch)
            uav_effort_batch = _flatten_helper(T, N, uav_effort_batch)
            action_dia_batch = _flatten_helper(T, N, action_dia_batch)
            value_pred_batch = _flatten_helper(T, N, value_pred_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            advantages_batch = _flatten_helper(T, N, advantages_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)

            yield obs_batch, uav_aoi_batch, uav_snr_batch, uav_tuse_batch, uav_effort_batch, action_dia_batch, \
                  value_pred_batch, return_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                  temporal_hidden_states

    def spatial_att_feed_forward_generator(self, advantages):
        # mini_batch_size = (self.num_steps- params.rnn_seq_len) // self.mini_batch_num  # 500-5/4
        mini_batch_size = params.batch_size
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps - params.rnn_seq_len)), mini_batch_size,
                               drop_last=True)
        T = params.rnn_seq_len
        N = mini_batch_size
        for indices in sampler:
            obs_batch = self.obs[indices]
            uav_aoi_batch = self.uav_aoi[indices]
            uav_snr_batch = self.uav_snr[indices]
            uav_tuse_batch = self.uav_tuse[indices]
            uav_effort_batch = self.uav_effort[indices]
            action_dia_batch = self.action_dia[indices]
            value_pred_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            advantages_batch = advantages[indices]
            masks_batch = self.masks[indices]
            temporal_hidden_states = self.temporal_hidden_states[indices]
            spatial_hidden_states=self.spatial_hidden_states[indices]
            sample_index = np.array(indices)
            for t in range(1, params.rnn_seq_len):
                cur_index = sample_index + t
                cur_index = cur_index.tolist()
                obs_batch_i = self.obs[cur_index]
                uav_aoi_batch_i = self.uav_aoi[cur_index]
                uav_snr_batch_i = self.uav_snr[cur_index]
                uav_tuse_batch_i = self.uav_tuse[cur_index]
                uav_effort_batch_i = self.uav_effort[cur_index]
                action_dia_batch_i = self.action_dia[cur_index]
                value_pred_batch_i = self.value_preds[cur_index]
                return_batch_i = self.returns[cur_index]
                old_action_log_probs_batch_i = self.action_log_probs[cur_index]
                advantages_batch_i = advantages[cur_index]
                masks_batch_i = self.masks[cur_index]
                if t == 1:
                    obs_batch = torch.stack((obs_batch, obs_batch_i))
                    uav_aoi_batch = torch.stack((uav_aoi_batch, uav_aoi_batch_i))
                    uav_snr_batch = torch.stack((uav_snr_batch, uav_snr_batch_i))
                    uav_tuse_batch = torch.stack((uav_tuse_batch, uav_tuse_batch_i))
                    uav_effort_batch = torch.stack((uav_effort_batch, uav_effort_batch_i))
                    action_dia_batch = torch.stack((action_dia_batch, action_dia_batch_i))
                    value_pred_batch = torch.stack((value_pred_batch, value_pred_batch_i))
                    return_batch = torch.stack((return_batch, return_batch_i))
                    old_action_log_probs_batch = torch.stack((old_action_log_probs_batch, old_action_log_probs_batch_i))
                    advantages_batch = torch.stack((advantages_batch, advantages_batch_i))
                    masks_batch = torch.stack((masks_batch, masks_batch_i))
                else:
                    obs_batch = torch.cat((obs_batch, obs_batch_i.unsqueeze(0)), dim=0)
                    uav_aoi_batch = torch.cat((uav_aoi_batch, uav_aoi_batch_i.unsqueeze(0)), dim=0)
                    uav_snr_batch = torch.cat((uav_snr_batch, uav_snr_batch_i.unsqueeze(0)), dim=0)
                    uav_tuse_batch = torch.cat((uav_tuse_batch, uav_tuse_batch_i.unsqueeze(0)), dim=0)
                    uav_effort_batch = torch.cat((uav_effort_batch, uav_effort_batch_i.unsqueeze(0)), dim=0)
                    action_dia_batch = torch.cat((action_dia_batch, action_dia_batch_i.unsqueeze(0)), dim=0)
                    value_pred_batch = torch.cat((value_pred_batch, value_pred_batch_i.unsqueeze(0)), dim=0)
                    return_batch = torch.cat((return_batch, return_batch_i.unsqueeze(0)), dim=0)
                    old_action_log_probs_batch = torch.cat(
                        (old_action_log_probs_batch, old_action_log_probs_batch_i.unsqueeze(0)), dim=0)
                    advantages_batch = torch.cat((advantages_batch, advantages_batch_i.unsqueeze(0)), dim=0)
                    masks_batch = torch.cat((masks_batch, masks_batch_i.unsqueeze(0)), dim=0)

            obs_batch = _flatten_helper(T, N, obs_batch)
            uav_aoi_batch = _flatten_helper(T, N, uav_aoi_batch)
            uav_snr_batch = _flatten_helper(T, N, uav_snr_batch)
            uav_tuse_batch = _flatten_helper(T, N, uav_tuse_batch)
            uav_effort_batch = _flatten_helper(T, N, uav_effort_batch)
            action_dia_batch = _flatten_helper(T, N, action_dia_batch)
            value_pred_batch = _flatten_helper(T, N, value_pred_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            advantages_batch = _flatten_helper(T, N, advantages_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)

            yield obs_batch, uav_aoi_batch, uav_snr_batch, uav_tuse_batch, uav_effort_batch, action_dia_batch, \
                  value_pred_batch, return_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                  temporal_hidden_states,spatial_hidden_states
