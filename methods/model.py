from utils.distributions import DiagGaussian
from utils.base_utils import Counter
from main_setting import Params
from methods.RelationalGRU import RelationalGRU
from methods.SpatialAttGRU import SpatialAttGRU
import torch.nn as nn
import torch

params = Params()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    else:
        print('None bias')
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, obs_shape, num_of_uav, device, trainable=True, hidden_size=params.temporal_hidden_size):
        # todo 1: add parameter trainable=True
        super(Model, self).__init__()
        # feature extract
        self.base = NNBase(obs_shape[0], device, trainable)
        # actor
        self.dist_dia = DiagGaussian(hidden_size + num_of_uav * 4, params.uav_action_dim * num_of_uav,
                                     device)  # continuous, (dx, dy,v,coll_t)
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        # critic
        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size + num_of_uav * 4, 1))
        )
        self.device = device
        # todo 2: distinguish train and eval
        if trainable:
            self.train()
        else:
            self.eval()

    def act(self, inputs, uav_aoi, uav_snr, uav_compl, uav_tc_compl, temporal_hidden_state=None, mask=None,
            spatial_hidden_state=None):
        if params.use_rnn:
            if params.use_spatial_att:
                obs_feature, temporal_hidden_state, spatial_hidden_state, = self.base(inputs, temporal_hidden_state,
                                                                                      mask, spatial_hidden_state)
            else:
                obs_feature, temporal_hidden_state = self.base(inputs, temporal_hidden_state, mask)

        else:
            obs_feature = self.base(inputs)
        full_obs_feature = torch.cat(
            [obs_feature, uav_aoi.float(), uav_snr.float(), uav_compl.float(), uav_tc_compl.float()], dim=1)
        value = self.critic(full_obs_feature)
        dist_dia = self.dist_dia(full_obs_feature)
        action_dia = dist_dia.sample()

        action_log_probs_dia = dist_dia.log_probs(action_dia)

        # print('action log probs dia', action_log_probs_dia.mean())
        if params.use_rnn:
            if params.use_spatial_att:
                return value, action_dia, action_log_probs_dia, temporal_hidden_state, spatial_hidden_state
            else:
                return value, action_dia, action_log_probs_dia, temporal_hidden_state

        else:
            return value, action_dia, action_log_probs_dia

    def get_value(self, inputs, uav_aoi, uav_snr, uav_compl, uav_tc_compl, temporal_hidden_state=None, mask=None,
                  spatial_hidden_state=None):
        if params.use_rnn:
            if params.use_spatial_att:
                obs_feature, _, _ = self.base(inputs, temporal_hidden_state, mask, spatial_hidden_state)
            else:
                obs_feature, _ = self.base(inputs, temporal_hidden_state, mask)
        else:
            obs_feature = self.base(inputs)

        full_obs_feature = torch.cat([obs_feature, uav_aoi, uav_snr, uav_compl, uav_tc_compl], dim=1)
        value = self.critic(full_obs_feature)
        return value

    def evaluate_actions(self, inputs, uav_aoi, uav_snr, uav_compl, uav_tc_compl, action, temporal_hidden_state=None,
                         mask=None, spatial_hidden_state=None):
        if params.use_rnn:
            if params.use_spatial_att:
                obs_features, _, _ = self.base(inputs, temporal_hidden_state, mask, spatial_hidden_state)
            else:
                obs_features, _ = self.base(inputs, temporal_hidden_state, mask)

        else:
            obs_features = self.base(inputs)

        full_obs_features = torch.cat([obs_features, uav_aoi, uav_snr, uav_compl, uav_tc_compl], dim=1)
        value = self.critic(full_obs_features)
        dist_dia = self.dist_dia(full_obs_features)
        action_log_probs_dia = dist_dia.log_probs(action)
        dist_entropy_dia = dist_dia.entropy().mean()

        return value, action_log_probs_dia, dist_entropy_dia

    def print_grad(self):
        for name, p in self.base.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


class NNBase(nn.Module):
    def __init__(self, num_inputs, device, trainable=True, hidden_size=params.temporal_hidden_size):
        super(NNBase, self).__init__()
        self._feature_size = hidden_size

        if params.use_rnn is False or params.use_spatial_att is False:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain('relu'))

            self.feature = nn.Sequential(
                # input: 3*80*80
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
                nn.LayerNorm([32, 19, 19]),
                nn.ReLU(inplace=True),
                # input: 32*19*19
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.LayerNorm([64, 8, 8]),
                nn.ReLU(inplace=True),
                # input: 64*8*8
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.LayerNorm([32, 6, 6]),
                nn.ReLU(inplace=True),
                # output: 32*6*6
            ).to(device)

            init_ = lambda m: init(m,
                                   nn.init.orthogonal_,
                                   lambda x: nn.init.constant_(x, 0))

            self.conv_to_flat = nn.Sequential(
                Flatten(),
                init_(nn.Linear(32 * 6 * 6, self._feature_size)),
                nn.LayerNorm([self._feature_size]),
                nn.ReLU(inplace=True),
            ).to(device)

        if params.use_rnn:
            if params.use_relational_att:
                # self.gru = RelationalGRU(input_size=hidden_size, hidden_dim=hidden_size, use_att=params.use_att).to(
                #     device)
                self.gru = RelationalGRU(input_size=hidden_size, hidden_dim=hidden_size, use_att=False).to(
                    device)
            elif params.use_spatial_att:
                self.gru = SpatialAttGRU(input_dim=num_inputs).to(device)
            else:
                self.gru = nn.GRU(hidden_size, hidden_size).to(device)

            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        if trainable:
            self.train()
        else:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def _forward_gru(self, x, h_c, masks):
        if x.size(0) == h_c.size(0):
            x, h_c = self.gru(x.unsqueeze(0), (h_c * masks).unsqueeze(0))
            x = x.squeeze(0)
            h_c = h_c.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = h_c.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            h_c = h_c.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, h_c = self.gru(
                    x[start_idx:end_idx],
                    h_c * (masks[start_idx].unsqueeze(0).unsqueeze(-1)))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            h_c = h_c.squeeze(0)

        return x, h_c

    def _forward_spatial_gru(self, x, t_h, masks, s_h):
        if x.size(0) == t_h.size(0):
            x, t_h, s_h = self.gru(x.unsqueeze(0), (t_h * masks).unsqueeze(0), s_h.unsqueeze(0))
            x = x.squeeze(0)
            t_h = t_h.squeeze(0)
            s_h = s_h.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = t_h.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(-3), x.size(-2), x.size(-1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            t_h = t_h.unsqueeze(0)
            s_h = s_h.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, t_h, s_h = self.gru(
                    x[start_idx:end_idx],
                    t_h * (masks[start_idx].unsqueeze(0).unsqueeze(-1)),
                    s_h)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            t_h = t_h.squeeze(0)
            s_h = s_h.squeeze(0)

        return x, t_h, s_h

    def forward(self, inputs, temporal_hidden_state=None, masks=None, spatial_hidden_state=None):
        if not params.use_rnn:
            x = self.feature(inputs)
            x = self.conv_to_flat(x)
            return x

        if params.use_rnn and params.use_spatial_att is False:  # TODO
            x = self.feature(inputs)
            x = self.conv_to_flat(x)
            x, temporal_hidden_state = self._forward_gru(x, temporal_hidden_state, masks)
            return x, temporal_hidden_state

        if params.use_rnn and params.use_spatial_att:
            x = inputs
            x, temporal_hidden_state, spatial_hidden_state, = self._forward_spatial_gru(x, temporal_hidden_state, masks,
                                                                                        spatial_hidden_state)
            return x, temporal_hidden_state, spatial_hidden_state


class Shared_grad_buffers():
    def __init__(self, model, device):
        self.grads = {}
        self.counter = Counter()

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_().to(device)

    def add_gradient(self, model):
        self.counter.increment()
        for name, p in model.named_parameters():
            if p.requires_grad:
                # print("name:",name)
                # print("data:", p.grad)
                self.grads[name + '_grad'] += p.grad.data

    def average_gradient(self):
        counter_num = self.counter.get()
        for name, grad in self.grads.items():
            self.grads[name] /= counter_num

    def print_gradient(self):
        for grad in self.grads:
            if 'base.critic' in grad:
                # if grad == 'fc1.weight_grad':
                print(grad, '  ', self.grads[grad].mean())
        for name, grad in self.grads.items():
            if 'base.critic' in name:
                print(name, self.grads[name].mean())

    def reset(self):
        self.counter.reset()
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)

# Maybe not necessary in image inputs
class Shared_obs_stats():
    def __init__(self, num_inputs,device):
        self.n = torch.zeros(num_inputs).share_memory_().to(device)
        self.mean = torch.zeros(num_inputs).share_memory_().to(device)
        self.mean_diff = torch.zeros(num_inputs).share_memory_().to(device)
        self.var = torch.zeros(num_inputs).share_memory_().to(device)

    def observes(self, obs):
        # observation mean var updates
        x = obs.data.squeeze()
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean.unsqueeze(0).expand_as(inputs)
        obs_std = torch.sqrt(self.var).unsqueeze(0).expand_as(inputs)
        return torch.clamp((inputs - obs_mean) / obs_std, -5., 5.)
