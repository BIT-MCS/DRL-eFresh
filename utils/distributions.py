import torch
import torch.nn as nn
import torch.distributions
from torch.utils.data import WeightedRandomSampler
from utils.base_utils import AddBias, init, init_normc_
import torch.nn.functional as F

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# TODO: recover original distribution method
FixedCategorical = torch.distributions.Categorical

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)


# entropy = FixedNormal.entropy
# FixedNormal.entropy = lambda self: entropy(self).sum(-1)

# FixedNormal.mode = lambda self: self.mean

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super(Categorical, self).__init__()
        self.device = device
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(num_inputs),
            init_(nn.Linear(num_inputs, num_outputs)),
        )

        self.dis_cat = None
        self.logits = 0
        self.sampler = torch.distributions.Categorical
        self.weighted_sampler = WeightedRandomSampler
        self.train()

    def forward(self, x):
        x = self.linear(x)
        self.logits = x
        self.dis_cat = torch.distributions.Categorical(logits=x)
        return self.dis_cat

    def gumble_sample(self):
        U = torch.distributions.Uniform(torch.zeros(self.logits.shape).to(self.device),
                                        torch.ones(self.logits.shape).to(self.device))
        u = U.sample()

        return torch.argmax(self.logits - torch.log(-torch.log(u)), dim=-1, keepdim=True)

    def sample(self):
        return self.dis_cat.sample()

    def gumble_softmax_sample(self, tau):
        dist = F.gumbel_softmax(self.logits, tau=tau, hard=False)
        action = torch.tensor(list(self.weighted_sampler(dist, 1, replacement=False))).to(self.device)
        # print('step', step, 'in rank', self.rank, dist, action)
        return action

    def mode(self):
        action = torch.argmax(self.logits, dim=1, keepdim=True)
        return action

    def softmax_sample(self):
        U = torch.distributions.Uniform(torch.zeros(self.logits.shape).to(self.device),
                                        torch.ones(self.logits.shape).to(self.device))
        u = U.sample()
        a = (self.logits - torch.log(-torch.log(u)))
        exp_a = torch.exp(a)
        z = torch.sum(exp_a, dim=-1, keepdim=True)
        probs = exp_a / z
        # print('probs', probs)
        cat = self.sampler(probs=probs)
        return cat.sample()

    # TODO: change log prob
    def log_probs(self, action):
        dist = self.dis_cat.log_prob(action.squeeze(-1))
        return dist

    def entropy(self):
        return self.dis_cat.entropy()

    def _entropy(self, logits):
        a = logits - torch.sum(logits, dim=-1, keepdim=True)
        exp_a = torch.exp(a)
        z = torch.sum(exp_a, dim=-1, keepdim=True)
        p = exp_a / z
        return torch.sum(p * (torch.log(z)) - a, dim=-1)

    def _log_probs(self, logits, action):
        self.cross_entropy_loss(logits, action)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        self.device = device
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(num_inputs),
            init_(nn.Linear(num_inputs, num_outputs)),

        )
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean = torch.tanh(action_mean)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.to(self.device)

        action_logstd = self.logstd(zeros)
        action_logstd = torch.tanh(action_logstd)
        # print('action log std in uav_collection charge2:',action_logstd)
        # print('mean', action_mean, 'std', action_logstd)
        # TODO: Fixed Normal input(mean, standard deviation)
        return FixedNormal(action_mean, action_logstd.exp().sqrt())
