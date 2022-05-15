import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt
from main_setting import Params
from matplotlib.backends.backend_pdf import PdfPages
import os

params=Params()

def prRed(prt): print("\033[91m {}\033[00m".format(prt))


def prGreen(prt): print("\033[92m {}\033[00m".format(prt))


def prYellow(prt): print("\033[93m {}\033[00m".format(prt))


def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))


def prPurple(prt): print("\033[95m {}\033[00m".format(prt))


def prCyan(prt): print("\033[96m {}\033[00m".format(prt))


def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))


def prBlack(prt): print("\033[98m {}\033[00m".format(prt))


class Util:
    def __init__(self, device):
        self.device = device
        self.USE_CUDA = True if 'cuda' in device.type else False

    def to_numpy(self, var, is_deep_copy=True):

        # list type [ Tensor, Tensor ]
        if isinstance(var, list) and len(var) > 0:
            var_ = []
            for v in var:
                temp = v.cpu().data.numpy() if self.USE_CUDA else v.data.numpy()

                # this part is meaningless if Tensor is in gpu
                if is_deep_copy:
                    var_.append(copy.deepcopy(temp))
            return var_

        # dict type { key, Tensor }
        if isinstance(var, dict) and len(var) > 0:
            var_ = {}
            for k, v in var.iteritems():
                temp = v.cpu().data.numpy() if self.USE_CUDA else v.data.numpy()

                # this part is meaningless if Tensor is in gpu
                if is_deep_copy:
                    var_[k] = copy.deepcopy(temp)
            return var_

        var = var.cpu().data.numpy() if self.USE_CUDA else var.data.numpy()
        # this part is meaningless if Tensor is in gpu
        if is_deep_copy:
            var = copy.deepcopy(var)
        return var

    def to_tensor(self, ndarray, requires_grad=False, is_deep_copy=True):
        if ndarray is None:
            return ndarray

        # this part is meaningless if tensor is in gpu
        if is_deep_copy:
            ndarray = copy.deepcopy(ndarray)

        if isinstance(ndarray, list) and len(ndarray) > 0:
            var_ = []
            for v in ndarray:
                temp = torch.from_numpy(v)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_.append(temp)
            return var_
        if isinstance(ndarray, dict) and len(ndarray) > 0:
            var_ = {}
            for k, v in ndarray.iteritems():
                temp = torch.from_numpy(v)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_[k] = temp

            return var_

        #TODO:bug?
        # ndarray = torch.from_numpy(ndarray).type(dtype)
        ndarray = torch.from_numpy(ndarray)
        ndarray = ndarray.to(self.device)
        ndarray.requires_grad = requires_grad

        return ndarray

    def to_int_tensor(self, ndarray, requires_grad=False, is_deep_copy=True):
        if ndarray is None:
            return ndarray

        # this part is meaningless if tensor is in gpu
        if is_deep_copy:
            ndarray = copy.deepcopy(ndarray)

        if isinstance(ndarray, list) and len(ndarray) > 0:
            var_ = []
            for v in ndarray:
                temp = torch.from_numpy(v)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_.append(temp)
            return var_
        if isinstance(ndarray, dict) and len(ndarray) > 0:
            var_ = {}
            for k, v in ndarray.iteritems():
                temp = torch.from_numpy(v)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_[k] = temp

            return var_

        ndarray = torch.from_numpy(ndarray)
        ndarray = ndarray.to(self.device)
        ndarray.requires_grad = requires_grad

        return ndarray


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu
        np.random.seed(123456)

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx

        return self.X


import torch.multiprocessing as mp


class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)


class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


# def get_vec_normalize(venv):
#     if isinstance(venv, VecNormalize):
#         return venv
#     elif hasattr(venv, 'venv'):
#         return get_vec_normalize(venv.venv)
#
#     return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py #L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

def plot_line(name,list_data,full_path):
    pdf = PdfPages(full_path+"/"+name+'.pdf')

    plt.figure(figsize=(20, 13))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.plot(list_data,linewidth=4)
    if params.trainable is True:
        plt.xlabel("Training episode",fontsize=40)
    else:
        plt.xlabel("Step",fontsize=40)
    plt.ylabel(name,fontsize=40)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.savefig(full_path + "/"+name+'.png')
    pdf.savefig()

    plt.close()
    pdf.close()

def plot_error_lines(name,list_mean,list_min,list_max,list_var,full_path):
    plt.plot(list_mean,color='green',label="mean")
    plt.plot(list_var,color='red',label="std")
    plt.plot(list_min,color='b',alpha=0.2,label="min")
    plt.plot(list_max,color='b',alpha=0.2,label="max")
    if params.trainable is True:
        plt.xlabel("Training episode")
    else:
        plt.xlabel("Step")

    plt.fill_between(range(len(list_mean)),list_min,list_max,color='b',alpha=0.2)
    plt.ylabel(name)
    plt.grid(True)
    plt.legend()
    plt.grid(linestyle='--')
    plt.savefig(full_path + "/"+name+'.png')
    plt.close()

def plot_age_lines(uav_aoi_list,move_aoi_list,collect_aoi_list,sendback_aoi_list,full_path):
    uav_aoi_array=np.array(uav_aoi_list).transpose((1,0))
    move_aoi_array = np.array(move_aoi_list).transpose((1, 0))
    collect_aoi_array = np.array(collect_aoi_list).transpose((1, 0))
    sendback_aoi_array = np.array(sendback_aoi_list).transpose((1, 0))
    uav_num=uav_aoi_array.shape[0]
    for i in range(uav_num):
        if params.trainable is True:
            plt.xlabel("Training episode")
        else:
            plt.xlabel("Step")
        plt.ylabel("second")
        plt.plot(uav_aoi_array[i],label="total_aoi")
        plt.plot(move_aoi_array[i], label="move_aoi")
        plt.plot(collect_aoi_array[i], label="collect_aoi")
        plt.plot(sendback_aoi_array[i], label="sendback_aoi")
        plt.grid(True)
        plt.legend()
        plt.grid(linestyle='--')
        plt.savefig(full_path + "/" + "Age_uav_%d"%i + '.png')
        plt.close()


def plot_reward_lines(uav_reward_list, uav_penalty_list,full_path):
    uav_reward_array=np.array(uav_reward_list).transpose((1,0))
    uav_penalty_array = np.array(uav_penalty_list).transpose((1, 0))
    uav_num = uav_reward_array.shape[0]
    for i in range(uav_num):
        if params.trainable is True:
            plt.xlabel("Training episode")
        else:
            plt.xlabel("Step")
        plt.ylabel("Reward/Penalty")
        plt.plot(uav_reward_array[i],label="positive gain")
        plt.plot(uav_penalty_array[i], label="negative penalty")
        plt.grid(True)
        plt.grid(linestyle='--')
        plt.legend()
        plt.savefig(full_path + "/" + "Reward_uav_%d"%i + '.png')
        plt.close()

def plot_fairness_lines(jain_fairness_list,gs_fairness_list,full_path):
    plt.plot(jain_fairness_list,label="jain_fairness")
    plt.plot(gs_fairness_list, label="gs_fairness")
    if params.trainable is True:
        plt.xlabel("Training episode")
    else:
        plt.xlabel("Step")
    plt.ylabel("fairness")
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig(full_path + "/"+"different_fairness"+'.png')
    plt.close()

