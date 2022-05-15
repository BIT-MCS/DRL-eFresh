from main_setting import Params
import torch.multiprocessing as mp
from methods.model import Model, Shared_grad_buffers
from utils.base_utils import TrafficLight, Counter
from train_agent import train
from test_agent import test
import torch.optim as optim
from chief import chief
import torch
import numpy as np
import time
import os

params = Params()
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    if params.log_dir is None:
        local_time = str(time.strftime("%Y %m-%d %H-%M-%S", time.localtime()))
    else:
        local_time = params.log_dir

    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    params.log_info(local_time)

    # Training
    if params.trainable:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.cuda_device)

        device = torch.device('cuda' if params.use_cuda else 'cpu')
        traffic_light = TrafficLight()
        counter = Counter()
        son_process_counter = Counter()

        # -------------get environment information------------
        obs_shape = (params.image_depth, params.image_size, params.image_size)
        # --------------create shared model-------------------
        shared_model = Model(obs_shape, params.uav_num, device)
        shared_model.share_memory().to(device)

        # ------------create shared grad buffer list----------
        shared_grad_buffer = Shared_grad_buffers(shared_model, device)

        # -----------create optimizer list --------------------
        optimizer = optim.Adam(list(shared_model.parameters()), lr=params.lr, eps=params.adam_eps)
        exponential_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.ppo_epoch * (
                    params.max_time_steps // params.batch_size), gamma=0.999)

        processes = []

        p = mp.Process(target=chief, args=(
            params.update_threshold, traffic_light, counter, shared_model, shared_grad_buffer, optimizer,
            son_process_counter, params.max_grad_norm, local_time, exponential_lr_scheduler))
        p.start()
        processes.append(p)
        if params.train:
            for rank in range(0, params.num_processes):
                p = mp.Process(target=train, args=(
                    rank, traffic_light, counter, shared_model, shared_grad_buffer,
                    local_time, son_process_counter, device))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        print("Thank you for your waiting！ O(∩_∩)O")
    else:
        device = torch.device('cpu')
        model_path = params.test_path+params.test_model_path  # dppo model path
        print(str(model_path))
        test(local_time, model_path, device)
