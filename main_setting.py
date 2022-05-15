import torch.nn as nn
import csv
import os

class Params(nn.Module):
    def __init__(self):
        super(Params, self).__init__()
        self.trainable = True
        self.debug_mode = False
        self.test_random = False
        self.cuda_device = 0

        self.log_dir = None # TODO:None
        self.test_path='/home/linc/dzp/dppo_age/result0507/5uavs/'
        # self.test_path = '/home/linc/dzp/dppo_age/result/b64_try_ppo_tc_train/'
        self.test_model_path="ckpt/model_3399.pt"
        self.root_path = 'result/'

        self.save_interval = 100
        self.use_cuda = True
        self.use_opt = True

        # --------------PPO parameters------------------
        self.lr = 3e-4  # TODO:3e-4, try 1e-4
        self.clip = 0.1
        self.ent_coeff = 0.01
        self.value_coeff = 0.1
        self.clip_coeff = 1.
        self.max_train_episode = 2500  # TODO:2500
        self.max_time_steps = 500 # TODO:500
        self.num_processes = 15  # TODO：6 ->10G
        self.max_test_episode = 1  # TODO:50
        self.mini_batch_num = 3
        self.batch_size = 64  # TODO:64
        self.ppo_epoch = 3
        self.max_grad_norm = 250
        self.update_threshold = self.num_processes - 1
        self.adam_eps = 1e-5
        self.seed = 1
        self.use_obs_norm = False  # Not necessary
        self.use_adv_norm = False

        # ------Method & NN parameters------------
        self.use_rnn = True
        self.use_relational_att = False
        self.use_spatial_att = False
        self.temporal_hidden_size = 256  # == hidden_state_size in rnn
        self.spatial_hidden_size = 128
        self.rnn_seq_len = 10

        # -----discounted return parameters-------
        self.use_gae = True
        self.gamma = 0.99
        self.gae_lambda = 0.95

        # ----------environment parameters---------
        self.uav_num = 2
        self.uav_action_dim = 3
        self.image_size = 80
        self.image_depth = 2

    def log_info(self, local_time):
        log_file_path = os.path.join(self.root_path, str(local_time))
        os.makedirs(log_file_path)
        log_file_path = os.path.join(log_file_path, 'hyper_parameters.csv')
        log_file = open(log_file_path, 'a', newline='')
        file_reader = csv.writer(log_file)
        for p in self.__dict__:
            if p[0] == '_':
                continue
            file_reader.writerow([p, self.__getattribute__(p)])
        log_file.close()
