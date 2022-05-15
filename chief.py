import torch.nn as nn
import time
import datetime
import os
from main_setting import Params

params = Params()

def chief(update_threshold, traffic_light, counter, shared_model, shared_grad_buffers, optimizer, son_process_counter,
          max_grad_norm, local_time,lr_scheduler):
    start_time = datetime.datetime.now()
    last_lr=0
    while True:
        time.sleep(1)
        if counter.get() > update_threshold:
            now_lr=optimizer.state_dict()['param_groups'][0]['lr']
            if now_lr !=last_lr:
                print("LR=",now_lr)
                last_lr=now_lr

            optimizer.zero_grad()
            shared_grad_buffers.average_gradient()
            for n, p in shared_model.named_parameters():
                p._grad = shared_grad_buffers.grads[n + '_grad'].clone().detach()
            nn.utils.clip_grad_norm_(shared_model.parameters(), max_grad_norm)
            # shared_model.print_grad()
            # shared_grad_buffers.print_gradient()
            optimizer.step()
            shared_grad_buffers.reset()
            counter.reset()
            # print(counter.get())
            traffic_light.switch()  # workers start new loss computation
            # print('update')
            lr_scheduler.step()
        if son_process_counter.get() > update_threshold:
            break

    total_time = datetime.datetime.now() - start_time
    time_root_path = os.path.join(params.root_path, str(local_time))
    time_file = open(os.path.join(time_root_path, 'run_time.txt'), 'w', newline='')
    time_file.write(str(total_time))
    time_file.close()
