# DRL-eFresh
This is the code accompanying the paper: "[Delay-Sensitive Energy-Efficient UAV
Crowdsensing by Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9540290/)" accepted in TMC.

## :page_facing_up: Description
Mobile crowdsensing (MCS) by unmanned aerial vehicles (UAVs) servicing delay-sensitive applications becomes popular by navigating a group of UAVs to take advantage of their equipped high-precision sensors and durability for data collection in harsh environments. In this paper, we aim to simultaneously maximize collected data amount, geographical fairness, and minimize the energy consumption of all UAVs, as well as to guarantee the data freshness by setting a deadline in each timeslot. Specifically, we propose a centralized control, distributed execution framework by decentralized deep reinforcement learning (DRL) for delay-sensitive and energy-efficient UAV crowdsensing, called "DRL-eFresh". It includes a synchronous computational architecture with GRU sequential modeling to generate multi-UAV navigation decisions. Also, we derive an optimal time allocation solution for data collection while considering all UAV efforts and avoiding much data dropout due to limited data upload time and wireless data rate.

## :wrench: Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/DRL-eFresh.git
    cd DRL-eFresh
    ```
2. Install dependent packages
    ```sh
    conda create -n mcs python==3.8
    conda activate mcs
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 tensorboard future
    pip install -r requirements.txt
    ```


## :computer: Training

Train our solution
```bash
# set "trainable=True" in main_setting.py
python main.py
```


## :checkered_flag: Testing

Test with the trained models 

```sh
# set "trainable=False" and "test_path" in main_setting.py
python main.py
```

Random test the env

```sh
python test_random_agent.py
```

## :clap: Reference
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


## :scroll: Acknowledgement

This paper was supported by National Natural Science
Foundation of China (No. 62022017).
<br>
Corresponding author: Chi Harold Liu.

## :e-mail: Contact

If you have any question, please email `3120215520@bit.edu.cn`.

## Paper
If you are interested in our work, please cite our paper as

```
@ARTICLE{dai2021delay,
    author={Dai, Zipeng and Liu, Chi Harold and Han, Rui and Wang, Guoren and Leung, Kin and Tang, Jian},  
    journal={IEEE Transactions on Mobile Computing (TMC)},   
    title={Delay-Sensitive Energy-Efficient UAV Crowdsensing by Deep Reinforcement Learning},   
    year={2021},  
    pages={1-1},  
    }
```
