# DRL-freshMCS
Additional materials for paper "[Mobile Crowdsensing for Data Freshness:
A Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/document/9488791)" accepted to INFOCOM 2021.

## :page_facing_up: Description
Data collection by mobile crowdsensing (MCS) is emerging as data sources for smart city applications, however how to ensure data freshness has sparse research exposure but quite important in practice. In this paper, we consider to use a group of mobile agents (MAs) like UAVs and driverless cars which are equipped with multiple antennas to move around in the task area to collect data from deployed sensor nodes (SNs). Our goal is to minimize the age of information (AoI) of all SNs and energy consumption of MAs during movement and data upload. To this end, we propose a centralized deep reinforcement learning (DRL)-based solution called “DRL-freshMCS” for controlling MA trajectory planning and SN scheduling. We further utilize implicit quantile networks to maintain the accurate value estimation and steady policies for MAs. Then, we design an exploration and exploitation mechanism by dynamic distributed prioritized experience replay. We also derive the theoretical lower bound for episodic AoI. Extensive simulation results show that DRLfreshMCS significantly reduces the episodic AoI per remaining energy, compared to five baselines when varying different number of antennas and data upload thresholds, and number of SNs. We also visualize their trajectories and AoI update process for clear illustrations.

## :wrench: Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/DRL-freshMCS.git
    cd DRL-freshMCS
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
python train.py --config realAoI_iqn_lstm.json --log-dir ./rltime_logs
```


## :checkered_flag: Testing

Test with the trained models 

```sh
python eval.py --path ./rltime_logs/your_model_path
```

Random test the env

```sh
python try_real_aoi.py
```

## :clap: Reference
- https://github.com/opherlieber/rltime


## :scroll: Acknowledgement

This paper was supported by National Natural Science
Foundation of China (No. 62022017).
<br>
Corresponding author: Chi Harold Liu.

## :e-mail: Contact

If you have any question, please email `3120215520@bit.edu.cn`.
