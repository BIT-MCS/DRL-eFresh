import os
import matplotlib
import pandas as pd
from main_setting import Params
from utils.base_utils import plot_line, plot_age_lines, plot_reward_lines, plot_fairness_lines, plot_error_lines

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

params = Params()


class Log(object):
    def __init__(self, worker_id=None, num_of_step=None, root_path=None, time=None):
        self.time = time
        self.num_of_step = num_of_step
        self.full_path = os.path.join(root_path, self.time)
        self.worker_id = worker_id
        if worker_id is not None:
            self.full_path = os.path.join(self.full_path, str(worker_id))
        os.makedirs(self.full_path)

        self.file_path = self.full_path + '/REPORT.txt'
        file = open(self.file_path, 'w')
        file.close()
        self.result_path = self.full_path + '/' + 'result.npz'
        self.color_list = ["red", "purple", "green", "grey", "blue"]

        if worker_id == 0:
            self.reward_list = []
            self.data_collection_list = []
            self.energy_comsumption_list = []
            self.eficiency_list = []
            self.aoi_list = []
            self.uav_aoi_list = []
            self.move_aoi_list = []
            self.collect_aoi_list = []
            self.send_back_aoi_list = []
            self.uav_reward_list = []
            self.uav_penalty_list = []
            self.jain_fairness_list = []

            self.time_usage_ratio_list = []
            self.time_usage_ratio_min_list = []
            self.time_usage_ratio_max_list = []
            self.time_usage_ratio_std_list = []

            self.collection_effort_ratio_list = []
            self.collection_effort_ratio_min_list = []
            self.collection_effort_ratio_max_list = []
            self.collection_effort_ratio_std_list = []

            self.data_completion_ratio_list = []
            self.data_completion_ratio_min_list = []
            self.data_completion_ratio_max_list = []
            self.data_completion_ratio_std_list = []

    def log(self, values):
        if isinstance(values, dict):
            with open(self.file_path, 'a') as file:
                for key, value in values.items():
                    file.write(str([key, value]) + '\n')
                    # print key, value, "file",file
        elif isinstance(values, list):
            with open(self.file_path, 'a') as file:
                for value in values:
                    file.write(str(value) + '\n')
                    # print value,"file",file
        else:
            with open(self.file_path, 'a') as file:
                file.write(str(values) + '\n')
                # print values, "file",file

    def circle(self, x, y, r, color=np.stack([1., 0., 0.]), count=50):
        xarr = []
        yarr = []
        for i in range(count):
            j = float(i) / count * 2 * np.pi
            xarr.append(x + r * np.cos(j))
            yarr.append(y + r * np.sin(j))
        plt.plot(xarr, yarr, c=color)

    def draw_path(self, env, episode):
        full_path = os.path.join(self.full_path, 'Path')
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
        xxx = []
        colors = []
        for x in range(env.map_size_x):  # 16
            xxx.append((x, 1))
        for y in range(env.map_size_y):  # 16
            c = []
            for x in range(env.map_size_x):
                if env.map_obstacle[x][y] == env.map_obstacle_value:
                    c.append((1, 0, 0, 1))
                else:
                    c.append((1, 1, 1, 1))
            colors.append(c)

        Fig = plt.figure(figsize=(6, 6))
        PATH = env.uav_trace
        POI_PATH = [[] for _ in range(env.uav_num)]
        POWER_PATH = [[] for _ in range(env.uav_num)]
        DEAD_PATH = [[] for _ in range(env.uav_num)]

        for i in range(env.uav_num):
            for j, pos in enumerate(PATH[i]):
                if env.uav_state[i][j] == 0:  # DEAD
                    DEAD_PATH[i].append(pos)
                elif env.uav_state[i][j] == 1:  # collect data
                    POI_PATH[i].append(pos)
                elif env.uav_state[i][j] == -1:  # charge power(unnecessary)
                    POWER_PATH[i].append(pos)
                else:
                    print(" error")

        for i1 in range(env.map_size_y):
            plt.broken_barh(xxx, (i1, 1), facecolors=colors[i1])

        plt.scatter(env.poi_data_pos[:, 0], env.poi_data_pos[:, 1], c=env.init_poi_data_val[:])

        for i in range(env.uav_num):
            # M = Fig.add_subplot(1, 1, i + 1)
            plt.ylim(ymin=0, ymax=env.map_size_x)
            plt.xlim(xmin=0, xmax=env.map_size_y)
            # color = np.random.random(3)
            if i<5:
                color = self.color_list[i]
            else:
                color="blue"
            plt.plot(np.stack(PATH[i])[:, 0], np.stack(PATH[i])[:, 1], color=color)

            if len(POI_PATH[i]) > 0:
                plt.scatter(np.stack(POI_PATH[i])[:, 0], np.stack(POI_PATH[i])[:, 1], color=color, marker='.')

            if len(POWER_PATH[i]) > 0:
                plt.scatter(np.stack(POWER_PATH[i])[:, 0], np.stack(POWER_PATH[i])[:, 1], color=color * 0.5, marker='+')

            if len(DEAD_PATH[i]) > 0:
                plt.scatter(np.stack(DEAD_PATH[i])[:, 0], np.stack(DEAD_PATH[i])[:, 1], color=color, marker='D')

        plt.grid(True, linestyle='-.', color='r')
        data_collection = np.round(env.data_collection_ratio(), 4)
        jain_fairness = np.round(env.get_jain_fairness(), 4)
        our_fairness = np.round(env.geographical_fairness(), 4)
        energy_comsumption = np.round(env.energy_consumption_ratio(), 4)
        aoi = np.round(env.age_of_information(), 4)
        rew = np.round(env.episodic_total_uav_reward, 4)
        effi = np.round(env.time_energy_efficiency(), 4)

        time_usage_ratio = np.round(env.time_usage_ratio(), 4)
        time_usage_ratio_min, time_usage_ratio_max, time_usage_ratio_std = np.round(env.time_usage_ratio_min_max_std(),
                                                                                    4)
        collection_effort_ratio = np.round(env.collection_effort_ratio(), 4)
        collection_effort_ratio_min, collection_effort_ratio_max, collection_effort_ratio_std = np.round(
            env.collection_effort_ratio_min_max_std(), 4)
        data_completion_ratio = np.round(env.data_completion_ratio(), 4)
        data_completion_ratio_min, data_completion_ratio_max, data_completion_ratio_std = np.round(
            env.data_completion_ratio_min_max_std(), 4)

        plt.title(str(episode) + ':d_c=' + str(data_collection) + ' g_f=' + str(our_fairness) + ' e_c=' + str(
            energy_comsumption) + ' aoi=%.1f' % aoi + '\nt_usage=' + str(time_usage_ratio) + ' effort=' +
                  str(collection_effort_ratio) + ' compl=' + str(data_completion_ratio) + '\nr='
                  + str(rew) + ' eff=' + str(effi))
        plt.scatter(env.bs_pos[:, 0], env.bs_pos[:, 1], marker='^', color="blue", s=200)

        # for i in range(env.bs_pos.shape[0]):
        #     self.circle(env.bs_pos[i, 0], env.bs_pos[i, 1], env.c_range / 1000 * 16)

        Fig.savefig(full_path + '/path_' + str(episode) + '.pdf')

        plt.close()

        if self.worker_id == 0:
            self.reward_list.append(rew)
            self.data_collection_list.append(data_collection)
            self.energy_comsumption_list.append(energy_comsumption)
            self.jain_fairness_list.append(jain_fairness)
            self.aoi_list.append(aoi)
            self.eficiency_list.append(effi)
            self.uav_aoi_list.append(env.uav_aoi_list)
            self.move_aoi_list.append(env.move_aoi_list)
            self.collect_aoi_list.append(env.collect_aoi_list)
            self.send_back_aoi_list.append(env.send_back_aoi_list)
            self.uav_reward_list.append(env.uav_reward_list)
            self.uav_penalty_list.append(env.uav_penalty_list)

            self.time_usage_ratio_list.append(time_usage_ratio)
            self.time_usage_ratio_min_list.append(time_usage_ratio_min)
            self.time_usage_ratio_max_list.append(time_usage_ratio_max)
            self.time_usage_ratio_std_list.append(time_usage_ratio_std)
            self.collection_effort_ratio_list.append(collection_effort_ratio)
            self.collection_effort_ratio_min_list.append(collection_effort_ratio_min)
            self.collection_effort_ratio_max_list.append(collection_effort_ratio_max)
            self.collection_effort_ratio_std_list.append(collection_effort_ratio_std)
            self.data_completion_ratio_list.append(data_completion_ratio)
            self.data_completion_ratio_min_list.append(data_completion_ratio_min)
            self.data_completion_ratio_max_list.append(data_completion_ratio_max)
            self.data_completion_ratio_std_list.append(data_completion_ratio_std)

            if params.trainable is True:
                plot_error_lines("time_usage_ratio", self.time_usage_ratio_list,
                                 self.time_usage_ratio_min_list, self.time_usage_ratio_max_list,
                                 self.time_usage_ratio_std_list, self.full_path)
                plot_error_lines("collection_effort_ratio", self.collection_effort_ratio_list,
                                 self.collection_effort_ratio_min_list,
                                 self.collection_effort_ratio_max_list,
                                 self.collection_effort_ratio_std_list, self.full_path)
                plot_error_lines("data_completion_ratio", self.data_completion_ratio_list,
                                 self.data_completion_ratio_min_list,
                                 self.data_completion_ratio_max_list,
                                 self.data_completion_ratio_std_list, self.full_path)
            else:
                plot_line("time_usage_ratio", self.time_usage_ratio_list, self.full_path)
                plot_line("collection_effort_ratio", self.collection_effort_ratio_list, self.full_path)
                plot_line("data_completion_ratio", self.data_completion_ratio_list, self.full_path)

            plot_line("Data collection ratio", self.data_collection_list, self.full_path)
            plot_line("Geographical fairness", self.jain_fairness_list, self.full_path)
            plot_line("Energy consumption Ratio", self.energy_comsumption_list, self.full_path)
            plot_line("Age of information (second)", self.aoi_list, self.full_path)
            plot_line("Energy efficiency", self.eficiency_list, self.full_path)
            plot_age_lines(self.uav_aoi_list, self.move_aoi_list, self.collect_aoi_list, self.send_back_aoi_list,
                           self.full_path)
            plot_reward_lines(self.uav_reward_list, self.uav_penalty_list, self.full_path)

            if params.trainable is False and params.debug_mode is False and len(
                    self.time_usage_ratio_list) == params.max_test_episode:
                df = pd.DataFrame(columns=["test_episode",
                                           "data_collection_ratio", "fairness", "energy",
                                           "time_usage", "collection_effort", "data_completion",
                                           "tm", "tc", "ts", "t_total","efficiency"])
                for index in range(params.max_test_episode):
                    df.loc[index] = [
                        index,
                        self.data_collection_list[index], self.jain_fairness_list[index],
                        self.energy_comsumption_list[index],
                        self.time_usage_ratio_list[index], self.collection_effort_ratio_list[index],
                        self.data_completion_ratio_list[index],
                        np.mean(self.move_aoi_list[index]), np.mean(self.collect_aoi_list[index]),
                        np.mean(self.send_back_aoi_list[index]), self.aoi_list[index],
                        self.eficiency_list[index]
                    ]

                df.sort_values("efficiency", inplace=True)

                df.loc[params.max_test_episode] = [
                    "Overall performance",
                    np.mean(self.data_collection_list), np.mean(self.jain_fairness_list),
                    np.mean(self.energy_comsumption_list),
                    np.mean(self.time_usage_ratio_list), np.mean(self.collection_effort_ratio_list),
                    np.mean(self.data_completion_ratio_list),
                    np.mean(self.move_aoi_list), np.mean(self.collect_aoi_list), np.mean(self.send_back_aoi_list),
                    np.mean(self.aoi_list),
                    np.mean(self.eficiency_list)
                ]

                if params.test_random:
                    df.to_csv(self.full_path + "record_performance.csv", index=0)
                else:
                    df.to_csv(params.test_path + "record_performance.csv", index=0)

    def draw_convert(self, observ, img, step, name):
        max_val = np.max(observ)
        min_val = np.min(observ)
        for i in range(80):
            for j in range(80):

                if observ[i, j] < 0:
                    img[i, j, 0] = np.uint8(255)
                if observ[i, j] > 0:
                    if max_val > 0:
                        img[i, j, 1] = np.uint8(255 * (observ[i, j] / max_val))

        full_path = os.path.join(self.full_path, 'Observ')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        save_path = full_path + '/observ_' + str(step) + name + '.png'

        # cv2.imwrite(save_path, img)

    def draw_observation(self, env, step, is_start=True):
        observ = env.get_observation()
        observ_0 = observ[0, :, :, 0]
        observ_1_1 = observ[0, :, :, 1]
        observ_1_2 = observ[1, :, :, 1]

        img_0 = np.zeros([80, 80, 2], dtype=np.uint8)
        img_1 = np.zeros([80, 80, 2], dtype=np.uint8)
        img_2_1 = np.zeros([80, 80, 2], dtype=np.uint8)
        img_2_2 = np.zeros([80, 80, 2], dtype=np.uint8)
        if is_start:
            end = 'start'
        else:
            end = 'end'
        self.draw_convert(observ_0, img_0, step, 'wall_poi' + end)
        self.draw_convert(observ_1_1, img_2_1, step, 'uav_power_1' + end)
        self.draw_convert(observ_1_2, img_2_2, step, 'uav_power_2' + end)
