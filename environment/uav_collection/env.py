from environment.uav_collection.env_setting import Setting
import numpy as np
from common.image.draw_util import *
import copy
import os


def myint(a):
    return int(np.floor(a))


class Env(object):
    def __init__(self, log):

        self.log = log
        self.sg = Setting(log)
        self.draw = Draw(self.sg)
        if self.sg.V["DEBUG_MODE"]:
            print('uav num ', self.sg.V['NUM_UAV'], 'uav pos ', self.sg.V['INIT_POSITION'], 'station num ',
                  str(1), 'station pos ', self.sg.V['BASE_STATION'])

        # make save directory
        self.sg.log()
        self.log_dir = log.full_path

        # [ 16 , 16 ]
        self.map_size_x = self.sg.V['MAP_X']  # 16 unit
        self.map_size_y = self.sg.V['MAP_Y']  # 16 unit

        # map obstacle [ 16 , 16 ]
        self.map_obstacle = np.zeros((self.map_size_x, self.map_size_y)).astype(np.int8)
        self.map_obstacle_value = 1
        self.init_map_obstacle()

        # data collecting range
        self.c_range = self.sg.V['RANGE']  # m

        # uav energy
        self.init_uav_energy = np.asarray(
            [self.sg.V['INITIAL_ENERGY']] * self.sg.V['NUM_UAV'],
            dtype=np.float32
        )
        # num of uavs
        self.uav_num = self.sg.V['NUM_UAV']

        # action  [ K , 3 ]
        self.action = np.zeros(
            shape=[self.sg.V['NUM_UAV'],
                   self.sg.V['ACT_NUM']],
            dtype=np.float32
        )

        # PoI
        self.poi_data_pos, self.init_poi_data_val = self.filter_PoI_data()
        self.bs_pos = self.filter_bs_pos()
        # poi_data_pos->[256,2]  poi_data_val->[256]

        # for render
        self.max_uav_energy = self.sg.V['INITIAL_ENERGY'] / self.sg.V['INITIAL_ENERGY']
        self.max_distance = self.sg.V["MAX_DISTANCE"]

        self.reset()

        self.action_space = self.sg.V['NUM_UAV'] * self.sg.V['ACT_NUM']


    def reset(self):

        self.observ = None  # TODO:observation  [ K , 80 , 80 , 2 ]
        self.observ_0 = None  # Border Obstacle PoI
        self.init_observation()

        self.uav_trace = [[] for i in range(self.uav_num)]
        self.uav_state = [[] for i in range(self.uav_num)]
        # TODO: change initial position
        self.cur_uav_pos = np.copy(self.sg.V['INIT_POSITION'])
        self.cur_bs_pos = np.copy(self.sg.V['BASE_STATION'])
        for i in range(self.uav_num):
            self.uav_trace[i].append(copy.deepcopy(self.cur_uav_pos[i]))
            self.uav_state[i].append(1)  # 0th step->collect

        self.cur_poi_data_val = np.copy(self.init_poi_data_val)
        # poi_data_val->[256]

        self.cur_uav_energy_list = copy.deepcopy(self.init_uav_energy)

        self.uav_energy_consuming_list = np.zeros(shape=[self.uav_num])
        self.uav_aoi_list = np.zeros([self.uav_num])
        self.uav_snr_list = np.zeros([self.uav_num])
        self.data_completion_ratio_list = np.zeros([self.uav_num])
        self.time_usage_ratio_list = np.zeros([self.uav_num])
        self.collection_effort_ratio_list = np.zeros([self.uav_num])
        # self.uav_energy_charging = np.zeros(shape=[self.uav_num])
        # self.charge_counter = np.zeros(shape=[self.uav_num], dtype=np.int)
        # [K]

        self.dead_uav_list = [False] * self.uav_num

        self.total_collected_data = 0.
        self.episodic_aoi_list = []
        self.episodic_total_uav_reward = 0.

        self.episodic_time_usage_ratio_list = [[] for _ in range(self.uav_num)]
        self.episodic_collection_effort_ratio_list = [[] for _ in range(self.uav_num)]
        self.episodic_data_completion_ratio_list = [[] for _ in range(self.uav_num)]

        self.bs_list=[0,0,0,0,0,0,0,0]
        self.bs_aoi_list = [0, 0, 0, 0, 0, 0, 0, 0]
        self.bs_move_list = [0, 0, 0, 0, 0, 0, 0, 0]
        self.bs_collect_list = [0, 0, 0, 0, 0, 0, 0, 0]
        self.bs_send_list = [0, 0, 0, 0, 0, 0, 0, 0]

        return self.get_observation(), np.array(self.uav_aoi_list), np.array(self.uav_snr_list), np.array(
            self.data_completion_ratio_list), np.array(self.collection_effort_ratio_list)

    def init_map_obstacle(self):
        obs = self.sg.V['OBSTACLE']
        # self.pos_obstacle_dict = {}
        # draw obstacles in map_obstacle [16 , 16]    the obstacle is 1 , others is 0
        for i in obs:
            # self.pos_obstacle_dict[(i[0], i[1])] = i
            for x in range(i[0], i[0] + i[2], 1):
                for y in range(i[1], i[1] + i[3], 1):
                    self.map_obstacle[x][y] = self.map_obstacle_value

    def filter_PoI_data(self):

        PoI = np.reshape(self.sg.V['PoI'], (-1, 3)).astype(np.float32)

        # TODO:replace the PoI in obstacle position with the PoI out of obstacle position
        # #---------------------------------------------------------------------------------
        # print("crazy PoIs adjustment\n")
        # dx = [-0.2, -0.2, -0.2, 0, 0, 0, 0.2, 0.2, 0.2]
        # dy = [-0.2, 0, 0.2, -0.2, 0, 0.2, -0.2, 0, 0.2]
        # # replace the POI in obstacle position with the POI out of obstacle position
        # for index in range(PoI.shape[0]):
        #     need_adjust = True
        #     while need_adjust:
        #         need_adjust=False
        #         for i in range(len(dx)):
        #             if self.map_obstacle[min(myint(PoI[index][0] * self.map_size_x + dx[i]), self.map_size_x - 1)][
        #                 min(myint(PoI[index][1] * self.map_size_y + dy[i]), self.map_size_y - 1)] == self.map_obstacle_value:
        #                 need_adjust=True
        #                 break
        #         if need_adjust is True:
        #             print("change initial POI!!!")
        #             PoI[index] = [np.random.uniform(low=0, high=1), np.random.uniform(low=0, high=1), np.random.uniform(low=0.8, high=1.2)]
        #
        #
        # for i, poi_i in enumerate(PoI):
        #     if i == 0:
        #         print("[[%.10e,%.10e,%.10e]," % (poi_i[0], poi_i[1], poi_i[2]))
        #     elif i == PoI.shape[0]-1:
        #         print("[%.10e,%.10e,%.10e]]\n" % (poi_i[0], poi_i[1], poi_i[2]))
        #     else:
        #         print("[%.10e,%.10e,%.10e]," % (poi_i[0], poi_i[1], poi_i[2]))
        # exit(0)
        # # ---------------------------------------------------------------------------------

        # PoI data value [ 256]  核算实际量纲
        poi_data_val = np.copy(PoI[:, 2]) * self.sg.V["MAX_POI_VALUE"]

        # PoI data Position  [ 256 , 2 ]
        poi_data_pos = PoI[:, 0:2] * self.map_size_x

        # sum of all PoI data values
        self.totaldata = np.sum(poi_data_val)
        if self.sg.V["DEBUG_MODE"]:
            print("totaldata:", self.totaldata)
        self.log.log(PoI)

        return poi_data_pos, poi_data_val

    # Add num of bs == 1
    def filter_bs_pos(self):
        bs = np.asarray(self.sg.V['BASE_STATION'])
        #
        for index in range(1):
            while self.map_obstacle[myint(bs[index, 0])][myint(bs[index, 1])] == self.map_obstacle_value:
                bs[index, :] = np.random.rand(2)
                print("base position has been reseted, error in memory")
                exit(0)
        # Power Position  [ 50 , 2 ]
        bs_pos = bs

        return bs_pos

    def init_observation(self):
        # observation  [ 80 , 80 , 3 ]
        self.observ = np.zeros(
            shape=[self.sg.V['OBSER_X'],
                   self.sg.V['OBSER_Y'],
                   self.sg.V['OBSER_C']],
            dtype=np.float32
        )

        self.observ_0 = np.zeros([self.sg.V['OBSER_X'], self.sg.V['OBSER_Y']], dtype=np.float32)  # Border Obstacle PoI

        # empty wall  ----layer1
        # draw walls in the border of the map (self._image_data)
        # the value of the wall is -1
        # the width of the wall is 4, which can be customized in image/flag.py
        # after adding four wall borders, the shape of the map is still [80,80]
        self.draw.draw_border(self.observ_0)

        if self.sg.V['GODS_PERSPECTIVE']:
            obs = self.sg.V['OBSTACLE']
            for ob in obs:
                self.draw.draw_obstacle(x=ob[0], y=ob[1], width=ob[2], height=ob[3], map=self.observ_0)

            for index in range(self.poi_data_pos.shape[0]):
                self.draw.draw_point(x=self.poi_data_pos[index, 0], y=self.poi_data_pos[index, 1],
                                     value=self.init_poi_data_val[index], map=self.observ_0)

        self.observ[:, :, 0] = copy.deepcopy(self.observ_0)

        if self.sg.V['GODS_PERSPECTIVE']:
            self.draw_base_station(self.observ[:, :, 1])

        # loop of uav
        for i in range(self.uav_num):
            # draw uav
            # draw uavs in the map (self._image_position[i_n], i_n is the id of uav)
            # the position of uav is [x*4+8,y*4+8] of the [80,80] map,
            # where x,y->[0~15]
            # the size of uav is [4,4]
            # the value of uav is 1.

            self.draw.draw_UAV(self.sg.V['INIT_POSITION'][i][0], self.sg.V['INIT_POSITION'][i][1],
                               self.sg.V['INITIAL_ENERGY'] / self.sg.V['INITIAL_ENERGY'],
                               self.observ[:, :, 1])

    #
    def draw_base_station(self, map):
        for bs in self.bs_pos:
            self.draw.draw_point(x=bs[0], y=bs[1], value=self.sg.V['BS_VALUE'],
                                 map=map)

    def cal_distance(self, pos1, pos2):
        distance_unit = np.sqrt(
            np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2)
        )
        return distance_unit * self.sg.V['ACTUAL_SIZE'] / self.sg.V['MAP_X']

    def cal_energy_consuming(self, move_aoi=0, collect_aoi=0, sendback_aoi=0):
        energy_of_dis = move_aoi * self.sg.V['DISTANCE_ENERGY_PER_T']
        energy_of_poi = collect_aoi * self.sg.V['COLLECT_ENERGY_PER_T']
        energy_of_sendback = sendback_aoi * self.sg.V['SEND_BACK_ENERGY_PER_T']

        return energy_of_poi + energy_of_dis + energy_of_sendback

    def cal_move_aoi(self, distance, velocity):
        return min(distance / (velocity + self.sg.V["EPSILON"]), self.sg.V['TIME_SLOT'])

    def cal_collect_aoi(self, i, move_aoi):
        collect_aoi = min(self.action[i][2], self.sg.V['TIME_SLOT'] - move_aoi)

        if self.sg.V["DEBUG_MODE"]:
            print("move_aoi:", move_aoi, ", collect_aoi:", collect_aoi)
            print("move_energy:", self.cal_energy_consuming(move_aoi=move_aoi),
                  " collect_energy:", self.cal_energy_consuming(collect_aoi=collect_aoi))
        return collect_aoi

    def cal_signal_noise_ratio(self, bs_distance):

        # self.bs_list[int(bs_distance/100)]+=1
        # print(self.bs_list)

        # Field Space Path Loss (db)
        fspl = 20 * np.log10(4 * np.pi * bs_distance / self.sg.V["WAVE_LENGTH"] + self.sg.V["EPSILON"])

        # Rx Power in dbm
        rx_power = self.sg.V["TX_POWER"] - fspl - 30

        # Singal-Noise-Ratio(SNR),db
        SNR = rx_power - self.sg.V["NOISE_POWER"]

        # Shannon Capacity, Mbit/s
        Capacity = self.sg.V["BAND_WIDTH"] * np.log2(1 + 10 ** (SNR / 10))

        if self.sg.V["DEBUG_MODE"]:
            print("bs_distance:", bs_distance, ", SNR:", SNR, ", shannon capacity:", Capacity)

        return SNR, Capacity

    def cal_send_back_aoi(self, exploited_data, Capacity):

        send_back_aoi = exploited_data / Capacity
        if self.sg.V["DEBUG_MODE"]:
            print("exploited_data:", exploited_data, ", Capacity:", Capacity, ", send_back_aoi:", send_back_aoi,
                  ", send_back_energy:", self.cal_energy_consuming(sendback_aoi=send_back_aoi))

        return send_back_aoi

    def cal_uav_next_pos(self, i, cur_pos, action):
        dx = action[0]
        dy = action[1]
        velocity = self.sg.V['MAX_VELOCITY']
        move_distance = np.sqrt(np.power(dx, 2) + np.power(dy, 2)) * self.sg.V['ACTUAL_SIZE'] / self.sg.V['MAP_X']
        try_move_aoi = self.cal_move_aoi(move_distance, velocity)
        energy_consuming = self.cal_energy_consuming(move_aoi=try_move_aoi)

        # move distance is larger than max distance
        if move_distance > self.max_distance:
            dx = action[0] * (self.max_distance / move_distance)
            dy = action[1] * (self.max_distance / move_distance)
            energy_consuming = energy_consuming * (self.max_distance / move_distance)

        # energy is enough
        if self.cur_uav_energy_list[i] >= energy_consuming:
            new_x = cur_pos[0] + dx
            new_y = cur_pos[1] + dy
        else:
            # energy is not enough
            new_x = cur_pos[0] + dx * (self.cur_uav_energy_list[i] / energy_consuming)
            new_y = cur_pos[1] + dy * (self.cur_uav_energy_list[i] / energy_consuming)

        return (new_x, new_y), velocity

    def judge_obstacle(self, next_pos, cur_pos):
        if 0 <= next_pos[0] < (self.map_size_x - 0.1) and 0 <= next_pos[1] < (self.map_size_y - 0.1):
            dx = next_pos[0] - cur_pos[0]
            dy = next_pos[1] - cur_pos[1]
            acc_range = self.sg.V["COLLISION_DETECT_ERROR"]
            for i in range(0, acc_range + 1):
                tmp_pos_x = myint(cur_pos[0] + i * dx / acc_range)
                tmp_pos_y = myint(cur_pos[1] + i * dy / acc_range)
                if self.map_obstacle[tmp_pos_x][tmp_pos_y] == self.map_obstacle_value:
                    if self.sg.V["DEBUG_MODE"]:
                        print("!!!collision!!!")
                    return True

            return False
        else:
            if self.sg.V["DEBUG_MODE"]:
                print("!!!collision!!!")
            return True

    def update_observ_0(self, pos, pos_val=None):
        # update poi
        self.draw.draw_point(x=pos[0], y=pos[1], value=pos_val, map=self.observ_0)

    def update_observ_1(self, i, cur_pos, pre_pos):
        self.draw.clear_uav(pre_pos[0], pre_pos[1], self.observ[:, :, 1])
        self.draw.draw_UAV(cur_pos[0], cur_pos[1], self.cur_uav_energy_list[i] / self.sg.V['INITIAL_ENERGY'],
                           self.observ[:, :, 1])

        if self.sg.V['GODS_PERSPECTIVE']:
            self.draw_base_station(self.observ[:, :, 1])

    def is_uav_out_of_energy(self, i):
        return self.cur_uav_energy_list[i] < self.sg.V['EPSILON']

    def use_energy(self, i, energy_consuming):
        # update uav energy
        self.uav_energy_consuming_list[i] += min(energy_consuming, self.cur_uav_energy_list[i])
        self.cur_uav_energy_list[i] = max(self.cur_uav_energy_list[i] - energy_consuming, 0)
        # update uav dead list
        if self.is_uav_out_of_energy(i):
            if self.sg.V["DEBUG_MODE"]:
                print("Energy should not run out!")
            self.dead_uav_list[i] = True

    def collect_and_sendback(self, i, uav_pos, move_aoi, collect_aoi, greedy_mode):

        select_poi_index_list = []
        exist_poi_index_list = []
        exist_poi_value_list = [0]
        select_poi_value_list = [0]
        exploited_data_list = []
        bs_distance = self.cal_distance(self.cur_bs_pos[0], uav_pos)
        SNR, Shannon_Capacity = self.cal_signal_noise_ratio(bs_distance)
        self.uav_snr_list[i] = SNR

        for poi_index, (poi_pos, poi_val) in enumerate(zip(self.poi_data_pos, self.cur_poi_data_val)):
            # this poi has no data left
            if self.cur_poi_data_val[poi_index] <= self.sg.V['EPSILON']:
                continue

            # cal distance between uav and poi
            distance = self.cal_distance(uav_pos, poi_pos)

            # bs_distance = self.cal_distance(self.cur_bs_pos[0],poi_pos)
            # self.bs_list[int(bs_distance/100)]+=1

            # if distance is within collecting range
            if distance <= self.sg.V['RANGE']:
                exist_poi_index_list.append(poi_index)
                exist_poi_value_list.append(self.cur_poi_data_val[poi_index])
                if collect_aoi > 0 or greedy_mode:
                    select_poi_index_list.append(poi_index)
                    select_poi_value_list.append(self.cur_poi_data_val[poi_index])

        # print(self.bs_list)
        # exit(0)

        num_select_poi = len(select_poi_index_list)
        collect_speed_per_poi = self.sg.V['COLLECT_SPEED'] / (num_select_poi + self.sg.V["EPSILON"])
        suppose_collect_aoi = np.max(select_poi_value_list) / collect_speed_per_poi

        if self.sg.V["DEBUG_MODE"]:
            print("exist %d PoIs" % len(exist_poi_index_list), "exist value=%f~" % np.sum(exist_poi_value_list))
            print("exploit %d PoIs" % num_select_poi, "suppose collect aoi=%f~" % suppose_collect_aoi)

        if num_select_poi > 0:
            # TODO:greedy tc
            if greedy_mode:
                Shannon_Capacity_per_poi = Shannon_Capacity / num_select_poi
                exploited_maximum_per_poi = min((self.sg.V["TIME_SLOT"] - move_aoi) / (
                        (1 / collect_speed_per_poi) + (1 / Shannon_Capacity_per_poi)),
                                                np.max(self.cur_poi_data_val[select_poi_index_list]))

                collect_aoi = exploited_maximum_per_poi / collect_speed_per_poi
                if self.sg.V["DEBUG_MODE"]:
                    print("///optimal collect_aoi=%f" % collect_aoi)

            exploited_maximum_per_poi = collect_speed_per_poi * collect_aoi
            collection_effort_ratio = min(suppose_collect_aoi / (collect_aoi + self.sg.V["EPSILON"]), 1.0)

            total_exploited_data_uav = 0
            for select_poi_index in select_poi_index_list:
                # try to collect data by sensors
                exploited_data = min(exploited_maximum_per_poi, self.cur_poi_data_val[select_poi_index])
                exploited_data_list.append(exploited_data)
                total_exploited_data_uav += exploited_data

            # send back or drop (1 base station)
            send_back_aoi = self.cal_send_back_aoi(total_exploited_data_uav, Shannon_Capacity)
            total_aoi = move_aoi + collect_aoi + send_back_aoi
            suppose_total_aoi = np.copy(total_aoi)
            time_usage_ratio = min((self.sg.V["TIME_SLOT"] - move_aoi) / (collect_aoi + send_back_aoi), 1.0)
            data_completion_ratio = min((self.sg.V["TIME_SLOT"] - move_aoi - collect_aoi) / send_back_aoi, 1.0)

            if data_completion_ratio < 1.0:
                if self.sg.V["DEBUG_MODE"]:
                    print("Time usage ratio=%f, data_completion_ratio=%f" % (time_usage_ratio, data_completion_ratio))
                # ---------------------PENALTY TWO: data dropout penalty --------------------------------------
                dropout_penalty = self.sg.V['DROPOUT_PENALTY'] * (1 - data_completion_ratio)
                self.uav_penalty_list[i] += dropout_penalty
                total_aoi = self.sg.V["TIME_SLOT"]
            else:
                if self.sg.V["DEBUG_MODE"]:
                    print("!!!Fully Collected!!!")

        else:
            if greedy_mode:
                collect_aoi = 0
                if self.sg.V["DEBUG_MODE"]:
                    print("///optimal collect_aoi=%f" % collect_aoi)

            send_back_aoi = 0
            total_aoi = move_aoi + collect_aoi + send_back_aoi
            suppose_total_aoi = np.copy(total_aoi)
            time_usage_ratio = 1
            data_completion_ratio = 1

            if len(exist_poi_index_list) == 0 and collect_aoi > 0:
                # ---------------------PENALTY THREE: no data but collection penalty --------------------------------
                useless_collection_penalty = self.sg.V['USELESS_COLLECTION_PENALTY']
                self.uav_penalty_list[i] += useless_collection_penalty
                if self.sg.V["DEBUG_MODE"]:
                    print("!!!Exploit no data but still collect!!!")
                collection_effort_ratio = 0
            elif len(exist_poi_index_list) > 0 and collect_aoi == 0:
                # # ---------------------PENALTY THREE: exist data but no collection penalty --------------------------------
                # useless_collection_penalty = self.sg.V['USELESS_COLLECTION_PENALTY']
                # self.uav_penalty_list[i] += useless_collection_penalty
                if self.sg.V["DEBUG_MODE"]:
                    print("!!!Exist some data but still no collect!!!")
                collection_effort_ratio = 1
            else:
                collection_effort_ratio = 1

        suppose_collect_aoi = np.copy(collect_aoi)
        suppose_send_back_aoi = np.copy(send_back_aoi)
        suppose_collect_energy_consuming = self.cal_energy_consuming(collect_aoi=suppose_collect_aoi)
        suppose_send_back_energy_consuming = self.cal_energy_consuming(sendback_aoi=suppose_send_back_aoi)

        send_back_aoi = send_back_aoi * data_completion_ratio
        collect_energy_consuming = self.cal_energy_consuming(collect_aoi=collect_aoi)
        self.use_energy(i=i, energy_consuming=collect_energy_consuming)
        send_back_energy_consuming = self.cal_energy_consuming(sendback_aoi=send_back_aoi)
        self.use_energy(i=i, energy_consuming=send_back_energy_consuming)
        self.collect_aoi_list[i] = collect_aoi
        self.send_back_aoi_list[i] = send_back_aoi
        self.time_usage_ratio_list[i] = time_usage_ratio
        self.collection_effort_ratio_list[i] = collection_effort_ratio
        self.data_completion_ratio_list[i] = data_completion_ratio

        # # TODO
        # self.bs_list[int(bs_distance / 100)] += 1
        # self.bs_move_list[int(bs_distance / 100)] += move_aoi
        # self.bs_collect_list[int(bs_distance / 100)] += collect_aoi
        # self.bs_send_list[int(bs_distance / 100)] += send_back_aoi
        # self.bs_aoi_list[int(bs_distance / 100)] += total_aoi
        #
        # print("\n")
        # print(self.bs_list)
        # print(self.bs_move_list)
        # print(self.bs_collect_list)
        # print(self.bs_send_list)
        # print(self.bs_aoi_list)

        self.episodic_data_completion_ratio_list[i].append(data_completion_ratio)
        self.episodic_time_usage_ratio_list[i].append(time_usage_ratio)
        self.episodic_collection_effort_ratio_list[i].append(collection_effort_ratio)

        total_collected_data_uav = 0
        for list_index, select_poi_index in enumerate(select_poi_index_list):
            if self.sg.V['DEBUG_MODE']:
                print("select_poi_data_val_%d:" % select_poi_index, self.cur_poi_data_val[select_poi_index])

            # update current poi data value
            collect_data = exploited_data_list[list_index] * data_completion_ratio
            raw = np.copy(self.cur_poi_data_val[select_poi_index])
            self.cur_poi_data_val[select_poi_index] -= collect_data
            self.cur_poi_data_val[select_poi_index] = max(0, self.cur_poi_data_val[select_poi_index])
            self.update_observ_0(
                pos=self.poi_data_pos[select_poi_index],
                pos_val=self.cur_poi_data_val[select_poi_index]
            )
            total_collected_data_uav += (raw - self.cur_poi_data_val[select_poi_index])

            if self.sg.V['DEBUG_MODE']:
                print("collected_poi_data_val_%d:" % select_poi_index, self.cur_poi_data_val[select_poi_index])

        self.step_uav_data_collection[i] += total_collected_data_uav
        self.total_collected_data += total_collected_data_uav

        return total_aoi, suppose_total_aoi, suppose_collect_energy_consuming, suppose_send_back_energy_consuming

    def get_observation(self):
        self.observ[:, :, 0] = self.observ_0[:, :]

        return copy.deepcopy(self.observ)  # [W,H,C]

    def get_observation_trans(self):
        self.observ[:, :, 0] = self.observ_0[:, :]

        return copy.deepcopy(self.observ).transpose([2, 0, 1])  # C,W,H

    def step(self, action, current_step=None, greedy_mode=False):
        if self.sg.V["DEBUG_MODE"]:
            print("\n************* STEP %d" % current_step, " ****************")
        last_observation = self.get_observation()
        # action [K,3]
        uavs_action = np.reshape(np.copy(action), [self.uav_num, -1])
        for i in range(self.uav_num):
            self.action[i, 0] = np.clip(uavs_action[i, 0], -1 * self.sg.V['MAP_X'], self.sg.V['MAP_X'])
            self.action[i, 1] = np.clip(uavs_action[i, 1], -1 * self.sg.V['MAP_Y'], self.sg.V['MAP_Y'])
            self.action[i, 2] = np.clip(uavs_action[i, 2], 0, 1) * self.sg.V['TIME_SLOT']

        self.uav_reward_list = np.zeros([self.uav_num])
        self.uav_aoi_list = np.zeros([self.uav_num])
        self.suppose_uav_aoi_list = np.zeros([self.uav_num])
        self.uav_snr_list = np.zeros([self.uav_num])
        self.uav_penalty_list = np.zeros([self.uav_num])

        self.step_uav_data_collection = np.zeros([self.uav_num])
        self.move_aoi_list = np.zeros([self.uav_num])
        self.collect_aoi_list = np.zeros([self.uav_num])
        self.send_back_aoi_list = np.zeros([self.uav_num])
        self.time_usage_ratio_list = np.ones([self.uav_num])
        self.collection_effort_ratio_list = np.ones([self.uav_num])
        self.data_completion_ratio_list = np.ones([self.uav_num])

        # loop of uavupdate_observ_0
        for i in range(self.uav_num):

            # skip the uav which runs out of energy
            if self.dead_uav_list[i]:
                continue

            uav_energy_consuming_list_pre = self.uav_energy_consuming_list[i]
            # cal uav next position
            next_uav_pos, velocity = self.cal_uav_next_pos(i, self.cur_uav_pos[i], self.action[i])
            move_distance = self.cal_distance(self.cur_uav_pos[i], next_uav_pos)
            if self.sg.V["DEBUG_MODE"]:
                print("\nuav:", i, ", delta_x:", self.action[i][0], ", delta_y:", self.action[i][1], ", v:",
                      velocity,
                      ", col_aoi:", self.action[i][2])
                print("cur_pos:", self.cur_uav_pos[i], ", next_pos:", next_uav_pos, ", dis:", move_distance)

            # ------------------------PENALTY ONE: if obstacle is in next position----------------------
            if self.judge_obstacle(next_uav_pos, self.cur_uav_pos[i]):
                # cal move distance
                # add obstacle penalty
                collision_penalty = self.sg.V['OBSTACLE_PENALTY']
                self.uav_penalty_list[i] += collision_penalty

                # stand still
                next_uav_pos = copy.deepcopy(self.cur_uav_pos[i])

            # update uav current position
            pre_pos = copy.deepcopy(self.cur_uav_pos[i])
            self.cur_uav_pos[i] = copy.deepcopy(next_uav_pos)

            # add uav current position to trace
            self.uav_trace[i].append(copy.deepcopy(self.cur_uav_pos[i]))

            # update current uav energy
            move_aoi = self.cal_move_aoi(move_distance, velocity)
            move_energy_consuming = self.cal_energy_consuming(move_aoi=move_aoi)
            self.move_aoi_list[i] = move_aoi
            self.use_energy(i=i, energy_consuming=move_energy_consuming)

            # judge whether a uav is out of energy
            if self.is_uav_out_of_energy(i):
                self.dead_uav_list[i] = True
                self.uav_state[i].append(int(0))  # 0-> uav out of energy
                continue

            # collect data and send back
            collect_aoi = self.cal_collect_aoi(i, move_aoi)
            self.uav_state[i].append(1)  # uav collects data

            uav_total_aoi, suppose_total_aoi, suppose_collect_energy_consuming, suppose_send_back_energy_consuming \
                = self.collect_and_sendback(i, self.cur_uav_pos[i], move_aoi, collect_aoi, greedy_mode)

            # update uav energy and pos
            self.update_observ_1(i=i, cur_pos=self.cur_uav_pos[i], pre_pos=pre_pos)

            # cal reward
            total_energy_consuming = self.uav_energy_consuming_list[i] - uav_energy_consuming_list_pre
            suppose_total_energy_consuming = move_energy_consuming + suppose_collect_energy_consuming + suppose_send_back_energy_consuming

            if self.step_uav_data_collection[i] > 0:
                uav_reward = self.collection_effort_ratio_list[i] * self.time_usage_ratio_list[i] * \
                             self.step_uav_data_collection[i] / (total_energy_consuming * self.sg.V['NORMALIZE'])
            else:
                uav_reward = 0
                # ---------------------PENALTY FOUR: no any collection penalty --------------------------------
                no_any_collection_penalty = self.sg.V['NO_ANY_COLLECTION_PENALTY'] * (1 - self.data_collection_ratio())
                self.uav_penalty_list[i] += no_any_collection_penalty

            if self.sg.V["DEBUG_MODE"]:
                print("step_uav_data_collection:", self.step_uav_data_collection[i],
                      ", total_energy_consuming:", total_energy_consuming,
                      ", suppose_total_energy_consuming:", suppose_total_energy_consuming,
                      ", collection_effort_ratio:", self.collection_effort_ratio_list[i],
                      ", \n uav_total_aoi:", uav_total_aoi,
                      ', suppose_total_aoi', suppose_total_aoi,
                      ", uav_reward:", uav_reward,
                      ", penalty:", self.uav_penalty_list[i])

            self.uav_reward_list[i] = uav_reward
            self.uav_aoi_list[i] = uav_total_aoi
            self.suppose_uav_aoi_list[i] = suppose_total_aoi

        # calculate reward
        # TODO:20200308
        fairness = self.geographical_fairness()
        e_cur = self.cal_left_energy_ratio()

        total_uav_reward = fairness * np.mean(self.uav_reward_list) + np.mean(self.uav_penalty_list)

        if self.sg.V["DEBUG_MODE"]:
            print("\n---total_uav_reward:", total_uav_reward, ", e_cur:", e_cur, ", fairness:", fairness, ", penalty:",
                  self.uav_penalty_list, ", efficiency:", self.time_energy_efficiency())

        self.episodic_total_uav_reward += total_uav_reward
        self.episodic_aoi_list.append(np.max(self.uav_aoi_list))

        if self.sg.V["DEBUG_MODE"]:
            print("---total_aoi:", self.episodic_aoi_list[-1],
                  " time_usage_ratio:", self.time_usage_ratio_list,
                  " collection_effort_ratio:", self.collection_effort_ratio_list,
                  " data_completion_ratio:", self.data_completion_ratio_list)
        # print total_uav_reward,'before clip'
        # total_uav_reward = np.clip(np.array(total_uav_reward) / self.sg.V['NORMALIZE'], -2., 1.)
        # print total_uav_reward,'after clip'

        # done = False if False in self.dead_uav_list else True
        done = False

        observation = self.get_observation()

        return observation, total_uav_reward, done, np.array(self.suppose_uav_aoi_list), np.array(
            self.uav_snr_list), np.array(self.data_completion_ratio_list), np.array(self.collection_effort_ratio_list)

    def get_uav_pos(self):
        return self.cur_uav_pos

    def get_jain_fairness(self):
        collection = np.copy(self.init_poi_data_val - self.cur_poi_data_val)
        for index, collect_i in enumerate(collection):
            collection[index] = collect_i / self.init_poi_data_val[index]
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        if sum_of_square < 1e-4:
            return 0.
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness

    # Not work
    def get_gs_fairness(self):
        remain = np.copy(self.cur_poi_data_val)
        x = np.copy(remain)
        k_1 = 1.0 / self.sg.V["K"]
        max_x = np.max(x)
        if max_x == 0:
            return 0
        else:
            x = np.clip(a=x, a_min=0.1, a_max=max_x)
            g_x = np.sin(pow(np.pi * x / (2 * max_x), k_1))
            # print(g_x)
            return np.min(g_x)

    def cal_left_energy_ratio(self):
        return np.sum(self.cur_uav_energy_list) / np.sum(self.init_uav_energy)

    def data_collection_ratio(self):  # successfully send back
        return self.total_collected_data / self.totaldata

    def energy_consumption_ratio(self):
        return np.sum(self.uav_energy_consuming_list) / (np.sum(self.init_uav_energy))

    # TODO
    def geographical_fairness(self):
        # calculate reward
        # TODO:20200308
        if self.sg.V['FAIRNESS_KIND'] == "value":
            return self.get_jain_fairness()
        elif self.sg.V['FAIRNESS_KIND'] == "gs":
            return self.get_gs_fairness()
        else:
            print("error")
            exit(0)

    def age_of_information(self):  # /s
        if len(self.episodic_aoi_list) == 0:
            return 0
        else:
            return np.mean(self.episodic_aoi_list)

    def time_energy_efficiency(self):
        if self.energy_consumption_ratio() > 0:
            return self.geographical_fairness() * self.data_collection_ratio() / self.energy_consumption_ratio()
            # return self.collection_effort_ratio() * self.data_completion_ratio() * self.geographical_fairness() * \
            #        self.data_collection_ratio() / self.energy_consumption_ratio()
        else:
            return 0

    # TODO:0319
    def collection_effort_ratio(self):  # /s
        result = []
        for i in range(self.uav_num):
            if len(self.episodic_collection_effort_ratio_list[i]) == 0:
                result.append(0)
            else:
                result.append(np.mean(self.episodic_collection_effort_ratio_list[i]))

        return np.mean(result)

    def collection_effort_ratio_min_max_std(self):  # /s
        result_min = []
        result_max = []
        result_std = []
        for i in range(self.uav_num):
            if len(self.episodic_collection_effort_ratio_list[i]) == 0:
                result_min.append(0)
                result_max.append(0)
                result_std.append(0)
            else:
                result_min.append(np.min(self.episodic_collection_effort_ratio_list[i]))
                result_max.append(np.max(self.episodic_collection_effort_ratio_list[i]))
                result_std.append(np.std(self.episodic_collection_effort_ratio_list[i]))

        return np.mean(result_min), np.mean(result_max), np.mean(result_std)

    def data_completion_ratio(self):  # /s
        result = []
        for i in range(self.uav_num):
            if len(self.episodic_data_completion_ratio_list[i]) == 0:
                result.append(0)
            else:
                result.append(np.mean(self.episodic_data_completion_ratio_list[i]))

        return np.mean(result)

    def data_completion_ratio_min_max_std(self):  # /s
        result_min = []
        result_max = []
        result_std = []
        for i in range(self.uav_num):
            if len(self.episodic_data_completion_ratio_list[i]) == 0:
                result_min.append(0)
                result_max.append(0)
                result_std.append(0)
            else:
                result_min.append(np.min(self.episodic_data_completion_ratio_list[i]))
                result_max.append(np.max(self.episodic_data_completion_ratio_list[i]))
                result_std.append(np.std(self.episodic_data_completion_ratio_list[i]))

        return np.mean(result_min), np.mean(result_max), np.mean(result_std)

    def time_usage_ratio(self):  # /s
        result = []
        for i in range(self.uav_num):
            if len(self.episodic_time_usage_ratio_list[i]) == 0:
                result.append(0)
            else:
                result.append(np.mean(self.episodic_time_usage_ratio_list[i]))

        return np.mean(result)

    def time_usage_ratio_min_max_std(self):  # /s
        result_min = []
        result_max = []
        result_std = []
        for i in range(self.uav_num):
            if len(self.episodic_time_usage_ratio_list[i]) == 0:
                result_min.append(0)
                result_max.append(0)
                result_std.append(0)
            else:
                result_min.append(np.min(self.episodic_time_usage_ratio_list[i]))
                result_max.append(np.max(self.episodic_time_usage_ratio_list[i]))
                result_std.append(np.std(self.episodic_time_usage_ratio_list[i]))

        return np.mean(result_min), np.mean(result_max), np.mean(result_std)
