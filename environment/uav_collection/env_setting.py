from main_setting import Params
from utils.adjusted_poi import PoI_512,PoI_256,PoI_128,PoI_64,PoI_32

params = Params()


class Setting(object):
    def __init__(self, log=None):
        self.V = {
            'GODS_PERSPECTIVE': True,
            'DEBUG_MODE': params.debug_mode,

            # --------------------------------------------------------------------------------------------------
            # 以实际量为单位
            # TODO：Add 5G base stationuav_next_pos and AOI
            'RANGE': 60,  # m
            'TRANS_SPEED': 3e8,  # m/s, too fast
            'COLLECT_SPEED': 0.4,  # Mbit/s
            'WAVE_LENGTH': 3e8 / 2.4e9,  # m, lambda=light velocity(c)/carrier frequency
            'TX_POWER': 26,  # dbm, FCC, UAV
            'NOISE_POWER': -90,  # dbm, minimal received signal power of wireless network in 802.11 variants
            'BAND_WIDTH': 1,  # MHz, 2.4GHz~2.483GHz, 80M/80 users=1M
            'MAX_POI_VALUE': 30,  # Mbit

            # Action
            # dx,dy:0-maxdistance   0,1
            # velocity:0~20m/s     2  ???
            # collect_data_AOI:0-TIMESLOT    3
            # TODO:4or3?
            'ACT_NUM': 3,

            # max factors within one time step
            'MAX_DISTANCE': 1000,  # m
            'MAX_VELOCITY': 20,  # m/s
            'TIME_SLOT': 60,  # second (max time in a step)

            # TODO:initial energy of a uav(no charging)
            'INITIAL_ENERGY': 4e5,  # J=w.s
            'DISTANCE_ENERGY_PER_T': 10.,  # W=FVT=FS
            'COLLECT_ENERGY_PER_T': 5,  # w
            'SEND_BACK_ENERGY_PER_T': 0.398,  # 10^[Tx Power(db)/10]

            # Penalty and reward
            # #TODO:1234
            'OBSTACLE_PENALTY': -1.,
            'DROPOUT_PENALTY': -1.,
            'USELESS_COLLECTION_PENALTY': -1.,
            'NO_ANY_COLLECTION_PENALTY': -1.,
            'FAIRNESS_KIND': "value",  # value,gs
            'K': 1.,
            'NORMALIZE': 0.01,
            'EPSILON': 1e-3,
            # ------------------------------------------------------------------------------------------------

            # 以unit（16x16）为单位
            # Map Info
            'MAP_X': 16,  # 16 units=1km
            'MAP_Y': 16,
            'BS_VALUE': -1.,
            'ACTUAL_SIZE': 1000,  # m->x,y
            'COLLISION_DETECT_ERROR': 1000,

            # state
            'VISIT': 1. / 1000.,

            # Observation
            'NUM_UAV': params.uav_num,
            'INIT_POSITION': [(8., 8.) for _ in range(params.uav_num)],
            'OBSER_X': 80,
            'OBSER_Y': 80,
            'OBSER_C': 2,
            'BORDER_VALUE': -1.,
            'BORDER_WIDTH': 4,

            # Specific
            # Observation 0
            'BASE_STATION': [[8, 8]],  # unit(62.5km)
            'PoI': PoI_256,  # PoI in map [x,y,value]

            'OBSTACLE': [
                [0, 4, 1, 1],
                [4, 6, 1, 1],
                [4, 12, 1, 1],
                [5, 13, 1, 1],
                [6, 5, 1, 1],
                [10, 3, 1, 1],
                [10, 13, 1, 2],
                [12, 0, 1, 1],
                [15, 11, 1, 1]
            ],  # obstacle in map [x,y,w,h]

        }
        if log is not None:
            self.LOG = log
            self.time = log.time

    def log(self):
        self.LOG.log(self.V)


if __name__ == '__main__':
    import json

    settings = Setting()
    json_str = json.dumps(settings.V)
    print(json_str)
    print(type(json_str))

    new_dict = json.loads(json_str)

    print(new_dict)
    print(type(new_dict))

    with open("record.json", "w") as f:
        json.dump(new_dict, f)

        print("加载入文件完成...")


# TODO:128 PoI

