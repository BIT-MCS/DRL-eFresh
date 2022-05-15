import numpy as np

poi_num=32

poi_data = [[np.random.uniform(low=0, high=1), np.random.uniform(low=0, high=1), np.random.uniform(low=0.8, high=1.2)]
            for _ in range(poi_num)]

for i, poi_i in enumerate(poi_data):
    if i == 0:
        print("[[%.10e,%.10e,%.10e]," % (poi_i[0], poi_i[1], poi_i[2]))
    elif i == poi_num-1:
        print("[%.10e,%.10e,%.10e]]" % (poi_i[0], poi_i[1], poi_i[2]))
    else:
        print("[%.10e,%.10e,%.10e]," % (poi_i[0], poi_i[1], poi_i[2]))

sum = 0.0
for i in poi_data:
    sum += i[2]

print("sum:", sum)
