
class Draw:
    def __init__(self, sg):
        self.sg = sg
        self.__width = self.sg.V['OBSER_X']  #80
        self.__height = self.sg.V['OBSER_Y'] #80
    def draw_border(self, map):
        border_value = self.sg.V['BORDER_VALUE']
        border_width = self.sg.V['BORDER_WIDTH']
        for j in range(0, 80, 1):
            for i in range(80 - border_width, 80, 1):
                self.draw_sqr(i, j, 1, 1, border_value, map)
            for i in range(0, border_width, 1):
                self.draw_sqr(i, j, 1, 1, border_value, map)
        for i in range(0, 80, 1):
            for j in range(0, border_width, 1):
                self.draw_sqr(i, j, 1, 1, border_value, map)
            for j in range(80 - border_width, 80, 1):
                self.draw_sqr(i, j, 1, 1, border_value, map)

    def get_value(self, x, y, map):
        x, y = self.__trans(x, y)
        return map[x][y]

    def __trans(self, x, y):
        return int(4 * x + 8), int(y * 4 + 8)

    def draw_obstacle(self, x, y, width, height, map):
        # self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, width * 4, height * 4, self.sg.V['BORDER_VALUE'], map)

    # def draw_chargestation(self, x, y, map):
    #     self.clear_cell(x, y, map)
    #     x, y = self.__trans(x, y)
    #     self.draw_sqr(x, y + 1, 4, 2, 1, map)
    #     self.draw_sqr(x + 1, y, 2, 4, 1, map)

    def draw_point(self, x, y, value, map):
        # self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 2, 2, value, map)

    def clear_point(self, x, y, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 2, 2, 0, map)

    def clear_uav(self, x, y, map):
        self.clear_cell(x, y, map)

    def draw_UAV(self, x, y, value, map):
        x = -1 if x < -1 else self.sg.V['MAP_X'] if x > self.sg.V['MAP_X'] else x
        y = -1 if y < -1 else self.sg.V['MAP_Y'] if y > self.sg.V['MAP_Y'] else y
        self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        # self.draw_sqr(x, y + 1, 4, 2, value, map)
        # self.draw_sqr(x + 1, y, 2, 4, value, map)
        # value = self.get_value(x, y)
        self.draw_sqr(x, y, 4, 4, value, map)

    def clear_cell(self, x, y, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 4, 4, 0, map)

    def draw_sqr(self, x, y, width, height, value, map):
        assert 0 <= x < self.__width and 0 <= y < self.__height, 'the position ({0}, {1}) is not correct.'.format(x, y)
        for i in range(x, x + width, 1):
            for j in range(y, y + height, 1):
                map[i][j] = value
