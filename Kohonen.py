import random

import matplotlib.pyplot as plt
import numpy as np
import Helper as hp


class Kohonen:
    def __init__(self, n, m, alpha, radius,name, loop_x=False):
        """
        Initialize the network with the given parameters
        :param n: height of the net
        :param m: width of the nett
        :param alpha: the initial learning rate
        :param radius: the radius of effected neurons in the neighborhood of the selected neuron
        :param name: name of the net
        :param loop_x: should we connect the last neuron with the first one
        """
        self.network = np.random.random((n, m, 3))
        self.alpha = alpha
        self.radius = radius
        self.height = n - 1
        self.width = m - 1
        self.mask = hp.get_radius_mask(radius)
        self.loop_x = loop_x
        self.name = name

    def _fit_one(self, pt, alpha):
        """
        update the network given 1 sample of the data
        :param pt: the point
        :param alpha: the current learning rate
        :return: None
        """
        pt = np.array([pt[0],pt[1],0])
        dist = np.sqrt(np.sum((self.network - pt) ** 2, axis=2))
        min_index = np.array(np.where(dist == dist.min()))
        if min_index[0].shape[0] > 1:
            min_index = min_index[0]
        min_y = max(0, int(min_index[0]) - self.radius)
        max_y = min(self.height, int(min_index[0]) + self.radius)
        min_x = max(0, int(min_index[1]) - self.radius)
        max_x = min(self.width, int(min_index[1]) + self.radius)

        mask , mask_cap = hp.fit_mask_position(int(min_index[1]), int(min_index[0]), self.width, self.height, self.radius,
                                    self.mask)

        dir = np.array(self.network[min_y:max_y + 1, min_x:max_x + 1]).reshape(-1, 3)
        dir = np.array([(pt - cell) for cell in dir]) \
            .reshape(self.network[min_y:max_y + 1, min_x:max_x + 1].shape) \
            .astype(float)
        self.network[min_y:max_y + 1, min_x:max_x + 1] += ((dir[:, :] * alpha).T * mask.T).T
        if self.loop_x:
            if max_x != int(min_index[1]) + self.radius:
                dx = int(min_index[1]) + self.radius - max_x
                dir = np.array(self.network[min_y:max_y + 1, 0:dx]).reshape(-1, 2)
                dir = np.array([(pt - cell) for cell in dir]) \
                    .reshape(self.network[min_y:max_y + 1, 0:dx].shape) \
                    .astype(float)
                self.network[min_y:max_y + 1, 0:dx] += ((dir[:, :] * alpha).T * mask_cap.T).T
            elif min_x != int(min_index[1]) - self.radius:
                dx = int(min_index[1]) - self.radius
                dir = np.array(self.network[min_y:max_y + 1, self.width + dx:self.width]).reshape(-1, 2)
                dir = np.array([(pt - cell) for cell in dir]) \
                    .reshape(self.network[min_y:max_y + 1,self.width + dx:self.width].shape) \
                    .astype(float)
                self.network[min_y:max_y + 1,self.width + dx:self.width] += ((dir[:, :] * alpha).T * mask_cap.T).T


    def fit(self, data, epochs, draw):
        """
        fit the whole data set
        :param data: the data
        :param epochs: how many times to train on the data set
        :param draw: a list of which iterations to draw
        :return:
        """
        i = 0
        dt = data
        for e in range(epochs):
            if i in draw:
                self.draw(data, i)
            alpha = self.alpha * np.exp(-i / 300)
            np.random.permutation(dt)
            for pt in dt:
                self._fit_one(pt, alpha)
            i += 1
            # print(i)
        self.draw(data,i)

    def draw(self, data, iter):
        """
        draws the network and the data in its current state
        :param data: the data
        :param iter: the iterations
        :return:
        """
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.1)
        for x in range(self.network.shape[1]):
            for y in range(self.network.shape[0]):
                if y != 0:
                    x1 = (self.network[y - 1, x, 0])
                    y1 = (self.network[y - 1, x, 1])
                    x2 = (self.network[y, x, 0])
                    y2 = (self.network[y, x, 1])
                    plt.plot([x1, x2], [y1, y2], "b-")
                if x != 0:
                    x1 = (self.network[y, x - 1, 0])
                    y1 = (self.network[y, x - 1, 1])
                    x2 = (self.network[y, x, 0])
                    y2 = (self.network[y, x, 1])
                    plt.plot([x1, x2], [y1, y2], "b-")
                if self.loop_x and x==self.width:
                    x1 = (self.network[y, 0, 0])
                    y1 = (self.network[y, 0, 1])
                    x2 = (self.network[y, x, 0])
                    y2 = (self.network[y, x, 1])
                    plt.plot([x1, x2], [y1, y2], "b-")

        ax.scatter(self.network[:, :, 0], self.network[:, :, 1], c='red')
        ax.set_title(f"iter:{iter} | radius:{self.radius} | alpha:{self.alpha} | net size:{self.network.shape[:2]}")
        fig.savefig(
            f"output/name:{self.name} | iter:{iter} | radius:{self.radius} | alpha:{self.alpha} | net size:{self.network.shape[:2]}.png")
        plt.show()



def partA(alpha , radius, epochs):
    data = hp.generate_data_part_1(1000)
    net = Kohonen(1 , 100, alpha, radius,"A1")
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 3500])

    net = Kohonen(10, 10, alpha, radius,"A1")
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 3500])

    data = hp.generate_data_part_2_1(1000)
    net = Kohonen(1, 100, alpha, radius,"A2_1")
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 3500])

    net = Kohonen(10, 10, alpha, radius,"A2_1")
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 3500])

    data = hp.generate_data_part_2_2(1000)
    net = Kohonen(1, 100, alpha, radius,"A2_2")
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 3500])

    net = Kohonen(10, 10, alpha, radius,"A2_2")
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 3500])

    data = hp.generate_data_part_3(1000)
    net = Kohonen(1, 30, alpha, radius,"A3",True)
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 3500])

def partB(alpha , radius , epochs):
    data = hp.generate_data_hand(1000,4)
    net = Kohonen(15, 15, alpha, radius, "B1")
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000,7000])
    data = hp.generate_data_hand(1000, 3)
    net.name = "B2"
    net.fit(data, epochs, [0, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000, 7000])



if __name__ == "__main__":
    for i in [0.1,0.5,1]:
        partB(i,7 , 1500 )

