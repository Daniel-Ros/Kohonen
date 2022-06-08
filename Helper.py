import numpy as np
import cv2 as cv

def get_radius_mask(size):
    rad = size + 1
    mid = size
    size = (size) * 2 + 1
    dist = [i + 1 / rad for i in np.arange(0.0, 1.0, 1 / rad)]
    dist.insert(0, 0)
    queue = [[mid, mid]]
    ret = np.zeros((size, size))
    ret[mid, mid] = dist[-1]
    while len(queue) > 0:
        pos = queue[0]
        y, x = pos

        if ret[y, x] != dist[1]:
            idx = dist.index(ret[y, x]) - 1

            if ret[y - 1, x] == 0:
                ret[y - 1, x] = dist[idx]
                queue.append([y - 1, x])

            if ret[y + 1, x] == 0:
                ret[y + 1, x] = dist[idx]
                queue.append([y + 1, x])

            if ret[y, x - 1] == 0:
                ret[y, x - 1] = dist[idx]
                queue.append([y, x - 1])

            if ret[y, x + 1] == 0:
                ret[y, x + 1] = dist[idx]
                queue.append([y, x + 1])

        queue.remove(queue[0])
    return ret


def fit_mask_position(x, y, w, h, r, circle):
    x_min = x - r
    x_max = x + r
    y_min = y - r
    y_max = y + r

    x_start = y_start = 0
    x_end = y_end = r * 2

    x_cap_min = x_cap_max = 0

    if (x_min < 0):
        x_cap_max -= x_min
        x_cap_min = 0
        x_start -= x_min
        x_min = 0



    if (x_max > w):
        x_cap_max = w
        x_cap_min = x_end + w - (x_max) + 1
        x_end += w - (x_max)
        x_max = w



    if (y_min < 0):
        y_start -= y_min
        y_min = 0

    if (y_max > h):
        y_end += h - (y_max)
        y_max = h

    return circle[y_start:y_end + 1, x_start: x_end + 1], circle[y_start:y_end + 1, x_cap_min: x_cap_max]


def generate_data_part_1(size):
    data = []
    while len(data) < size:
        x = np.random.randint(-1000, 1000) / 1000
        y = np.random.randint(-1000, 1000) / 1000
        if x**2 + y**2 < 1:
            data.append([x,y])
    return np.array(data)

def generate_data_part_2_1(size):
    data = []
    while len(data) < size/2:
        x = np.random.randint(-100, 100) / 1000
        y = np.random.randint(-100, 100) / 1000
        if x**2 + y**2 < 1:
            data.append([x,y])

    while len(data) < size:
        x = np.random.randint(-1000, 1000) / 1000
        y = np.random.randint(-1000, 1000) / 1000
        if x**2 + y**2 < 1:
            data.append([x,y])
    return np.array(data)

def generate_data_part_2_2(size):
    data = []
    while len(data) < size:
        x = np.random.randint(-1000, 1000) / 1000
        y = np.random.randint(-1000, 1000) / 1000

        dist = np.sqrt(x**2)
        rand = np.random.rand()

        if x**2 + y**2 < 1 and rand*5 < dist:
            data.append([x,y])
    return np.array(data)

def generate_data_part_3(size):
    data = []
    while len(data) < size:
        x = np.random.randint(-3000, 3000) / 1000
        y = np.random.randint(-3000, 3000) / 1000

        if 2<=x**2 + y**2 <=4:
            data.append([x,y])
    return np.array(data)

def generate_data_hand(size,fingers):
    file = f"{fingers}finger.png"
    img = cv.imread(file,0)

    data = []

    while len(data) < size:
        x = np.random.randint(0,img.shape[1])
        y = np.random.randint(0,img.shape[0])
        if img[y,x] == 0:
            data.append([x, -y])
    return np.array(data)
