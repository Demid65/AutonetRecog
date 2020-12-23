import numpy as np
def avg_hue(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    r = np.take(avg_color, 2)
    g = np.take(avg_color, 1)
    b = np.take(avg_color, 0)
    v = max(r, g, b)
    m = min(r, g, b)
    # print(r,g,b)
    if v == r and g >= b:
        h = 60 * (g - b) / (v - m)
    elif v == r and g < b:
        h = 60 * (g - b) / (v - m) + 360
    elif v == g:
        h = 60 * (b - r) / (v - m) + 120
    elif v == b:
        h = 60 * (r - g) / (v - m) + 240
    return h


def hue2cid(hue):  # 0r 1b 2g 3y 4o
    # print(hue)
    if hue > 300:
        cid = 0
    elif hue < 10:
        cid = 0
    elif hue > 180 and hue < 300:
        cid = 1
    elif hue > 60 and hue < 180:
        cid = 2
    elif hue > 35 and hue < 60:
        cid = 3
    elif hue > 10 and hue < 35:
        cid = 4
    return cid
