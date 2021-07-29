import math
import numpy as np
import matplotlib.pyplot as plt

def show_sample_data(x_train, y_train, sample_n=25):
    sample_x = np.squeeze((x_train[:25] * 255), -1)
    sample_y = y_train[:25]

    fig = plt.figure(figsize=(8, 7))
    rows = cols = math.sqrt(sample_n)

    idx = 1
    for x, y in zip(sample_x, sample_y):
        ax = fig.add_subplot(rows, cols, idx)
        ax.imshow(x, cmap='gray')
        ax.set_xlabel("target: {}".format(str(y)))
        ax.set_xticks([]), ax.set_yticks([])
        idx += 1

    plt.show()