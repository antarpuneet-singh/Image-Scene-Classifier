import numpy as np


def get_random_points(I, alpha):
    

    # -----fill in your implementation here --------

    x= np.random.randint(0,I.shape[0], size=alpha)
    y= np.random.randint(0,I.shape[1], size=alpha)

    points = np.vstack((x, y)).T

    # ----------------------------------------------
    

    return points



