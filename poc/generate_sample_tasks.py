import numpy as np


class generate_sample_tasks:
    def __init__(self):
        pass

    def create_task(self):

        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        y = np.sin(20*x) / 2 - ((10 - 20*x) ** 2) / 50 + 2
        y /= np.max(y)

        return x, y


