from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import numpy as np


def closest(lst, K):
    """
    Finds the closest value in list
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


class PointPicker(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.mouseX = []
        self.mouseY = []
        self.events = []
        self.cutoffsI = []
        self.points = []
        self.idx = 0
        self.p = []

    def clear(self, event):
        self.events = []
        self.mouseX = []
        self.mouseY = []
        self.cutoffsI = []

        for p in self.points:
            p.remove()
        self.p = []
        event.canvas.draw()

        print('Cleared')

    def next(self, event):
        self.idx = 0
        for p in self.points:
            p.remove()
        self.p = []

    def done(self, event):
        plt.close('all')

    def __call__(self, event):
        self.event = event
        self.events.append(event)
        self.x, self.y = event.mouseevent.xdata, event.mouseevent.ydata
        print(self.x, self.y)
        print(self.idx)
        if (self.x is not None) and (self.idx == 1):
            self.mouseX.append(self.x)
            self.mouseY.append(self.y)

            self.p.append(self.ax.scatter(self.x, self.y, color='black'))
            self.cutoff[1, 0] = self.x
            self.cutoff[1, 1] = self.y
            self.cutoffsI.append(self.cutoff)
            print('First Point')

            plt.scatter(self.x, self.y)
            self.idx += 1

        if self.idx == 0:
            self.cutoff = np.array([[self.x, self.y], [0, 0]])
            self.p.append(self.ax.scatter(self.x, self.y, color='black'))
            self.idx += 1
            print('Second Point')
        event.canvas.draw()

