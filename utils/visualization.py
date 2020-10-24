#!/usr/bin/env python3

import cv2
import numpy as np
import torchvision


class Visualization():
    def __init__(self, winname=None):
        self.winname = winname
        self.white = (255, 255, 255)
        self.black = 0#(0, 0, 0)
        self.radius = 1
        self.thickness = 1

    def show(self, batch, points):
        for i in range(batch.size(0)):
            grid = torchvision.utils.make_grid(batch[i], padding=0).cpu()
            image = grid.numpy()[::-1].transpose((1, 2, 0))
            centers = points[i].flip(-1)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = circles(gray, centers, self.radius, self.black, self.thickness)
            cv2.imshow(self.winname+str(i), image)
        cv2.waitKey(1)


def circles(image, points, radius, color, thickness):
    for i in range(points.size(0)):
        image = cv2.circle(image, tuple(points[i]), radius, color, thickness)
    return image
