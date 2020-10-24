#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import torchvision


class Visualization():
    white = (255,255,255)
    black = (  0,  0,  0)
    blue  = (255,  0,  0)
    red   = (  0,  0,255)
    def __init__(self, winname=None):
        self.winname = winname
        self.radius = 1
        self.thickness = 1

    def show(self, batch, points):
        for i in range(batch.size(0)):
            grid = torchvision.utils.make_grid(batch[i], padding=0)
            image, centers = torch2cv(grid).copy(), point2pixel(points[i])
            image = circles(image, centers, self.radius, self.red, self.thickness)
            cv2.imshow(self.winname+str(i), image)
        cv2.waitKey(1)


def circles(image, points, radius, color, thickness):
    for i in range(points.size(0)):
        image = cv2.circle(image, tuple(points[i]), radius, color, thickness, cv2.LINE_AA)
    return image


def torch2cv(image):
    ''' CxHxW --> WxHxC, thus coordinates reversed. Please use point2pixel'''
    return (255*image).type(torch.uint8).cpu().numpy()[::-1].transpose((1, 2, 0))


def point2pixel(point):
    ''' See torch2cv '''
    assert(point.size(-1)==2)
    return point.flip(-1)
