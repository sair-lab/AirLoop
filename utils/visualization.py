#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import torchvision
from mpl_toolkits.mplot3d import Axes3D


class Visualization():
    white = (255,255,255)
    black = (  0,  0,  0)
    blue  = (255,  0,  0)
    red   = (  0,  0,255)
    def __init__(self, winname=None, debug=False):
        self.winname, self.debug = winname, debug
        self.radius = 1
        self.thickness = 1

    def show(self, batch, points):
        for i in range(batch.size(0)):
            grid = torchvision.utils.make_grid(batch[i], padding=0)
            image, centers = torch2cv(grid).copy(), point2pixel(points[i])
            image = circles(image, centers, self.radius, self.red, self.thickness)
            cv2.imshow(self.winname+str(i), image)
        cv2.waitKey(1)

    def showmatch(self, img1, pts1, img2, pts2):
        img1, pts1 = torch2cv(img1).copy(), point2pixel(pts1)
        img2, pts2 = torch2cv(img2).copy(), point2pixel(pts2)
        image = matches(img1,pts1,img2,pts2,self.blue,2)
        cv2.imshow(self.winname, image)
        cv2.waitKey(1)


def matches(img1, pts1, img2, pts2, color, flags):
    ''' Assume pts1 are matched with pts2, respectively.
    '''
    kpts1 = [cv2.KeyPoint(x=int(pts1[i,0]), y=int(pts1[i,1]), _size=1) for i in range(pts1.size(0))]
    kpts2 = [cv2.KeyPoint(x=int(pts2[i,0]), y=int(pts2[i,1]), _size=1) for i in range(pts2.size(0))]
    matches = [cv2.DMatch(i,i,0) for i in range(min(pts1.size(0),pts2.size(0)))]
    return cv2.drawMatches(img1,kpts1,img2,kpts2,matches,None,matchColor=color, flags=flags)


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
