#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import torchvision
import kornia.geometry.conversions as C


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
        b, c, h, w = batch.shape
        points = C.denormalize_pixel_coordinates(points, h, w)
        for i in range(batch.size(0)):
            grid = torchvision.utils.make_grid(batch[i], padding=0)
            image = torch2cv(grid).copy()
            image = circles(image, points[i], self.radius, self.red, self.thickness)
            cv2.imshow(self.winname+str(i), image)
        cv2.waitKey(1)

    def showmatch(self, img1, pts1, img2, pts2):
        h, w = img1.size(-2), img1.size(-1)
        pts1 = C.denormalize_pixel_coordinates(pts1, h, w)
        pts2 = C.denormalize_pixel_coordinates(pts2, h, w)
        img1, img2 = torch2cv(img1).copy(), torch2cv(img2).copy()
        image = matches(img1,pts1,img2,pts2,self.blue,2)
        cv2.imshow(self.winname, image)
        cv2.waitKey(1)

    def reprojectshow(self, imgs, pts_src, pts_dst, src, dst):
        pts_src, pts_dst = pts_src[src], pts_src[dst]
        for i in range(src[0].size(0)):
            pts1 = pts_src[i].unsqueeze(0)
            pts2 = pts_dst[i].unsqueeze(0)
            img1 = torch2cv(imgs[src[0][i]]).copy()
            img2 = torch2cv(imgs[dst[0][i]]).copy()
            image = matches(img1,pts1,img2,pts2,self.blue,2)
            cv2.imshow(self.winname+'-dst', image)
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
    return (255*image).type(torch.uint8).cpu().numpy()[::-1].transpose((1, 2, 0))
