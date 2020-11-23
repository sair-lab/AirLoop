#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import torchvision
from matplotlib import cm
import matplotlib.colors as mc
import kornia.geometry.conversions as C


class Visualization():
    def __init__(self, winname=None, debug=False):
        self.winname, self.debug = winname, debug
        self.radius = 1
        self.thickness = 1

    def show(self, images, points=None, color='red', nrow=2, values=None, vmin=None, vmax=None):
        b, c, h, w = images.shape
        if c == 3:
            images = torch2cv(images)
        elif c == 1: # show colored values
            images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
            images = get_colors(color, images.squeeze(-1), vmin, vmax)

        if points is not None:
            points = C.denormalize_pixel_coordinates(points, h, w)
            for i, pts in enumerate(points):
                colors = get_colors(color, [0]*len(pts) if values is None else values[i], vmin, vmax)
                images[i] = circles(images[i], pts, self.radius, colors, self.thickness)

        if nrow is not None:
            images = torch.tensor(images.copy()).permute((0, 3, 1, 2))
            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1).permute((1, 2, 0))
            cv2.imshow(self.winname, grid.numpy())
        else:
            for i, img in enumerate(images):
                cv2.imshow(self.winname+str(i), img)
        cv2.waitKey(1)

    def showmatch(self, img1, pts1, img2, pts2, color='blue', values=None, vmin=None, vmax=None):
        assert len(pts1) == len(pts2)
        h, w = img1.size(-2), img1.size(-1)
        pts1 = C.denormalize_pixel_coordinates(pts1, h, w)
        pts2 = C.denormalize_pixel_coordinates(pts2, h, w)
        img1, img2 = torch2cv(torch.stack([img1, img2]))
        colors = get_colors(color, [0]*len(pts1) if values is None else values, vmin, vmax)
        image = matches(img1, pts1, img2, pts2, colors)
        cv2.imshow(self.winname, image)
        cv2.waitKey(1)

    def reprojectshow(self, imgs, pts_src, pts_dst, src, dst):
        # TODO not adapted for change in torch2cv
        pts_src, pts_dst = pts_src[src], pts_src[dst]
        for i in range(src[0].size(0)):
            pts1 = pts_src[i].unsqueeze(0)
            pts2 = pts_dst[i].unsqueeze(0)
            img1 = torch2cv(imgs[src[0][i]]).copy()
            img2 = torch2cv(imgs[dst[0][i]]).copy()
            image = matches(img1,pts1,img2,pts2,self.blue,2)
            cv2.imshow(self.winname+'-dst', image)
            cv2.waitKey(1)


def matches(img1, pts1, img2, pts2, colors, circ_radius=3, thickness=1):
    ''' Assume pts1 are matched with pts2, respectively.
    '''
    H1, W1, C = img1.shape
    H2, W2, _ = img2.shape
    new_img = np.zeros((max(H1, H2), W1 + W2, C), img1.dtype)
    new_img[:H1, :W1], new_img[:H2, W1:W1+W2] = img1,  img2
    new_img = circles(new_img, pts1, circ_radius, colors, thickness)
    pts2[:, 0] += W1
    new_img = circles(new_img, pts2, circ_radius, colors, thickness)
    return lines(new_img, pts1, pts2, colors, thickness)


def circles(image, points, radius, colors, thickness):
    for pt, c in zip(points, colors):
        image = cv2.circle(image.copy(), tuple(pt), radius, tuple(c.tolist()), thickness, cv2.LINE_AA)
    return image


def lines(image, pts1, pts2, colors, thickness):
    for pt1, pt2, c in zip(pts1, pts2, colors):
        image = cv2.line(image.copy(), tuple(pt1), tuple(pt2), tuple(c.tolist()), thickness, cv2.LINE_AA)
    return image


def get_colors(name, values=[0], vmin=None, vmax=None):
    if name in mc.get_named_colors_mapping():
        rgb = mc.to_rgba_array(name)[0, :3]
        rgb = np.tile(rgb, (len(values), 1))
    else:
        values = np.array(values)
        normalize = mc.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(name)
        rgb = cmap(normalize(values))
    return (rgb[..., 2::-1] * 255).astype(np.uint8)


def torch2cv(images):
    rgb = (255 * images).type(torch.uint8).cpu().numpy()
    bgr = rgb[:, ::-1, ...].transpose((0, 2, 3, 1))
    return bgr
