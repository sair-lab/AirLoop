#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
import torchvision
from matplotlib import cm
import matplotlib.colors as mc
import kornia.geometry.conversions as C


class Visualizer():
    vis_id = 0

    def __init__(self, display='imshow', default_name=None, **kwargs):
        self.radius, self.thickness = 1, 1
        self.default_name = 'Visualizer %d' % self.vis_id if default_name is None else default_name
        Visualizer.vis_id += 1

        if display == 'imshow':
            self.displayer = ImshowDisplayer()
        elif display == 'tensorboard':
            self.displayer = TBDisplayer(**kwargs)
        elif display == 'video':
            self.displayer = VideoFileDisplayer(**kwargs)

    def show(self, images, points=None, color='red', nrow=4, values=None, vmin=None, vmax=None, name=None, step=0):
        b, c, h, w = images.shape
        if c == 3:
            images = torch2cv(images)
        elif c == 1:  # show colored values
            images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
            images = get_colors(color, images.squeeze(-1), vmin, vmax)

        if points is not None:
            points = C.denormalize_pixel_coordinates(points, h, w).to(torch.int)
            for i, pts in enumerate(points):
                colors = get_colors(color, [0]*len(pts) if values is None else values[i], vmin, vmax)
                images[i] = circles(images[i], pts, self.radius, colors, self.thickness)

        disp_name = name if name is not None else self.default_name

        if nrow is not None:
            images = torch.tensor(images.copy()).permute((0, 3, 1, 2))
            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1).permute((1, 2, 0))
            self.displayer.display(disp_name, grid.numpy(), step)
        else:
            for i, img in enumerate(images):
                self.displayer.display(disp_name + str(i), img, step)

    def showmatch(self, imges1, points1, images2, points2, color='blue', values=None, vmin=None, vmax=None, name=None, step=0, nrow=2):
        match_pairs = []
        for i, (img1, pts1, img2, pts2) in enumerate(zip(imges1, points1, images2, points2)):
            assert len(pts1) == len(pts2)
            h, w = img1.size(-2), img1.size(-1)
            pts1 = C.denormalize_pixel_coordinates(pts1, h, w)
            pts2 = C.denormalize_pixel_coordinates(pts2, h, w)
            img1, img2 = torch2cv(torch.stack([img1, img2]))
            colors = get_colors(color, [0]*len(pts1) if values is None else values[i], vmin, vmax)
            match_pairs.append(torch.tensor(matches(img1, pts1, img2, pts2, colors)))

        images = torch.stack(match_pairs).permute((0, 3, 1, 2))
        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=1).permute((1, 2, 0))
        self.displayer.display(name if name is not None else self.default_name, grid.numpy(), step)

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

    def close(self):
        self.displayer.close()


class VisDisplayer():
    def display(self, name, frame, step=0):
        raise NotImplementedError()

    def close(self):
        pass


class ImshowDisplayer(VisDisplayer):
    def display(self, name, frame, step=0):
        cv2.imshow(name, frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class TBDisplayer(VisDisplayer):
    def __init__(self, writer):
        self.writer = writer

    def display(self, name, frame, step=0):
        self.writer.add_image(name, frame[:, :, ::-1], step, dataformats='HWC')


class VideoFileDisplayer(VisDisplayer):
    def __init__(self, save_dir=None, framerate=10):
        if save_dir is None:
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.save_dir = os.path.join('.', 'vidout', current_time)
        else:
            self.save_dir = save_dir
        self.framerate = framerate
        self.writer = {}

    def display(self, name, frame, step=0):
        if name not in self.writer:
            os.makedirs(self.save_dir, exist_ok=True)
            self.writer[name] = cv2.VideoWriter(os.path.join(self.save_dir, '%s.avi' % name),
                                                cv2.VideoWriter_fourcc(*'avc1'),
                                                self.framerate, (frame.shape[1], frame.shape[0]))
        self.writer[name].write(frame)

    def close(self):
        for wn in self.writer:
            self.writer[wn].release()


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
        if not torch.any(pt.isnan()):
            image = cv2.circle(image.copy(), tuple(pt), radius, tuple(c.tolist()), thickness, cv2.LINE_AA)
    return image


def lines(image, pts1, pts2, colors, thickness):
    for pt1, pt2, c in zip(pts1, pts2, colors):
        if not torch.any(pt1.isnan() | pt2.isnan()):
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
