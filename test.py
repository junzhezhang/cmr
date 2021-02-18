"""
Test of CMR.
leverage from main.py
Demo of CMR.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import numpy as np
import skimage.io as io

import torch

from nnutils import test_utils
from nnutils import predictor as pred_util
from utils import image as img_util

import os.path as osp
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import scipy.io as sio
from collections import OrderedDict

from data import cub as cub_data
from utils import visutil
from utils import bird_vis
from utils import image as image_utils
from nnutils import train_utils
from nnutils import loss_utils
from nnutils import mesh_net

from nnutils import geom_utils

flags.DEFINE_string('renderer_opt', 'nmr', 'which renderer to choose')
flags.DEFINE_string('dataset', 'cub', 'cub or pascal or p3d')
flags.DEFINE_string('eval_save_dir', 'eval_save', 'which renderer to choose')
# flags.DEFINE_string('renderer_opt', 'nmr', 'which renderer to choose')
# flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS


def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    return img


def visualize(img, outputs, renderer):
    vert = outputs['verts'][0]
    cam = outputs['cam_pred'][0]
    texture = outputs['texture'][0]
    shape_pred = renderer(vert, cam)
    print('shape_pred',shape_pred.shape)
    img_pred = renderer(vert, cam, texture=texture)
    print('img_pred',shape_pred.shape)
    # Different viewpoints.
    vp1 = renderer.diff_vp(
        vert, cam, angle=30, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp2 = renderer.diff_vp(
        vert, cam, angle=60, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp3 = renderer.diff_vp(
        vert, cam, angle=60, axis=[1, 0, 0], texture=texture)

    img = np.transpose(img, (1, 2, 0))
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(shape_pred)
    plt.title('pred mesh')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(img_pred)
    plt.title('pred mesh w/texture')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(vp1)
    plt.title('different viewpoints')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(vp2)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(vp3)
    plt.axis('off')
    plt.draw()
    plt.show()
    print('saving file to vis/')
    filename = './vis/'+opts.demo_stem+'.png'
    print('saving file to:'+filename)
    # plt.savefig('demo.png')
    plt.savefig(filename)

def main(_):
    
    if opts.dataset == 'cub':
        self.data_module = cub_data
    else: 
        raise NotImplementedError
    print('opts.split',opts.split)
    self.dataloader = self.data_module.data_loader(opts)
    import ipdb; pdb.set_trace
    # import pdb; pdb.set_trace()
    # return img array (3, 257, 257)
    img = preprocess_image(opts.img_path, img_size=opts.img_size)
    
    batch = {'img': torch.Tensor(np.expand_dims(img, 0))}
    # init predictor, opts.texture = True, opts.use_sfm_ms = False
    predictor = pred_util.MeshPredictor(opts)
    # outputs keys: ['kp_pred', 'verts', 'kp_verts', 'cam_pred', 'mask_pred', 'texture', 'texture_pred', 'uv_image', 'uv_flow']
    # [(k,v.shape) for k, v in outputs.items()]
    #  ('texture', torch.Size([1, 1280, 6, 6, 6, 3])), ('texture_pred', torch.Size([1, 3, 256, 256])), ('uv_image', torch.Size([1, 3, 128, 256])), ('uv_flow', torch.Size([1, 128, 256, 2]))]
    outputs = predictor.predict(batch)

    # This is resolution
    renderer = predictor.vis_rend
    renderer.set_light_dir([0, 1, -1], 0.4)

    visualize(img, outputs, predictor.vis_rend)


if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
