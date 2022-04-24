# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np

from common.utils.img_utils import img_to_uint8


def render_progress_imgs(inputs, output):
    """Actual render for progress image with label. It is perform in each step with a batch.
     Return a dict with list of image and filename. filenames should be irrelevant to progress
     Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
     Return None will not write anything.
     You should clone anything from input/output to forbid change of value
    """
    dic = {}

    # images
    img = output['img'][0].detach().cpu().numpy().copy().repeat(3, 0)  # (H, W, 3)
    gt = inputs['gt'][0].detach().cpu().numpy().copy().repeat(3, 0)  # (H, W, 3)

    img = img_to_uint8(img, transpose=[1, 2, 0])  # (H, W, 3)
    gt = img_to_uint8(gt, transpose=[1, 2, 0])  # (H, W, 3)

    sample_img = np.concatenate([img, gt], axis=1)  # (H, 2W, 3)
    name = ['sample1', 'sample2']

    dic['imgs'] = {'names': name, 'imgs': [sample_img] * 2}

    return dic


def write_progress_imgs(files, folder, epoch=None, step=None, global_step=None, eval=False):
    """Actual function to write the progress image from render image

    Args:
        files: a list of dict, each contains:
                imgs: with ['names', 'imgs'], each is the image and names
              You can also add other types of files (figs, etc) for rendering.
        folder: the main folder to save the result
        epoch: epoch, use when eval is False
        step: step, use when eval is False
        global_step: global_step, use when eval is True
        eval: If true, save name as 'eval_xxx.png', else 'epoch_step_global.png'
    """
    num_sample = len(files)

    # write the image
    if 'imgs' in files[0] and len(files[0]['imgs']['names']) > 0:
        for img_name in files[0]['imgs']['names']:
            os.makedirs(osp.join(folder, img_name), exist_ok=True)

        for idx, file in enumerate(files):
            for name, img in zip(file['imgs']['names'], file['imgs']['imgs']):
                img_folder = osp.join(folder, name)
                os.makedirs(img_folder, exist_ok=True)

                if eval:
                    img_path = osp.join(img_folder, 'eval_{:04d}.png'.format(idx))
                else:
                    if num_sample == 1:
                        img_path = osp.join(
                            img_folder, 'epoch{:06d}_step{:05d}_global{:08d}.png'.format(epoch, step, global_step)
                        )
                    else:
                        img_path = osp.join(
                            img_folder,
                            'epoch{:06d}_step{:05d}_global{:08d}_{:04d}.png'.format(epoch, step, global_step, idx)
                        )

                cv2.imwrite(img_path, img)
