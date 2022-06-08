# -*- coding: utf-8 -*-

import torch

from common.trainer.basic_trainer import BasicTrainer
from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from custom.datasets import get_dataset, get_model_feed_in
from custom.datasets.transform.augmentation import get_transforms
from custom.eval.eval_func import run_eval
from custom.loss import build_loss
from custom.metric import build_metric
from custom.models import build_model
from custom.visual.render_img import render_progress_imgs, write_progress_imgs


class CustomTrainer(BasicTrainer):
    """Trainer for Customized case"""

    def __init__(self, cfgs):
        super(CustomTrainer, self).__init__(cfgs)

    def get_model(self):
        """Get custom model"""
        self.logger.add_log('-' * 60)
        model = build_model(self.cfgs, self.logger)

        return model

    def prepare_data(self):
        """Prepare dataset for train, val, eval. Gets data loader and sampler"""
        self.logger.add_log('-' * 60)
        data = {}
        # train
        assert self.cfgs.dataset.train is not None, 'Please input train dataset...'
        tkwargs = {
            'batch_size': self.cfgs.batch_size,
            'num_workers': self.cfgs.worker,
            'pin_memory': True,
            'drop_last': True
        }
        data['train'], data['train_sampler'] = self.set_dataset('train', tkwargs)

        # val
        if not valid_key_in_cfgs(self.cfgs.dataset, 'val'):
            data['val'] = None
        else:
            data['val'], data['val_sampler'] = self.set_dataset('val', tkwargs)

        # eval
        if not valid_key_in_cfgs(self.cfgs.dataset, 'eval'):
            data['eval'] = None
        else:
            eval_bs = get_value_from_cfgs_field(self.cfgs.dataset.eval, 'eval_batch_size', 1)
            tkwargs_eval = {
                'batch_size': eval_bs,
                'num_workers': self.cfgs.worker,
                'pin_memory': True,
                'drop_last': False
            }
            data['eval'], _ = self.set_dataset('eval', tkwargs_eval)

        return data

    def set_dataset(self, mode, tkwargs):
        """Get loader, sampler and aug_info"""
        transforms, _ = get_transforms(getattr(self.cfgs.dataset, mode))
        dataset = get_dataset(
            self.cfgs.dataset, self.cfgs.dir.data_dir, logger=self.logger, mode=mode, transfroms=transforms
        )

        sampler = None
        if mode != 'eval' and self.cfgs.dist.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.cfgs.dist.world_size, rank=self.cfgs.dist.rank
            )
        loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler, shuffle=(sampler is None and mode != 'eval'), **tkwargs
        )

        return loader, sampler

    def set_loss_factory(self):
        """Set loss factory which will be use to calculate all the loss"""
        self.logger.add_log('-' * 60)
        loss_factory = build_loss(self.cfgs, self.logger)

        return loss_factory

    def set_eval_metric(self):
        """Set eval metric which will be used for evaluation"""
        self.logger.add_log('-' * 60)
        eval_metric = build_metric(self.cfgs, self.logger)

        return eval_metric

    def get_model_feed_in(self, inputs, device):
        """Get the core model feed in and put it to the model's device"""
        return get_model_feed_in(inputs, device)

    def render_progress_imgs(self, inputs, output):
        """Actual render for progress image with label. It is perform in each step with a batch.
         Return a dict with list of image and filename. filenames should be irrelevant to progress
         Image should be in bgr with shape(h,w,3), which can be directly writen by cv.imwrite().
         Return None will not write anything.
         You should clone anything from input/output to forbid change of value
        """
        return render_progress_imgs(inputs, output)

    def evaluate(self, data, model, metric_summary, device, max_samples_eval):
        """Actual eval function for the model. Use run_eval since we want to run it locally as well"""
        metric_info, files = run_eval(
            data,
            self.get_model_feed_in,
            model,
            self.logger,
            self.eval_metric,
            metric_summary,
            device,
            self.render_progress_imgs,
            max_samples_eval,
            show_progress=False
        )

        return metric_info, files

    def write_progress_imgs(self, files, folder, epoch=None, step=None, global_step=None, eval=False):
        """Actual function to write the progress images"""
        write_progress_imgs(files, folder, epoch, step, global_step, eval)
