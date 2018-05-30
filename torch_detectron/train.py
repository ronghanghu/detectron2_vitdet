r"""
Basic training script for PyTorch

Run with

python -m torch_detectron.train --config-file path_to_config_file

to perform training
"""
import torch

import importlib
import importlib.util

import logging
import os
import sys

import time


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# TODO improved to have default values
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("torch_detectron.config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def mkdir(path):
    import os, errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def setup_logger(save_dir, local_rank):
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if local_rank > 0:
        return
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


class MeterLogger(object):
    def __init__(self):
        from collections import defaultdict
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {:.4f} ({:.4f})'.format(name, meter.median, meter.global_avg))
        return '\t'.join(loss_str)

class Checkpoint(object):
    def __init__(self, model, optimizer, scheduler, save_dir, local_rank):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.local_rank = local_rank

    def __call__(self, name, **kwargs):
        if not self.save_dir:
            return

        if self.local_rank > 0:
            return

        data = {}
        data['model'] = self.model.state_dict()
        data['optimizer'] = self.optimizer.state_dict()
        # data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)

        torch.save(data, os.path.join(self.save_dir, '{}.pth'.format(name)))

    def load(self, f):
        checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint.pop('model'))
        self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
        # self.scheduler.load_state_dict(checkpoint.pop('scheduler'))

        # extra arguments taht were stored
        return checkpoint


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        from collections import deque
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


def train_one_epoch(model, data_loader, optimizer, scheduler, device, iteration, max_iter):
    logger = logging.getLogger('trainer')
    meters = MeterLogger()
    model.train()
    end = time.time()
    for i, (images, targets) in enumerate(data_loader):
        data_time = time.time() - end

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        meters.update(loss=losses, **loss_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if iteration % 20 == 0 or iteration == (max_iter - 1):
            logger.info('Iter: {0}\t'
                  '{meters}\t'
                  'lr {lr:.6f}\t'
                  'Max Memory {memory:.0f}'.format(iteration,
                      meters=str(meters),
                      lr=optimizer.param_groups[0]['lr'],
                      memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

        iteration += 1
        if iteration >= max_iter:
            break
    return iteration

def train(config):
    logger = logging.getLogger('trainer')
    data_loader, data_loader_val = config.get_data_loader(config.distributed)
    model = config.get_model()
    optimizer = config.get_optimizer(model)
    scheduler = config.get_scheduler(optimizer)

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
            device_ids=[config.local_rank], output_device=config.local_rank)

    arguments = {}
    arguments['iteration'] = 0

    checkpointer = Checkpoint(model, optimizer, scheduler, config.save_dir, config.local_rank)

    if config.checkpoint:
        extra_checkpoint_data = checkpointer.load(config.checkpoint)
        arguments.extend(extra_checkpoint_data)

    logger.info('Start training')
    while arguments['iteration'] < config.max_iter:
        start_epoch_time = time.time()
        iteration = arguments['iteration']
        if config.distributed:
            data_loader.sampler.set_epoch(iteration)
        try:
            iteration_end = train_one_epoch(model, data_loader, optimizer, scheduler, config.device, iteration, config.max_iter)
        except EOFError:  # dataloader that returns early might raise this exception.
            pass
        total_epoch_time = time.time() - start_epoch_time
        import datetime
        epoch_time_str = str(datetime.timedelta(seconds=total_epoch_time))
        logger.info('Total epoch time: {} ({:.4f} s / it)'.format(
            epoch_time_str, total_epoch_time / (iteration_end - iteration)))
        arguments['iteration'] = iteration_end

        checkpointer('model_{}'.format(arguments['iteration']), **arguments)

    if config.local_rank == 0 and config.do_test:
        if config.distributed:
            model = model.module
        from core.inference import inference
        torch.cuda.empty_cache()  # TODO check if it helps
        inference(model, data_loader_val, box_only=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Training')
    parser.add_argument('--config-file', default='/private/home/fmassa/github/detectron.pytorch/configs/rpn_r50.py',
            metavar='FILE', help='path to config file')

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
            init_method='env://')

    config = load_config(args.config_file)
    config.local_rank = args.local_rank
    config.distributed = args.distributed

    if config.save_dir:
        mkdir(config.save_dir)

    setup_logger(config.save_dir, args.local_rank)

    logger = logging.getLogger('trainer')
    logger.info(args)

    logger.info('Loaded configuration file {}'.format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = '\n' + cf.read()
        logger.info(config_str)
    logger.info({k:v for k,v in config.__dict__.items() if not k.startswith('__')})

    train(config)
