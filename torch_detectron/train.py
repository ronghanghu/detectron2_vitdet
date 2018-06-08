r"""
Basic training script for PyTorch

Run with

python -m torch_detectron.train --config-file path_to_config_file

to perform training
"""
import torch

import logging
import os

import time

from torch_detectron.utils.logging import setup_logger
from torch_detectron.utils.metric_logger import MetricLogger
from torch_detectron.utils.checkpoint import Checkpoint
from torch_detectron.utils.miscellaneous import mkdir

from torch_detectron.helpers.config_utils import load_config

def train_one_epoch(model, data_loader, optimizer, scheduler, device, iteration, max_iter):
    logger = logging.getLogger(__name__)
    meters = MetricLogger()
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
    logger = logging.getLogger(__name__)
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
        arguments.update(extra_checkpoint_data)

    logger.info('Start training')
    while arguments['iteration'] < config.max_iter:
        start_epoch_time = time.time()
        iteration = arguments['iteration']
        if config.distributed:
            data_loader.sampler.set_epoch(iteration)
        try:
            iteration_end = train_one_epoch(model, data_loader, optimizer, scheduler, config.device, iteration, config.max_iter)
        except (EOFError, ConnectionResetError):  # dataloader that returns early might raise this exception.
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
        from torch_detectron.core.inference import inference
        torch.cuda.empty_cache()  # TODO check if it helps
        inference(model, data_loader_val, box_only=False)


def main():
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

    setup_logger(__name__, config.save_dir, args.local_rank)

    logger = logging.getLogger(__name__)
    logger.info(args)

    logger.info('Loaded configuration file {}'.format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = '\n' + cf.read()
        logger.info(config_str)
    logger.info({k:v for k,v in config.__dict__.items() if not k.startswith('__')})

    train(config)


if __name__ == '__main__':
    main()
