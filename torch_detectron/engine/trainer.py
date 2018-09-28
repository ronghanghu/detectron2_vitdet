import datetime
import logging
import time

import torch

from torch_detectron.utils.metric_logger import MetricLogger


def train_one_epoch(
    model, data_loader, optimizer, scheduler, device, iteration, max_iter
):
    logger = logging.getLogger("torch_detectron.trainer")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    for _, (images, targets, _) in enumerate(data_loader):
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

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == (max_iter - 1):
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        iteration += 1
        if iteration >= max_iter:
            break
    return iteration


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    max_iter,
    device,
    use_distributed,
    arguments,
):
    logger = logging.getLogger("torch_detectron.trainer")
    logger.info("Start training")
    start_training_time = time.time()
    while arguments["iteration"] < max_iter:
        start_epoch_time = time.time()
        iteration = arguments["iteration"]
        if use_distributed:
            data_loader.batch_sampler.sampler.set_epoch(iteration)
        iteration_end = train_one_epoch(
            model, data_loader, optimizer, scheduler, device, iteration, max_iter
        )
        total_epoch_time = time.time() - start_epoch_time

        epoch_time_str = str(datetime.timedelta(seconds=total_epoch_time))
        logger.info(
            "Total epoch time: {} ({:.4f} s / it)".format(
                epoch_time_str, total_epoch_time / (iteration_end - iteration)
            )
        )
        arguments["iteration"] = iteration_end

        checkpointer.save("model_{:07d}".format(arguments["iteration"]), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
