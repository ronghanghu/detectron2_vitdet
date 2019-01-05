import torch
import torch.multiprocessing as mp
import torch.distributed as dist


# TODO support distributed launch here
def launch(main_func, num_gpus_per_machine,
           num_machines=1, machine_rank=0, dist_url=None, args=()):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        args (tuple): arguments passed to main_func
    """
    assert num_gpus_per_machine <= torch.cuda.device_count()
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes
        mp.spawn(_distributed_worker,
                 nprocs=num_gpus_per_machine,
                 args=(main_func, world_size, num_gpus_per_machine,
                       machine_rank, dist_url, args), daemon=False)
    else:
        main_func(*args)


def _distributed_worker(local_rank, main_func,
                        world_size, num_gpus_per_machine, machine_rank, dist_url, args):
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    dist.init_process_group(
        backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank
    )
    torch.cuda.set_device(local_rank)
    main_func(*args)
