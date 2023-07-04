"""
    This file provides utilities to launch multi-gpu and multi-node jobs.
    Modified from https://github.com/fundamentalvision/Deformable-DETR/blob/main/tools/launch.py
    and https://github.com/pytorch/pytorch/blob/173f224570017b4b1a3a1a13d0bff280a54d9cd9/torch/distributed/launch.py
    
"""
import sys
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER

import torch
import main

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=3,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=22500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    parser.add_argument("--training_script", type=str, default='main.py',
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")
    return parser


def main():
    parser = parse_args()
    args, train_args = parser.parse_known_args()
    args.training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.training_script)
    
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes


    summary = f"Number of GPUs available: {torch.cuda.device_count()}\n" \
              f"Number of nodes: {args.nnodes}\n" \
              f"Number of GPUs per node: {args.nproc_per_node}\n" \
              f"World size: {dist_world_size}\n" \
              f"Master addr: {args.master_addr}\n" \
              f"Master port: {args.master_port}"

    print(summary)
    print("\n" + "-" * 100 + "\n\n\n")

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # to pass the other args to main.py
        cmd = ["python", args.training_script, "--local_rank", str(local_rank)] + sys.argv[1:]
        print(f'CMD: {cmd}')
        #process = subprocess.Popen(cmd, env=current_env)
        process = subprocess.Popen(cmd + train_args, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)


if __name__ == "__main__":
    print('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Using Distributed Training', '-' * 100))    
    
    main()
