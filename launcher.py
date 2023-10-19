#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
To run mpc_autograd_cnn example:

$ python examples/mpc_autograd_cnn/launcher.py

To run mpc_linear_svm example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_autograd_cnn/mpc_autograd_cnn.py \
      examples/mpc_autograd_cnn/launcher.py
"""

import argparse
import logging
import os

from multiprocess_launcher import MultiProcessLauncher

parser = argparse.ArgumentParser(description="CrypTensor Test")

def validate_world_size(world_size):
    world_size = int(world_size)
    if world_size < 2:
        raise argparse.ArgumentTypeError(f"world_size {world_size} must be > 1")
    return world_size

parser.add_argument(
    "--world_size",
    type=validate_world_size,
    default=3,
    help="The number of parties to launch. Each party acts as its own process (default: 3)",
)
parser.add_argument(
    "--rank",
    default=0,
    type=int,
    help="rank of this launcher (default: 0)",
)
parser.add_argument(
    "--rendezvous",
    default="tcp://127.0.0.1:10010",
    type=str,
    help="rendezvous of this launcher",
)
parser.add_argument(
    "--multiprocess",
    default=True,
    action="store_true",
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    level = logging.INFO

    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    from mpc_sdap import run_mpc_sdap
    run_mpc_sdap() 

def main(run_experiment):
    args = parser.parse_args()

    # Run with a TTP (Rank = args.world_size+1 )
    args.world_size += 1

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        custom_env = {
            "WORLD_SIZE":str(args.world_size),
            "RANK":str(args.rank),
            "RENDEZVOUS":str(args.rendezvous),
        }

        for env_key,env_value in custom_env.items():
            os.environ[env_key] = env_value
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)

