#!/usr/bin/env python

import os
import argparse
import subprocess

description = 'Run beamy simulation'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('-p', '--path', help="Run simulation from specific file",
                    default=None)

parser.add_argument('-s', '--simulation', type=str,
                    help="Run simulation from default sims/ folder",
                    default=None)

args = parser.parse_args()

current_dir = os.getcwd()

if args.simulation:
    os.chdir(os.path.join('sims', args.simulation))
    subprocess.call(["python", 'run.py'])
    os.chdir(current_dir)
elif args.path:
    subprocess.call(["python", args.path])
else:
    print("No simulation given")
