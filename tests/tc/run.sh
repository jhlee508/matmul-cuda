#!/bin/bash

srun -p PV-Short --exclusive --gres=gpu:1 \
  ./main

  # nvprof ./main
  # compute-sanitizer ./main