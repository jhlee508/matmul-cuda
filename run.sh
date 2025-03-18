#!/bin/bash

# srun -p PV-Short --exclusive --gres=gpu:1 main $@

srun -p PV-Short --exclusive --gres=gpu:1 \
  main 4096 4096 4096 -w -n 5 -v