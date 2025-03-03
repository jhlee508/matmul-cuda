#!/bin/bash

# srun -p PV-Short --exclusive --gres=gpu:1 main $@

srun -p PV-Short --exclusive --gres=gpu:1 \
  ./main -v -w -n 5 4096 4096 4096 