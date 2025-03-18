#!/bin/bash

# srun -p PV-Short --exclusive --gres=gpu:1 main $@

srun -p PV-Short --exclusive --gres=gpu:1 \
  main -h