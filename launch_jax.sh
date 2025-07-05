#!/bin/bash

# Correctly set GPU visibility before Python starts
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

echo "Rank $OMPI_COMM_WORLD_RANK using GPU $CUDA_VISIBLE_DEVICES"

# Do not set CUDA_VISIBLE_DEVICES anywhere else!

exec python -u JAXMPI.py
