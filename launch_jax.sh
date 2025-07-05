#!/bin/bash

# Get the local rank assigned by OpenMPI
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}

# Bind this rank to one specific GPU
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

echo "Rank $LOCAL_RANK using GPU $CUDA_VISIBLE_DEVICES"

# Run your Python code
exec python -u JAXMPI.py
