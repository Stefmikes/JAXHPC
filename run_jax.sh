#!/bin/bash

# Set one unique GPU per rank based on the MPI local rank
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

echo "Rank $OMPI_COMM_WORLD_RANK using GPU $CUDA_VISIBLE_DEVICES on host $(hostname)"

# Now run your JAX script
python JAXMPI.py
