#!/bin/sh

rm -f results/Last/*
make -B
mpirun -n 1 ./bin/GAMMA -w