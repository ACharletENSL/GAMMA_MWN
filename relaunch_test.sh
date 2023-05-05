#!/bin/sh

rm -f results/Last/*
make -B
time mpirun -n 1 ./bin/GAMMA -w