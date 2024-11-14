#!/bin/sh

rm -f results/Last/*
python setup.py
make -B
time mpirun -n 1 ./bin/GAMMA -w