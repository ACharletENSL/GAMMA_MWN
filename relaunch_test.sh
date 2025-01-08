#!/bin/sh

rm -f results/Last/*
#python setup.py
make -B
#cp -f ./phys_input.ini ./results/Last/
mpirun -n 1 ./bin/GAMMA -w
