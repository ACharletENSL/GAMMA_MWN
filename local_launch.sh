#!/bin/sh

python setup.py
make -B
rm -f ./results/Last/*
cp -f ./phys_input.ini ./results/Last/
mpirun -n 1 ./bin/GAMMA -w