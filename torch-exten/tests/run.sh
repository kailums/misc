#!/bin/bash

PROF="nsys profile -o torch-linear-matmul -f true --trace=cuda,nvtx,cublas,cudnn"


$PROF python test.py
