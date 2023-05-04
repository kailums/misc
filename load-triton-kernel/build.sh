#!/bin/bash

objcopy -I binary -O elf64-x86-64 -B i386:x86-64 softmax_fp32_1024.hsaco softmax_fp32_1024.o
objcopy -I binary -O elf64-x86-64 -B i386:x86-64 softmax_fp32_2048.hsaco softmax_fp32_2048.o

hipcc -shared -fPIC -Wl,--whole-archive softmax_fp32_1024.o softmax_fp32_2048.o -Wl,--no-whole-archive load_kernel.cu -o libkernel.so

hipcc main.cu -o test -lkernel -L. -ldl


