#!/bin/bash

#objcopy -I binary -O elf64-x86-64 -B i386:x86-64 softmax_fp32_1024.hsaco softmax_fp32_1024.o
#objcopy -I binary -O elf64-x86-64 -B i386:x86-64 softmax_fp32_2048.hsaco softmax_fp32_2048.o

ar rcs softmax_fp16.a softmax_fp16_*
ar rcs softmax_fp32.a softmax_fp32_*

# hipcc -shared -fPIC -Wl,--whole-archive \
# 	softmax_fp16_1024.o  \
# 	softmax_fp16_16384.o \
# 	softmax_fp16_2048.o  \
# 	softmax_fp16_4096.o  \
# 	softmax_fp16_8192.o  \
# 	softmax_fp32_1024.o  \
# 	softmax_fp32_16384.o \
# 	softmax_fp32_2048.o \
# 	softmax_fp32_4096.o  \
# 	softmax_fp32_8192.o \
# 	-Wl,--no-whole-archive load_kernel.cu -o libkernel.so

hipcc -shared -fPIC -Wl,--whole-archive *.a -Wl,--no-whole-archive load_kernel.cu -o libkernel.so

# fail without --whole-archive
#hipcc -shared -fPIC *.a  load_kernel.cu -o libkernel.so

hipcc main.cu -o test -lkernel -L. -ldl


