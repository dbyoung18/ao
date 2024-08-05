#!/bin/bash

DEVICE=xpu
COMPILE=false
MODEL=llama2-7b DTYPE=bf16 DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama2-7b DTYPE=int8dq DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama2-7b DTYPE=int8wo DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama2-7b DTYPE=int4wo-64 DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama2-7b DTYPE=autoquant DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=true ./launch_model.sh

MODEL=llama3-8b DTYPE=bf16 DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama3-8b DTYPE=int8dq DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama3-8b DTYPE=int8wo DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama3-8b DTYPE=int4wo-64 DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=false ./launch_model.sh
MODEL=llama3-8b DTYPE=autoquant DEVICE=${DEVICE} COMPILE=${COMPILE} PREFILL=true ./launch_model.sh