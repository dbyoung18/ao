#!/bin/bash

: ${MODEL=${1:-llama2-7b}} # [llama2-7b | llama3-8b]
: ${DTYPE=${2:-bf16}}      # [bf16 | fp32 | int8dq | int8wo | int4wo-64 | autoquant]
: ${DEVICE=${3:-cuda}}     # [cuda | xpu | cpu]
: ${PROFILE=${4:-false}}   # enable torch.profiler
: ${COMPILE:=true}
: ${PREFILL:=true}


if [[ ${DEVICE} == "xpu" ]]; then
  echo "Set IPEX-XPU runtime env"
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
fi

echo "Set Checkpoints path"
export CKPT_ROOT=${HOME}/checkpoints # path to checkpoints folder
case ${MODEL} in
"llama2-7b") MODEL_REPO=meta-llama/Llama-2-7b-chat-hf ;;
"llama3-8b") MODEL_REPO=meta-llama/Meta-Llama-3-8B ;;
*)
  echo "Invalid model: ${MODEL}"
  exit 1
  ;;
esac

CKPT_PATH=${CKPT_ROOT}/${MODEL_REPO}/model.pth
echo "Checkpoint path: ${CKPT_PATH}"

mkdir -p logs

CMD="python generate.py --checkpoint_path ${CKPT_PATH} --write_result benchmark_results.txt --device ${DEVICE}"

case ${DTYPE} in
"fp32") CMD+=" --precision torch.float32" ;;
"bf16") CMD+=" --precision torch.bfloat16" ;;
"int8dq") CMD+=" --quantization int8dq" ;;
"int8wo") CMD+=" --quantization int8wo" ;;
"int4wo-64") CMD+=" --quantization int4wo-64" ;;
"autoquant") CMD+=" --quantization autoquant" ;;
*)
  echo "Invalid data type: ${DTYPE}"
  exit 1
  ;;
esac

[ ${COMPILE} == true ] && CMD+=" --compile"
[ ${PREFILL} == true ] && CMD+=" --compile_prefill"
[ ${PROFILE} == true ] && CMD+=" --profile ./profile/${MODEL}_${DTYPE}_${DEVICE}_${COMPILE}_${PREFILL}"
CMD+=" 2>&1 | tee ./logs/${MODEL}_${DTYPE}_${DEVICE}_${COMPILE}_${PREFILL}.log"

echo CMD=${CMD}
eval ${CMD}
