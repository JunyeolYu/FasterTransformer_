#!/bin/bash
SRC="/workspace/LlamaRanch/src/FasterTransformer"
CKPT="/model/llama/4-gpu"
TOK="/model/llama-30b"

start=$(date +%s.%N)

python preprocessing.py --tokenizer_path $TOK &
mpirun -n 4 --allow-run-as-root python main.py --output_len 1 --pipeline_para_size 4 --ckpt_path $CKPT --tokenizer_path $TOK --lib_path $SRC/build/lib/libth_transformer.so

finish=$(date +%s.%N)
time=$( echo "$finish - $start" | bc -l )
echo 'time:' $time
