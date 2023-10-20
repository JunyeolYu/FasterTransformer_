#!/bin/bash
start=$(date +%s.%N)

python preprocessing.py --tokenizer_path /model/llama-30b/ &
mpirun -n 4 --allow-run-as-root python main.py --output_len 1 --pipeline_para_size 4 --ckpt_path /model/llama/4-gpu/ --tokenizer_path /model/llama-30b/ --lib_path /root/FasterTransformer_/build/lib/libth_transformer.so

#python preprocessing.py --tokenizer_path $TOKENIZER_PATH &
#mpirun -n 4 --allow-run-as-root python main.py --output_len 1 --pipeline_para_size 4 --ckpt_path $CKPT_PATH --tokenizer_path $TOKENIZER_PATH --lib_path $LIB_PATH

finish=$(date +%s.%N)
time=$( echo "$finish - $start" | bc -l )
echo 'time:' $time