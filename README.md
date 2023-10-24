# FasterTransformer

This repository is forked from the [repository](https://github.com/vitrun/FasterTransformer) providing a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA.

It aims to efficiently perform LLaMA-30B model inference using Fastertransformer as an inference engine for zero-shot Hellaswag task processing.

>In NLP, encoder and decoder are two important components, with the transformer layer becoming a popular architecture for both components. FasterTransformer implements a highly optimized transformer layer for both the encoder and decoder for inference. On Volta, Turing and Ampere GPUs, the computing power of Tensor Cores are used automatically when the precision of the data and weights are FP16. FasterTransformer is built on top of CUDA, cuBLAS, cuBLASLt and C++. We provide at least one API of the following frameworks: TensorFlow, PyTorch and Triton backend. Users can integrate FasterTransformer into these frameworks directly. For supporting frameworks, we also provide example codes to demonstrate how to use, and show the performance on these frameworks.

## Features
- Supporting for FP-16 LLaMA-30B inference with PyTorch framework based on [this branch](https://github.com/vitrun/FasterTransformer/tree/llama_torch)
- Supporting pipeline parallelism for the target model
- Suited for Hellaswag task (sequence classification)
- Removed auto-regressive decode (generation) module
- Calculating prediction class from prefill module's output
- Modified to prevent unnecessary invocation of some of the existing implementations that are not related to performing the target sequence classification task
- Modified to prevent unnecessary loading of layer weights on each GPU
- Adjusting dataset for efficient inference
- Overlapping model loading process with data preprocessing process * ***Update during 2nd round*** *
- Removing unnecessary buffer used and key-value cache, existed in original FasterTransformer * ***Update during 2nd round*** *
- Adjusting batch size with more bins (more suitable, fine-grained) * ***Update during 2nd round*** *

## Setup
In a container with `pytorch-23.05-py3` image,

#### Partitioning Model Weights
```bash
cd /ft_workspace/FasterTransformer
sudo mkdir models && sudo chmod -R 777 ./*
python ./examples/cpp/llama/huggingface_llama_convert.py -saved_dir=./models/llama -in_file=$MODEL_PATH -infer_gpu_num=4 -weight_data_type=fp16 -model_name=llama
```
#### Build
```bash
cd /workspace/FasterTransformer_
mkdir build && cd build
git submodule init && git submodule update
pip3 install fire jax jaxlib transformers datasets sentencepiece numpysocket
apt-get install bc

CUDAFLAGS="-include stdio.h" cmake -DSM=70 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -D PYTHON_PATH=/usr/bin/python3 ..
make -j$(nproc)
```

## Inference
```bash
export $CKPT_PATH="" && export $TOKENIZER_PATH="" && export $LIB_PATH=""
cd /workspace/FasterTransformer_/examples/pytorch/llama

chmod 777 exec_evaluation.sh && bash exec_evaluation.sh
```

## Killilng Backgounrd Preprocessing Process
- If a port connection is performed in the main process before the binding is performed in the preprocessing process, an error can occur (but this happens almost rarely)
- Before running again, it is recommended to execute the `ps -a` command, then `kill -9 pythonpid`
