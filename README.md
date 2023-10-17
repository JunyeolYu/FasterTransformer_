# This is written for tb3 setting

#### Go to the directory below:

``` bash
cd /workspace/FasterTransformer_/examples/pytorch/llama
```

#### For main process, run:

``` bash
mpirun -n 4 --allow-run-as-root python main.py --output_len 1 --pipeline_para_size 4 --ckpt_path /llm/ft_models/llama_30b_pp/4-gpu --tokenizer_path /llm/model/30B_converted_hf --lib_path /workspace/src/FasterTransformer/build/lib/libth_transformer.so
```

#### For preprocessing process, run:

``` bash
python preprocessing.py
```

#### Or simply you can start with:

``` bash
chmod -R 777 ./run.sh && bash run.sh 
```