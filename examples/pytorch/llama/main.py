# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from __future__ import print_function

import socket
from numpysocket import NumpySocket
from torch.nn.utils.rnn import pad_sequence
import os
import sys
import argparse
import configparser
import timeit
import torch
import torch.distributed as dist
import datasets
import re
from tqdm import tqdm
import time 

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.llama.utils.llama import Llama

def calculate_accuracy(res):
    acc = 0
    nacc = 0

    for r in range(0,len(res), 4):
        try:
            outs = sorted(res[r:r+4], key=lambda x: x[6])
            # assert that outs order is correct
            assert outs[0][6] == 0
            assert outs[1][6] == 1
            assert outs[2][6] == 2
            assert outs[3][6] == 3

            # [self.request_id,len(self.context), self.context, 0.0, ending_tok, self.label, i]
            logs = [out[3] for out in outs]

            ending_lens = [len(out[4]) for out in outs]
            nlogs = [log/ending_lens[i] for i,log in enumerate(logs)]

            pred_label = logs.index(max(logs))
            norm_pred_label = nlogs.index(max(nlogs))

            label = outs[0][5]

            if pred_label == label:
                acc += 1
            if norm_pred_label == label:
                nacc += 1
        except:
            print("Failed while calculating accuracy")
            pass
    
    total_len = len(res)/4
    acc = acc/total_len
    nacc = nacc/total_len
    print("Accuracy:", acc)
    print("Normalized Accuracy:", nacc)

def main():
    start_main = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_len', type=int, default=1,
                        help='output sequence length to generate.')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, 
                        help='path to the checkpoint file.')
    parser.add_argument('--tokenizer_path', type=str, 
                        help='directory where the tokenizer file is located.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--sample_input_file', type=str,
                        help='path to the sample input file.')
    parser.add_argument('--start_id_file', type=str,
                        help='path to the start id file.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is enable the random seed.')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(args.ckpt_path, "config.ini"))
    head_num = int(config.get('llama', 'head_num'))
    size_per_head = int(config.get('llama', 'size_per_head'))
    inter_size = int(config.get('llama', 'inter_size'))
    vocab_size = int(config.get('llama', 'vocab_size'))
    layer_num = int(config.get('llama', 'num_layer'))
    rotary_embedding = int(config.get('llama', 'rotary_embedding'))
    layernorm_eps = float(config.get('llama', 'layernorm_eps'))
    start_id = int(config.get('llama', 'start_id'))
    end_id = int(config.get('llama', 'end_id'))
    use_gptj_residual = False
    weight_data_type = config.get('llama', 'weight_data_type')

    ckpt_path = args.ckpt_path
    tokenizer_path = args.tokenizer_path
    lib_path = args.lib_path
    output_len = args.output_len
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    inference_data_type = args.inference_data_type

    if tensor_para_size * pipeline_para_size > 1:
        dist.init_process_group("mpi")
    rank = dist.get_rank() if dist.is_initialized() else 0
    device_count = dist.get_world_size() if dist.is_initialized() else 1
    device = rank % device_count
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()

    # Only rank 3 process can print messages
    if rank<3:
        os.sys.stdout = open(os.devnull, "w")
        os.sys.stderror = open(os.devnull, "w")

    print("\n=============== Arguments ===============")
    args_ = vars(args)
    for arg in ["pipeline_para_size","tensor_para_size","ckpt_path","tokenizer_path","lib_path"]:
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")

	# Prepare model.
    start_load = time.time()
    llama = Llama(head_num, size_per_head, inter_size, vocab_size, rotary_embedding, layernorm_eps,
                  start_id, end_id, layer_num, max_seq_len, 
                  tensor_para_size, pipeline_para_size, 
                  use_gptj_residual, lib_path, 
                  inference_data_type=inference_data_type, 
                  weights_data_type=weight_data_type)

    if not llama.load(ckpt_path=ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")

    # PGJ : Data preprocessing / from preprocessing process
    final_reqs = []
    port = 9020 + rank
    
    with NumpySocket() as s:
        s.connect(("localhost", port))
        final_reqs = s.recv()
        final_reqs = final_reqs.tolist()
        s.close()
    
    ########### Eval Harness ###########
    res = []

    start_eval = time.time()
    for i in tqdm(range(len(final_reqs))):
        prompts = final_reqs[i]

        prompt_tokens = [prompt[2]+prompt[4][:-1] for prompt in prompts]
        prompt_size = [prompt[1]+prompt[-1]-1 for prompt in prompts]
        start_lengths = torch.IntTensor(prompt_size)#[prompt[1] for prompt in prompts]
        
        tokens = torch.full((len(prompts), max(prompt_size)), 0, dtype = torch.int32, device='cuda')
        for k, t in enumerate(prompt_tokens):
            tokens[k, : prompt_size[k]] = torch.tensor(t)

        with torch.no_grad():
            # PGJ: return_output_length를 True로, return_cum_log_probs를 0이 아니게 주면
            # forward 시 (output_ids, output_lengths, output_cum_log_probs)로 반환
            batch_size = len(prompts)
            tokens_batch, output_lengths, output_log_probs = llama(
                start_ids=tokens,
                start_lengths=start_lengths,
                output_len=0,
                beam_width=beam_width,
                top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                random_seed=torch.zeros([batch_size], dtype=torch.int64),
                return_output_length=True,
                return_cum_log_probs=0)

            if(rank == 3):
                min_tok = min([prompt[1]-1 for prompt in prompts])
                multi_logits = torch.nn.functional.log_softmax(output_log_probs[:,min_tok:,:], dim=-1)
                
                _res = []
                for logits, prompt in zip(multi_logits, prompts):
                    _input = prompt[1] -1 - min_tok 
                    logits = logits[_input:_input+prompt[-1]]  # [1, seq, vocab]
                    ending = torch.tensor(prompt[4], dtype=torch.long, device='cuda').view(-1,1)
                    answer = torch.gather(logits, 1, ending).sum().cuda()  # [1, ]
                    _res.append(answer)
                # temp = []
                for prompt, ans in zip(prompts, _res):
                    prompt[3] = ans
            if(rank == 3):
                res.extend(prompts)
    if(rank == 3):
        start_cal = time.time()
        res = sorted(res, key=lambda x: x[0])
        calculate_accuracy(res)

        end = time.time()

        t_total = end - start_main
        t_load_and_preproc = start_eval - start_load
        t_eval = start_cal - start_eval
        t_cal = end - start_cal

        print(f"Total_time    : {t_total} s")
        print(f"Model_loading + Preprocessing : {t_load_and_preproc} s")
        print(f"Evaluation    : {t_eval} s")
        print(f"Acc_Cal       : {t_cal} s")
    ####################################

if __name__ == '__main__':
    main()
