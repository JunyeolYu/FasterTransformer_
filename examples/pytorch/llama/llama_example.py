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

from torch.nn.utils.rnn import pad_sequence
import os
import sys
import argparse
import configparser
import timeit
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import datasets
import re
from tqdm import tqdm
import time 

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.llama.utils.llama import Llama

def load_hellaswag():
    hellaswag = datasets.load_dataset('hellaswag')
    validation = hellaswag['validation']
    validation_zeroshot = validation.filter(lambda example: example['split_type'] == 'zeroshot')
    print("Hellaswag dataset load finish , len: " + str(len(validation_zeroshot)))
    return validation_zeroshot

"""######################################KHAN######################################"""

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class ModelArgs:
    dim: int = 6656
    n_layers: int = 60
    n_heads: int = 52
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6

    max_batch_size: int = 32
    max_seq_len: int = 170


class EmbeddingModule(torch.nn.Module):
    def __init__(self, params: ModelArgs):

        super().__init__()
        
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        return h, start_pos, freqs_cis, mask



"""######################################KHAN######################################"""



class RequestInstance:
    def __init__(self, request_id, activity_label, context, endings, tokenizer, label, embedding_model):
        self.request_id = request_id
        self.activity_label = activity_label
        self.context = context
        self.ending1 = endings[0]
        self.ending2 = endings[1]
        self.ending3 = endings[2]
        self.ending4 = endings[3]
        self.endings = []
        self.embedding_model = embedding_model
        for i in range(4):
            self.endings.append(tokenizer.encode(self.preprocess(endings[i]))[1:])

        self.tokenizer = tokenizer
        self.label = label
        self.requests = self.build_requests()

    def preprocess(self,text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def build_requests(self):
        self.context = self.tokenizer.encode(self.preprocess(self.activity_label) + self.preprocess(": ") + self.preprocess(self.context))[1:]
        return [
            [self.request_id,
            len(self.context), 
            self.context, 0.0, ending_tok, self.label, i, len(ending_tok),
            self.embedding_model(torch.unsqueeze(torch.tensor(self.context + ending_tok[:-1]),0), 0)]
            for i,ending_tok in enumerate(self.endings)            
        ]

def engineering_dataset(validation_zeroshot, tokenizer, emb_model):
    requests = []
    for i, row in tqdm(enumerate(validation_zeroshot)):
        temp = RequestInstance(i, row['activity_label'], row['ctx'], row['endings'], tokenizer, int(row['label']), emb_model)
        requests.extend(temp.requests)

    requests = sorted(requests, key=lambda x: x[1] + x[7], reverse=True)

    max_tokens_40 = []
    max_tokens_80 = []
    max_tokens_120 = []
    max_tokens_170 = []

    for r in requests:
        ttt = r[1] + r[7]
        if ttt <= 40:
            max_tokens_40.append(r)
        elif ttt <= 80:
            max_tokens_80.append(r)
        elif ttt <= 120:
            max_tokens_120.append(r)
        elif ttt <= 170:
            max_tokens_170.append(r)

    max_batch_sizes_config = [248, 126, 69, 60][::-1] # FT 우리가 수정한 기본 버전으로 돌렸을 때 max [496, 252, 138, 120]
    
    print("bucket: ",len(max_tokens_40), len(max_tokens_80), len(max_tokens_120), len(max_tokens_170))
    for x,y in zip([max_tokens_40,max_tokens_80,max_tokens_120,max_tokens_170],max_batch_sizes_config[::-1]):
        print(len(x), y, len(x)/y)

    final_reqs = []
    
    for i in range(len(max_batch_sizes_config)):
        current_list = []
        if i == 3:
            current_list = max_tokens_40
        elif i == 2:
            current_list = max_tokens_80
        elif i == 1:
            current_list = max_tokens_120
        elif i == 0:
            current_list = max_tokens_170

        for j in range(0, len(current_list), max_batch_sizes_config[i]):
            final_reqs.append(current_list[j:j+max_batch_sizes_config[i]])
    return final_reqs

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
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")

    # sentencepiece needed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    start_dataset = time.time()
    validation_zeroshot = []
    final_reqs = []
    res = []
    validation_zeroshot = load_hellaswag()
    #load the embedding model 
    emb_model = torch.load(ckpt_path + "/emb_model.pt")
    emb_model.eval()
    #embedding_model = torch.nn.Embedding(vocab_size, size_per_head, padding_idx=0)
    final_reqs = engineering_dataset(validation_zeroshot, tokenizer, emb_model)

    # Prepare model.
    start_model = time.time()
    llama = Llama(head_num, size_per_head, inter_size, vocab_size, rotary_embedding, layernorm_eps,
                  start_id, end_id, layer_num, max_seq_len, 
                  tensor_para_size, pipeline_para_size, 
                  use_gptj_residual, lib_path, 
                  inference_data_type=inference_data_type, 
                  weights_data_type=weight_data_type)

    if not llama.load(ckpt_path=ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    
    ########### Eval Harness ###########
    res = []

    start_eval = time.time()
    for i in tqdm(range(len(final_reqs))):
        prompts = final_reqs[i]

        prompt_tokens = [prompt[2]+prompt[4][:-1] for prompt in prompts]
        #the embeds are (h, start_pos, freqs_cis, mask)
        prompt_embeds = [prompt[8] for prompt in prompts]
        #print(prompt_embeds[0][0].shape)
        prompt_size = [prompt[1]+prompt[-2]-1 for prompt in prompts]
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
            torch.cuda.empty_cache()
            if(rank == 3):
                multi_logits = torch.nn.functional.log_softmax(output_log_probs, dim=-1)
                
                _res = []
                for logits, prompt in zip(multi_logits, prompts):
                    _input, ending, el = prompt[1]-1, prompt[4], prompt[-2]
                    logits = logits[_input:_input+el].unsqueeze(0)  # [1, seq, vocab]
                    ending = torch.tensor(ending, dtype=torch.long, device='cuda').view(1,-1,1)
                    answer = torch.gather(logits, 2, ending).squeeze(-1).sum()  # [1, ]
                    _res.append(answer)
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
        t_before = start_dataset - start_main
        t_dataset = start_model - start_dataset
        t_load = start_eval - start_model
        t_eval = start_cal - start_eval
        t_cal = end - start_cal

        print(f"Total_time    : {t_total} s")
        print(f"Preprocessing : {t_dataset} s")
        print(f"Model_loading : {t_load} s")
        print(f"Evaluation    : {t_eval} s")
        print(f"Acc_Cal       : {t_cal} s")
        print(f"Others        : {t_before} s")
    ####################################

if __name__ == '__main__':
    main()
