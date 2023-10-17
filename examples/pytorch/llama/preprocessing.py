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

import re
import os
import sys
import argparse
import configparser
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
import time 
from numpysocket import NumpySocket
import numpy as np
from threading import Thread

#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(dir_path + "/../../..")

done = 0

def load_hellaswag():
    hellaswag = datasets.load_dataset('hellaswag')
    validation = hellaswag['validation']
    validation_zeroshot = validation.filter(lambda example: example['split_type'] == 'zeroshot')
    #print("Hellaswag dataset load finish , len: " + str(len(validation_zeroshot)))
    return validation_zeroshot

def client_thread(conn, final_reqs):
    with conn:
        conn.sendall(final_reqs)
        #print("{} conn send finish", conn)
    conn.close()
    global done
    done += 1
    #print("done + 1 , cur done {}".format(done))
    if done == 4:
        return
        os._exit(1)
    return

class RequestInstance:
    def __init__(self, request_id, activity_label, context, endings, tokenizer, label):
        self.request_id = request_id
        self.activity_label = activity_label
        self.context = context
        self.ending1 = endings[0]
        self.ending2 = endings[1]
        self.ending3 = endings[2]
        self.ending4 = endings[3]
        self.endings = []
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
            [self.request_id,len(self.context), self.context, 0.0, ending_tok, self.label, i, len(ending_tok)] for i,ending_tok in enumerate(self.endings)            
        ]

def engineering_dataset(validation_zeroshot, tokenizer):
    requests = []
    for i, row in tqdm(enumerate(validation_zeroshot)):
        temp = RequestInstance(i, row['activity_label'], row['ctx'], row['endings'], tokenizer, int(row['label']))
        requests.extend(temp.requests)

    requests = sorted(requests, key=lambda x: x[1] + x[-1], reverse=True)

    max_tokens_40 = []
    max_tokens_80 = []
    max_tokens_120 = []
    max_tokens_170 = []

    for r in requests:
        ttt = r[1] + len(r[4])
        if ttt <= 40:
            max_tokens_40.append(r)
        elif ttt <= 80:
            max_tokens_80.append(r)
        elif ttt <= 120:
            max_tokens_120.append(r)
        elif ttt <= 170:
            max_tokens_170.append(r)

    max_batch_sizes_config = [248, 126, 69, 60][::-1] # FT 우리가 수정한 기본 버전으로 돌렸을 때 max [496, 252, 138, 120]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default="/llm/model/30B_converted_hf", 
                        help='directory where the tokenizer file is located.')
    args = parser.parse_args()
    tokenizer_path = args.tokenizer_path
    # sentencepiece needed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    # PGJ : For Hellaswag
    start_dataset = time.time()
    validation_zeroshot = []
    final_reqs = []
    res = []
    validation_zeroshot = load_hellaswag()
    final_reqs = engineering_dataset(validation_zeroshot, tokenizer)
    final_reqs = np.array(final_reqs, dtype=object)

    final_dataset = time.time()

    rank_num = 4
    list_sock = []
    threads = []
    port = 9020

    global done

    #print("only dataset preprocessing time {}".format(final_dataset - start_dataset))

    for i in range(rank_num):
        cur_port = port + i
        s = NumpySocket()
        s.bind(("",cur_port))
        s.listen()
        list_sock.append(s)
        #print("Server listening on %d"% (port + i))

    while True:
        for j in range(len(list_sock)):
            conn, addr = list_sock[j].accept()
            t = Thread(target=client_thread, args=(conn,final_reqs))
            t.start()
            threads.append(t)
        for j in range(len(threads)):
            threads[j].join()
        break
    #end_sending_time = time.time()
    s.close()

    return

if __name__ == "__main__":
	main()
