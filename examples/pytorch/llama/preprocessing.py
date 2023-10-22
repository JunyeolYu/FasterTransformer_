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
    validation_zeroshot = hellaswag['validation']
    # validation = hellaswag['validation']
    # validation_zeroshot = validation.filter(lambda example: example['split_type'] == 'zeroshot')
    #print("Hellaswag dataset load finish , len: " + str(len(validation_zeroshot)))
    return validation_zeroshot

def client_thread(conn, final_reqs):
    with conn:
        conn.sendall(final_reqs)
    conn.close()
    global done
    done += 1
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

    # requests = requests[:4000]
    requests = sorted(requests, key=lambda x: x[1] + x[-1], reverse=True)

    seq_ = [40,60,80,130,170]
    max_tokens_ = [[] for i in range(len(seq_))]

    for r in requests:
        ttt = r[1] + len(r[4]) -1
        idx = 0
        for i,s in enumerate(seq_):
            if ttt<=s:
                idx = i
                break
        max_tokens_[idx].append(r)

    max_batch_sizes_config = [250, 170, 126, 78, 60]
    # max_batch_sizes_config = [248, 128, 84, 60] # FT 우리가 수정한 기본 버전으로 돌렸을 때 max [496, 252, 138, 120]
    
    final_reqs = []
    
    for i, b in enumerate(max_batch_sizes_config):
        current_list = max_tokens_[i]
        print(seq_[i], current_list[0][1] + len(current_list[0][4]) -1,  current_list[-1][1] + len(current_list[-1][4]) -1)
        for j in range(0, len(current_list), b):
            final_reqs.append(current_list[j:j+b])
        print(f"seq:{seq_[i]}, req:{len(current_list)}, bs:{b}, # of batches:{len(current_list)/b}")
        
    return final_reqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default="/model/llama-30b/", 
                        help='directory where the tokenizer file is located.')
    args = parser.parse_args()
    tokenizer_path = args.tokenizer_path

    # sentencepiece needed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, legacy=False)

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