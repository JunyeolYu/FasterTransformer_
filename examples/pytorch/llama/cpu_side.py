


def calculate_accuracy(res):



    #set the torch backend to cpu
    #set the default number of intraop threads to 1
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
    return acc, nacc




def cpu_job(q, start_value, acc, nacc):

    from torch.nn.utils.rnn import pad_sequence
    import os
    import sys
    import argparse
    import configparser
    import timeit
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, LlamaTokenizer
    import datasets
    import re
    from tqdm import tqdm
    import time 
    import pickle
    from multiprocessing import Process, Queue, Value

    output_model = torch.load("/llm/ft_models/llama_30b_pp/4-gpu/output_layer.pt")
    output_model = output_model.cpu()
    output_model.to(torch.float32)
    output_model.eval()
    torch.set_num_threads(26)


    res = []
    while True:     
        if not q.empty():
            try:
                req_no = q.get_nowait()
                #req_no = pickle.loads(req_no)

            except:
                continue

            
            prompts, output_log_probs = req_no[0], req_no[1]
            output_log_probs = output_log_probs.to(torch.float32).cpu()

            multi_logits = torch.nn.functional.log_softmax(output_model(output_log_probs), dim=-1)
            
            _res = []
            for logits, prompt in zip(multi_logits, prompts):
                _input, ending, el = prompt[1]-1, prompt[4], prompt[-1]
                logits = logits[_input:_input+el].unsqueeze(0)  # [1, seq, vocab]
                ending = torch.tensor(ending, dtype=torch.long, device='cpu').view(1,-1,1)
                answer = torch.gather(logits, 2, ending).squeeze(-1).sum()  # [1, ]
                _res.append(answer)
            for prompt, ans in zip(prompts, _res):
                prompt[3] = ans
            
            res.extend(prompts)



        else:
            if start_value.value == 0:
                break
            else:
                continue




    #         multi_logits = torch.nn.functional.log_softmax(output_log_probs, dim=-1)
            
    #         


    start_cal = time.time()
    res = sorted(res, key=lambda x: x[0])
    a, na = calculate_accuracy(res)

    end = time.time()



    acc.value = a
    nacc.value = na


    ###################################


def main():


    q = Queue()
    start_value = Value('i', 0)
    
    start_value.value = 1
    acc = Value('d', 0.0)
    nacc = Value('d', 0.0)

    for i in final_reqs[:10]:
        d = (torch.rand(len(i), i[0]+len(i[4]), 6656), i)
        q.put(pickle.dumps(d))

    p = Process(target=cpu_job, args=(q, start_value, acc, nacc))

    p.start()

    time.sleep(5)
    p.join()


if __name__=="__main__":
    main()