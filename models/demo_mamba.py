from Mamba2 import Mamba, ModelArgs
from transformers import AutoTokenizer
import os
"""import transformers
print(transformers.__version__)

import torch
print(torch.__version__)""" 
# 估计是transformer版本问题，导致, 不重要
"""
Traceback (most recent call last):
  File "/home/kxg/gyy/Time-Series-Library-main/models/demo_mamba.py", line 20, in <module>
    model = Mamba.from_pretrained(pretrained_model_name)
  File "/home/kxg/gyy/Time-Series-Library-main/models/Mamba2.py", line 134, in from_pretrained
    state_dict = load_state_dict_hf(pretrained_model_name)
  File "/home/kxg/gyy/Time-Series-Library-main/models/Mamba2.py", line 124, in load_state_dict_hf
    return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
  File "/home/kxg/.conda/envs/LaST-copy/lib/python3.8/site-packages/torch/serialization.py", line 607, in load
    return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
  File "/home/kxg/.conda/envs/LaST-copy/lib/python3.8/site-packages/torch/serialization.py", line 880, in _load
    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
TypeError: 'weights_only' is an invalid keyword argument for Unpickler()

"""
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #还是解决不了打不开huggingface的问题，无法导入AutoTokenizer.from_pretrained
#export HF_ENDPOINT = "https://hf-mirror.com"
# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'
pretrained_model_name = 'mamba-370m'

model = Mamba.from_pretrained(pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b') # https://hf-mirror.com/EleutherAI/gpt-neox-20b可以下载
import torch
import torch.nn.functional as F

# 生成文本任务
def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 50,
             sample: bool = True,
             top_k: int = 40):
    model.eval()
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]
        
        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape
        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)
        
        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        
        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions

print(generate(model, tokenizer, 'Mamba is the'))
"""
from transformers import BertTokenizer
 
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
 
# Transformer's tokenizer - input_ids
sequence = "A Titan RTX has 24GB of VRAM"
print("Original sequence: ",sequence)
tokenized_sequence = tokenizer.tokenize(sequence)
print("Tokenized sequence: ",tokenized_sequence)
encodings = tokenizer(sequence)
encoded_sequence = encodings['input_ids']
print("Encoded sequence: ", encoded_sequence)
decoded_encodings=tokenizer.decode(encoded_sequence)
print("Decoded sequence: ", decoded_encodings)
"""