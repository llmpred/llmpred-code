import pandas as pd
import numpy as np
import os
import tqdm
from pathlib import Path
import gc
import sys
import time
import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


# create the path directories
path_curr = Path.cwd()
path_parent = path_curr.parent.parent.__str__()
path_in = path_parent + '/data/data_1_input_data'
path_model = path_parent + '/models'
path_out = path_parent + '/data/data_2_output_data'
path_out_meta = path_parent + '/data/data_2_output_meta'

# create the folder if not created for the generator ouput
if not os.path.exists(path_out):
    os.makedirs(path_out)
# create the folder if not created to store meta results
if not os.path.exists(path_out_meta):
    os.makedirs(path_out_meta)

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example.")

# Add arguments
parser.add_argument("--train_len", type=int, help="Train length")
parser.add_argument("--test_len", type=int, help="Test length", required=True)
parser.add_argument("--dataset", type=str, help="Dataset", required=True)
parser.add_argument("--freq", type=str, help="Whether it is high or low frequency", required=True)
parser.add_argument("--cut_off", type=float, default=15.0, help="cutoff_frequencey used", required=False)
parser.add_argument("--num_cutoffs", type=int, default=1, help="Number of frequency components", required=False)
parser.add_argument("--alpha", type=float, default=0.7,
                    help="Alpha value used to combine the mse and 1/cosine sim metrics to decide the cutoff freq",
                    required=False)
parser.add_argument("--num_feat", type=int, help="Number of features", required=True)
parser.add_argument("--spec_feat", type=int, default=100, help="Specific feature to be select", required=False)
parser.add_argument("--max_tokens", type=int, help="maximum tokens to be predicted", required=True)
parser.add_argument("--model_name", type=str, help="models to be used", required=True)
parser.add_argument("--limit", type=int, help="limit of prompts", required=True)
parser.add_argument("--num_samples", type=int, default=8, help="batch size", required=False)
parser.add_argument("--exp", type=str, help="name of the experiment conducted", required=True)

# Parse the arguments
args = parser.parse_args()

train_len = args.train_len
test_len = args.test_len
dataset = args.dataset
freq = args.freq
cut_off = args.cut_off
num_cutoffs = args.num_cutoffs
alpha = args.alpha
num_feat = args.num_feat
spec_feat = args.spec_feat
max_tokens = args.max_tokens
num_samples = args.num_samples
model_name = args.model_name
limit = args.limit
exp = args.exp

# open .txt file to save the data. Since we are working with text prompts, it is preferrable to go for .txt file than
# csv files
filt_output = f'{exp}_{dataset}_{model_name}_{train_len}_{test_len}_{num_cutoffs}_{alpha}_{num_feat}_{spec_feat}_{max_tokens}_{freq}'
print(path_out + '/' + filt_output + '.txt')
if os.path.exists(path_out + '/' + filt_output + '.txt'):
    print(f'Already executed the script {filt_output}')
    sys.exit(0)

print(f'This model has not been run {filt_output}')
f = open(path_out + '/' + filt_output + '.txt', mode='w+')


class Model(nn.Module):

    def __init__(self, path_model, model_name):
        super().__init__()

        self.device = f"cuda:{0}"

        # LLM model for tokenizer
        self.llama_tokenizer = AutoTokenizer.from_pretrained(path_model + '/' + model_name)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        # LLM model
        self.llama = AutoModelForCausalLM.from_pretrained(
            path_model + '/' + model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
        )

    def forward(self, prompt, bs, max_tokens):
        batch = self.llama_tokenizer([prompt], return_tensors="pt")
        # print(batch['input_ids'].shape)
        num_tokens_per_input = [len(t) for t in batch["input_ids"]]

        batch = {k: v.repeat(bs, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}
        num_input_ids = batch['input_ids'].shape[1]

        max_tokens = num_tokens_per_input[0] + 50

        generate_ids = self.llama.generate(**batch,
                                           max_new_tokens=max_tokens,  # Limits only generated output
                                           do_sample=True,  # Enables randomness
                                           temperature=1.0,  # Adjusts creativity (increase if still repeating)
                                           top_p=0.9,  # Nucleus sampling for diverse output
                                           renormalize_logits=False,
                                           )

        gen_strs = self.llama_tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return gen_strs


def add_front_prompt(prompt):
    FRONT_PROMPT = "consider the vertical distribution patterns of the data and predict the next few lines"
    prompt = FRONT_PROMPT + '\n' + prompt

    return prompt


def get_mean_of_segments(list_data, seg_size, num_segs):
    list_mean_val_plus_raw_data = []
    segment_size = seg_size
    num_segments = len(list_data) // segment_size

    for i in range(num_segments - num_segs):
        d = list_data[i * segment_size:(i + 1) * segment_size]
        list_mean_val_plus_raw_data.append(np.mean(np.array(d).astype(float)))

    d = list_data[(num_segments - num_segs) * (segment_size):]
    d = [float(i) for i in d]
    list_mean_val_plus_raw_data.extend(d)

    formatted_list = [f"{x:.2f}" for x in list_mean_val_plus_raw_data]
    formatted_str = ",".join(formatted_list)

    return formatted_str


def get_train_test_prompts_single_feat(prompt, train_len, test_len):
    front_prompt = 'consider the vertical and horizontal distribution patterns of the data and predict the next few lines'
    list_row = prompt.split(',')  # for the single feature we make it as ','.
    list_train_prompts = list_row[:train_len]
    list_test_prompts = list_row[train_len:train_len + test_len]

    # train_prompt_mean_extracted = get_mean_of_segments(list_train_prompts, seg_size, num_segs)

    train_prompt = ",".join(list_train_prompts)
    train_prompt = front_prompt + '\n' + train_prompt

    test_prompt = ",".join(list_test_prompts)

    return train_prompt, test_prompt  # , train_prompt_mean_extracted


# get the range of fetures
def get_train_test_prompts_range_of_feat(prompt, train_len, test_len):
    front_prompt = 'Consider the vertical distribution. Predict the nex few lines. INTEGER component of the value SHOULD be SAME as the train data. ONLY provide numerical values'
    list_row = prompt.split('\n')

    list_train_prompts = list_row[:train_len]
    list_test_prompts = list_row[train_len:train_len + test_len]

    train_prompt = "\n".join(list_train_prompts)
    train_prompt = front_prompt + '\n' + train_prompt

    test_prompt = "\n".join(list_test_prompts)

    return train_prompt, test_prompt


def convert_str_to_val(pred_str):
    list_vals = pred_str[0].split('\n')
    list_rows = []
    for row in list_vals:
        # print(row)
        row = row.split(',')
        row = np.asarray(row).astype(float)
        list_rows.append(row)

    pred_values = np.asarray(list_rows)

    return pred_values


def select_individual_feat(prompt, feat):
    prompt = prompt[0][1:]
    list_prt = prompt.split('\n')[:-1]  # remove the last " character
    list_feat = []
    for p in list_prt:
        p = p.split(',')[feat]
        list_feat.append(p)
    prompt = ",".join(list_feat)

    return prompt


# select the features requried. If the spec_feat defined to 100
# the function will return the range of features considering multivariate generation.
# otherwise select a unique feature value
def select_range_of_feat(prompt, feats, spec_feat):
    final_prompt = ''
    prompt = prompt[0][1:]
    list_prt = prompt.split('\n')[:-1]  # remove the last " character

    for p in list_prt:
        if spec_feat == 100:
            p = p.split(',')[:feats]
        else:
            p = [p.split(',')[spec_feat]]
        temp_prompt = ",".join(p)
        temp_prompt = temp_prompt + '\n'
        final_prompt += temp_prompt

    return final_prompt


print('loading the model')
model = Model(path_model=path_model,
              model_name=model_name)
list_times = []
with torch.no_grad():
    path_data = path_in + '/' + dataset + '/' + str(train_len) + '_' + str(
        test_len) + '/' + dataset + '_' + str(alpha) + '_' + freq + '.csv'
    print(f'Read data from {path_data}')
    df_prompts = pd.read_csv(path_data)

    # generate the prompts from the data
    prompts = df_prompts.values

    num_prompts = len(prompts)
    for idx in tqdm.tqdm(range(num_prompts)):
        prompt = prompts[idx]

        # select the range of feats and get the prompt
        print("select the range of feat. Start the begining")
        prompt = select_range_of_feat(prompt, feats=num_feat, spec_feat=spec_feat)

        # removed the pe
        train_prompt, test_prompt = get_train_test_prompts_range_of_feat(prompt,
                                                                         train_len,
                                                                         test_len)
        # store the data for the including the historical prompt, future ground truth prompt (mentioned as 'test_prompt'
        print('print prompt')
        print(train_prompt)
        f.write('print prompt\n')
        f.write(train_prompt)
        f.write('\n')
        print('print test_prompt')
        print(test_prompt)
        f.write('print test_prompt\n')
        f.write(test_prompt)
        f.write('\n')

        bs = num_samples

        t1 = time.time()
        pred_str = model(train_prompt, bs, max_tokens)
        t2 = time.time()
        list_times.append(t2 - t1)

        print('Print predictions')
        f.write('Print predictions')
        for i in range(len(pred_str)):
            print(pred_str[i])
            f.write(pred_str[i])
            f.write('\n')
        print('')
        f.write('\n')

        if idx == limit:
            break

    f.flush()
    f.close()

    with open(path_out_meta + '/' + filt_output + '.txt',
              'w+') as f_m:
        f_m.writelines(str(item) + "\n" for item in list_times)

    torch.cuda.empty_cache()
    gc.collect()
