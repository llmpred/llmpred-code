import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse

# open .txt file to save the data. Since we are working with text prompts, it is preferrable to go for .txt file than
# csv files
path_curr = Path.cwd()
path_parent = path_curr.parent.parent.__str__()
path_real_x = path_parent+ '/data_uni/data_1_input_data'
path_pred_y = path_parent+ '/data_uni/data_4_pre_process_nn'
path_out = path_parent + '/data_uni/data_6_gaussian_transform'

if not os.path.exists(path_out):
    os.makedirs(path_out)

# # Create the parser
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


# extract the real_x distribution
offsets = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
# offset_feat = offsets[:num_feat]

path_data = path_real_x + '/' + dataset + '/' + str(train_len) + '_' + str(test_len) + '/' + dataset + '_' + str(
    alpha) + '_high.csv'
prompts = pd.read_csv(path_data).values
list_train_data = []
for prompt in prompts:
    lines = prompt[0].split('\n')[1:train_len + 1]
    for line in lines:
        if spec_feat == 100:
            arr = np.fromstring(line, sep=',', dtype=np.float32)[:num_feat]
            offset_feat = offsets[:num_feat]
        else:
            arr = np.fromstring(line, sep=',', dtype=np.float32)[spec_feat]
            offset_feat = offsets[spec_feat]
        arr = (arr - offset_feat) * 2
        list_train_data.append(arr)

arr_train_data = np.asarray(list_train_data)

# extract the mean and std of the distributions
mean_train_data = np.mean(arr_train_data, axis=0)
std_train_data = np.std(arr_train_data, axis=0)

# extract the pred_y distribution
pred_y_csv = f'{exp}_{dataset}_{model_name}_{train_len}_{test_len}_{num_cutoffs}_{alpha}_{num_feat}_{spec_feat}_{max_tokens}_{freq}'
path_in = path_pred_y + '/' + pred_y_csv + '.csv'
col_pred = [f"pred_{i + 1}" for i in range(num_feat)]
df = pd.read_csv(path_in)
arr_pred_data = df[col_pred].values

mean_pred_data = np.mean(arr_pred_data, axis=0)
std_pred_data = np.std(arr_pred_data, axis=0)

pred_new = ((arr_pred_data - mean_pred_data) / std_pred_data) * std_train_data + mean_train_data

for f in range(num_feat):
    df[col_pred[f]] = pred_new[:, f]

df.to_csv(path_out + '/' + pred_y_csv + '.csv', index=False)
