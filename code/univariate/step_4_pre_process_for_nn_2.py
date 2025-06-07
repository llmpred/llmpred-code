'''
Chamara
Preprocess post processed low frequency output of the llm for NN training.
'''
import os.path

import pandas as pd
import numpy as np
from pathlib import Path
import statistics
import argparse

def validate_length(pred):
    df = pred[['prompt_id', 'batch_id']]
    grouped = df.groupby(['prompt_id', 'batch_id'])
    # Iterate through groups
    list_lens = []
    for (key1, key2), group in grouped:
        list_lens.append(len(group))

    mode_value = statistics.mode(list_lens)

    return mode_value


def truncate_or_pad(arr, length):
    """Truncates or pads a NumPy array to a fixed length with 0."""
    arr = np.array(arr)  # Ensure input is a NumPy array
    if len(arr) > length:
        return arr[:length]  # Truncate if longer
    else:
        return np.pad(arr, (0, length - len(arr)), mode='constant', constant_values=0)  # Pad with 0s


def adjust_dataframe(df, t):
    """
    Adjusts the DataFrame by either truncating it to length t if it exceeds t,
    or repeating the last row until it reaches length t.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        t (int): Threshold length

    Returns:
        pd.DataFrame: Adjusted DataFrame
    """
    if len(df) < t:
        last_row = df.iloc[-1:]  # Get the last row as a DataFrame
        repeat_times = t - len(df)  # Number of times to repeat
        df = pd.concat([df] + [last_row] * repeat_times, ignore_index=True)
    else:
        df = df.iloc[:t]  # Truncate to threshold

    return df


def group_and_sort(pred, gt, num_feat):
    prompts = np.unique(gt['prompt_id'].values)

    num_gt_vales = 0
    num_pred_vales = 0

    list_df_feats = []

    list_dfs = []

    valid_batch_len = validate_length(pred)

    for p in prompts:
        gt_prompt = gt.loc[gt['prompt_id'] == p]
        pred_prompt = pred.loc[pred['prompt_id'] == p]

        pred_prompt_batch = pred_prompt[['batch_id']]

        batches = pred_prompt_batch['batch_id'].unique()

        for b in batches:
            pred_batch = pred_prompt.loc[pred_prompt['batch_id'] == b]

            # pad or truncate the values baed on the valid batch len
            pred_batch_validated = adjust_dataframe(pred_batch, valid_batch_len)
            gt_batch_validated = adjust_dataframe(gt_prompt, valid_batch_len)

            # check whether the prompt ids are the same
            assert pred_batch_validated.iloc[0]['prompt_id'] == gt_batch_validated.iloc[0]['prompt_id']

            # create new dataframe extracting the columns
            new_df = pred_batch_validated[['prompt_id', 'batch_id']]
            new_df['valid_batch_len'] = valid_batch_len
            for feat in range(num_feat):
                new_df['pred_' + str(feat + 1)] = pred_batch_validated[['f' + str(feat + 1)]].values
                new_df['gt_' + str(feat + 1)] = gt_batch_validated[['f' + str(feat + 1)]].values


            list_dfs.append(new_df)

    final_df= pd.concat(list_dfs, axis=0, ignore_index=True)

    return final_df


# open .txt file to save the data. Since we are working with text prompts, it is preferrable to go for .txt file than
# csv files
path_curr = Path.cwd()
path_parent = path_curr.parent.parent.__str__()
path_llm_post_processed = path_parent + '/data_uni/data_3_llm_post_processed'
path_pre_process_nn = path_parent + '/data_uni/data_4_pre_process_nn'

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

llm_output = f'{exp}_{dataset}_{model_name}_{train_len}_{test_len}_{num_cutoffs}_{alpha}_{num_feat}_{spec_feat}_{max_tokens}_{freq}'
path_in = path_llm_post_processed + '/' + llm_output

pred_data = pd.read_csv(path_in + '/pred_y.csv')
gt_data = pd.read_csv(path_in + '/real_y.csv')

final_df = group_and_sort(pred_data, gt_data, num_feat)

# folder to contain the data feature by feature for the nn and
path_out = path_pre_process_nn + '/' + llm_output + '.csv'
print(path_out)
final_df.to_csv(path_out, index=False)
