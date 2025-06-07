import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse
import re


def check_integer_part(val, expected_value):
    # Get the decimal part of the numbers
    decimal_part = val - np.floor(val)  # or arr % 1
    int_part = val - decimal_part

    if int_part == 0:
        is_bad_segment = False
        is_zero_int = True
    else:
        if int_part != expected_value:
            is_bad_segment = True
            is_zero_int = False
        else:
            is_bad_segment = False
            is_zero_int = False

    return is_bad_segment, is_zero_int

def remove_offsets(data, num_of_feat, spec_feat):
    if spec_feat != 100:
        offsets = [[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5][spec_feat]]  # sel
    else:
        offsets = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5][
                  :num_of_feat]  # select only the offsets related to the number of features
    offsets = np.asarray(offsets)

    # validate and change the range. # if the ranges are not in the range change the first decimal points
    is_bad_segment = False
    is_zero_int = False
    for f in range(num_of_feat):
        is_bad_segment, is_zero_int = check_integer_part(data[f],
                                            expected_value=int(offsets[f]))

    if is_zero_int:
        return data, is_bad_segment
    else:
        return (data - offsets) * 2, is_bad_segment


def descale_vales(data, max_val):
    data = data * max_val

    return data


def is_numeric(s):
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


def get_fetures_data(path_llm_out_text, path_max_file_csv, limit, num_feat, spec_feat):
    # text distribution with the horizontal lines
    f = open(path_llm_out_text, 'r')

    # # read the max value distributions
    max_vals = pd.read_csv(path_max_file_csv).values[:, : num_feat]

    lines = f.readlines()

    prompts = 0
    c = 0

    list_all_prompts_x = []
    list_all_prompts_y = []
    list_all_preds = []

    list_all_prompts_x_descaled = []
    list_all_prompts_y_descaled = []
    list_all_preds_descaled = []

    list_x_values = []
    list_y_values = []
    list_preds = []

    list_x_values_descaled = []
    list_y_values_descaled = []
    list_preds_descaled = []

    while c < len(lines):

        l = lines[c]
        if 'consider' in l:
            prompts += 1
            c += 1  # increment to the next line which contains the numerical data
            l = lines[c][:-1]  # last charcater is '\n'
            # next line can be either numeric or "print test_prompt". check for that
            if 'print test_prompt' in l:
                c -= 1
                is_data = False
            else:
                is_data = True
            while (is_data):
                real_x_values = l.split(',')
                real_x_values = np.asarray(real_x_values).astype(float)
                real_x_values, is_bad_segment = remove_offsets(real_x_values, num_feat, spec_feat)

                # de scale the values
                real_x_values_descaled = descale_vales(real_x_values, max_vals[prompts])

                real_x_values = np.insert(real_x_values, 0, prompts)
                real_x_values_descaled = np.insert(real_x_values_descaled, 0, prompts)

                list_x_values.append(real_x_values)
                list_x_values_descaled.append(real_x_values_descaled)

                c += 1  # increment to the next line which contains the numerical data
                l = lines[c][:-1]  # last charcater is '\n'
                # next line can be either numeric or "print test_prompt". check for that
                if 'print test_prompt' in l:
                    c -= 1
                    is_data = False

                    break

        elif 'print test_prompt' in l:
            c += 1
            l = lines[c][:-1]

            if 'Print predictions' in l:
                c -= 1
                is_data = False
            else:
                is_data = True
            while (is_data):
                real_y_values = l.split(',')
                real_y_values = np.asarray(real_y_values).astype(float)
                real_y_values, is_bad_segment = remove_offsets(real_y_values, num_feat, spec_feat)

                # descale real y values
                real_y_values_descaled = descale_vales(real_y_values, max_vals[prompts])

                real_y_values = np.insert(real_y_values, 0, prompts)
                real_y_values_descaled = np.insert(real_y_values_descaled, 0, prompts)

                list_y_values.append(real_y_values)
                list_y_values_descaled.append(real_y_values_descaled)

                c += 1  # increment to the next line which contains the numerical data
                l = lines[c][:-1]  # last charcater is '\n'
                # next line can be either numeric or "print test_prompt". check for that
                if 'Print predictions' in l:
                    c -= 1
                    is_data = False
                    break
                # l = l[:-1]

        elif 'Print predictions' in l:
            c += 1
            l = lines[c]
            is_bad_segment = False
            sample = 0
            while (not 'print prompt' in l):
                sample += 1
                l = lines[c]
                list_temp = []
                list_temp_descaled = []

                while len(l) > 0 and l != '\n':
                    l = l[:-1]
                    arr = np.asarray(l.split(','))
                    arr = np.char.replace(arr, " ", "")  # Remove all spaces
                    mask = np.vectorize(is_numeric)(arr)
                    filtered_arr = arr[mask]

                    if len(filtered_arr) == num_feat:
                        filtered_arr = filtered_arr.astype(float)
                        filtered_arr, is_bad_segment = remove_offsets(filtered_arr, num_feat, spec_feat)
                        filtered_arr = filtered_arr.astype(float)

                        filtered_arr_descaled = descale_vales(filtered_arr, max_vals[prompts])

                        filtered_arr = np.insert(filtered_arr, [0, 0], [prompts, sample])
                        filtered_arr_descaled = np.insert(filtered_arr_descaled, [0, 0], [prompts, sample])
                        list_temp.append(filtered_arr)
                        list_temp_descaled.append(filtered_arr_descaled)

                    c += 1
                    if c < len(lines):
                        l = lines[c]
                    else:
                        if not is_bad_segment:
                            list_preds.append(list_temp)
                            list_preds_descaled.append(list_temp_descaled)
                        else:
                            print('Bad segment detected')
                        break

                    if ('\n' in l) and (len(l) == 1):
                        if not is_bad_segment:
                            list_preds.append(list_temp)
                            list_preds_descaled.append(list_temp_descaled)
                        else:
                            print('Bad segment detected')
                        break

                c += 1
                if c < len(lines):
                    l = lines[c]
                else:
                    break
            c -= 1
        c += 1

        if len(list_x_values) != 0 and len(list_y_values) != 0 and len(list_preds) != 0:
            list_all_prompts_x.append(list_x_values)
            list_all_prompts_y.append(list_y_values)
            list_all_preds.append(list_preds)

            list_all_prompts_x_descaled.append(list_x_values_descaled)
            list_all_prompts_y_descaled.append(list_y_values_descaled)
            list_all_preds_descaled.append(list_preds_descaled)

            list_x_values = []
            list_y_values = []
            list_preds = []

            list_x_values_descaled = []
            list_y_values_descaled = []
            list_preds_descaled = []

            if prompts == limit:
                break

    return (list_all_prompts_x, list_all_prompts_y, list_all_preds,
            list_all_prompts_x_descaled, list_all_prompts_y_descaled, list_all_preds_descaled)


def convert_list_data_to_df_real(list_data, num_feat):
    col = ['prompt_id']
    col.extend([f"f{i}" for i in range(1, num_feat + 1)])
    list_real = []
    for i in range(len(list_data)):
        data = np.asarray(list_data[i])
        df_data = pd.DataFrame(data,
                               columns=col)

        list_real.append(df_data)

    df_real = pd.concat(list_real, axis=0)

    return df_real


def convert_list_data_to_df_pred(list_data, num_feat):
    col = ['prompt_id', 'batch_id']
    col.extend([f"f{i}" for i in range(1, num_feat + 1)])
    list_pred = []
    for i in range(len(list_data)):
        prompt_data = list_data[i]
        for j in range(len(prompt_data)):
            batch_data = np.asarray(prompt_data[j])
            if batch_data.shape[0] == 0:
                continue
            df_data = pd.DataFrame(batch_data,
                                   columns=col)
            list_pred.append(df_data)

    df_pred = pd.concat(list_pred, axis=0)

    return df_pred


# store the post processed data
def store_post_processed_data(path_out, real_x, real_y, pred_y, num_feat):
    print(path_out)
    df_real_x = convert_list_data_to_df_real(real_x, num_feat)
    df_real_x.to_csv(path_out + '/real_x.csv', float_format='%.3f', index=False)

    df_real_y = convert_list_data_to_df_real(real_y, num_feat)
    df_real_y.to_csv(path_out + '/real_y.csv', float_format='%.3f', index=False)

    df_pred = convert_list_data_to_df_pred(pred_y, num_feat)
    df_pred.to_csv(path_out + '/pred_y.csv', float_format='%.3f', index=False)

    return


# store the post processed data
def store_post_processed_data_descaled(path_out, real_x, real_y, pred_y, num_feat):
    df_real_x = convert_list_data_to_df_real(real_x, num_feat)
    df_real_x.to_csv(path_out + '/real_x_renormalized.csv', float_format='%.3f', index=False)

    df_real_y = convert_list_data_to_df_real(real_y, num_feat)
    df_real_y.to_csv(path_out + '/real_y_renormalzied.csv', float_format='%.3f', index=False)

    df_pred = convert_list_data_to_df_pred(pred_y, num_feat)
    df_pred.to_csv(path_out + '/pred_y_renormalized.csv', float_format='%.3f', index=False)

    return


# # # Create the parser
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
path_curr = Path.cwd()
path_parent = path_curr.parent.parent.__str__()
path_max_val = path_parent + '/data_uni/data_1_input_data'
path_llm_out = path_parent + '/data_uni/data_2_output_data'
path_llm_post_processed = path_parent + '/data_uni/data_3_llm_post_processed'

llm_output = f'{exp}_{dataset}_{model_name}_{train_len}_{test_len}_{num_cutoffs}_{alpha}_{num_feat}_{spec_feat}_{max_tokens}_{freq}'
path_llm_out_text = path_llm_out + '/' + llm_output + '.txt'

'''
This is a fix for reading the max file for the original and low and high frequencey components.
'''
if freq == 'original':
    path_max_file_csv = path_max_val + '/' + dataset + '/' + f'{train_len}_{test_len}' + '/' + f'{dataset}_{0.4}_{freq}_max.csv'
else:
    path_max_file_csv = path_max_val + '/' + dataset + '/' + f'{train_len}_{test_len}' + '/' + f'{dataset}_{alpha}_{freq}_max.csv'

(list_all_prompts_x, list_all_prompts_y, list_all_preds,
 list_all_prompts_x_descaled, list_all_prompts_y_descaled, list_all_preds_descaled) = get_fetures_data(
    path_llm_out_text,
    path_max_file_csv,
    limit=limit,
    num_feat=num_feat,
    spec_feat=spec_feat)

# store these outputs for later MLP and Gaussian transformation training.
path_out = path_llm_post_processed + '/' + llm_output
if not os.path.exists(path_out):
    os.makedirs(path_out)

store_post_processed_data(path_out, list_all_prompts_x, list_all_prompts_y, list_all_preds, num_feat)
store_post_processed_data_descaled(path_out, list_all_prompts_x_descaled, list_all_prompts_y_descaled,
                                   list_all_preds_descaled, num_feat)
