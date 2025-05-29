'''
run nn for fine tunning the prediciton of the low frequency signal component
In the initial test found the following optima hyperparameters
epochs = 256
MLP = As is given in the class object
batch_size = 128
lr = 0.0001

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import argparse

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Custom dataset class
class CSVArrayDataset(Dataset):
    def __init__(self, file_path, num_feat):
        self.data = pd.read_csv(file_path)  # Load CSV
        self.valid_batch_len = self.data.iloc[0]['valid_batch_len']

        self.col_pred = [f"pred_{i + 1}" for i in range(num_feat)]
        self.col_gt = [f"gt_{i + 1}" for i in range(num_feat)]
        self.meta_data = ['prompt_id', 'batch_id']

    def __len__(self):
        tmp_df = self.data[['prompt_id', 'batch_id']]
        grouped = tmp_df.groupby(['prompt_id', 'batch_id']).ngropus

        return grouped

    def __getitem__(self, idx):
        x = self.data.loc[idx * self.valid_batch_len:(idx + 1) * self.valid_batch_len - 1][self.col_pred].values.astype(
            np.float32)
        y = self.data.loc[idx * self.valid_batch_len:(idx + 1) * self.valid_batch_len - 1][self.col_gt].values.astype(
            np.float32)
        meta = self.data.loc[idx * self.valid_batch_len:(idx + 1) * self.valid_batch_len - 1][
            self.meta_data].values.astype(
            np.float32)

        return x, y, meta


# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, num_feat):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(num_features=num_feat)
        self.tanh_1 = nn.Tanh()

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(num_features=num_feat)
        self.tanh_2 = nn.Tanh()

        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(num_features=num_feat)
        self.tanh_3 = nn.Tanh()

        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(num_features=num_feat)
        self.tanh_4 = nn.Tanh()

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(num_features=num_feat)
        self.tanh_5 = nn.Tanh()

        self.fc6 = nn.Linear(64, output_size)
        # self.tanh_5 = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.tanh_1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.tanh_2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.tanh_3(x)

        # x = self.fc4(x)
        # x = self.bn4(x)
        # x = self.tanh_4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.tanh_5(x)

        x = self.fc6(x)

        return x


def get_df(pred, gt, meta, round, num_feat):
    array = np.concatenate([meta, pred, gt], axis=2)

    num_batches = array.shape[0]
    valid_batch_len = array.shape[1]

    # Sample array (simulated with random values)
    col_pred = [f"pred_{i + 1}" for i in range(num_feat)]
    col_gt = [f"gt_{i + 1}" for i in range(num_feat)]

    # Reshape to (128 × 37, 4) -> (4736, 4)
    # reshaped_array = array.reshape(-1, num_feat * 2)
    reshaped_array = np.concatenate(array, axis=0)

    # Generate the ID column (1 for first 37, 2 for next 37, ..., 128 for last 37)
    ids = np.repeat(np.arange(1, num_batches + 1), valid_batch_len)  # 128 groups, each repeated 37 times

    # Create DataFrame
    cols = ['prompt_id', 'batch_id']
    cols.extend(col_pred)
    cols.extend(col_gt)

    # print(cols)
    df = pd.DataFrame(reshaped_array, columns=cols)

    # Insert ID column at the beginning
    # df.insert(0, 'ID', ids)
    # df['round'] = round

    return df


# open .txt file to save the data. Since we are working with text prompts, it is preferrable to go for .txt file than
# csv files
path_curr = Path.cwd()
path_pre_process_nn = path_curr.parent.__str__() + '/data_4_pre_process_nn'
path_nn_out = path_curr.parent.__str__() + '/data_5_nn_out'
if not os.path.exists(path_nn_out):
    os.makedirs(path_nn_out)

# Create the parser
# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example.")

# Add arguments
parser.add_argument("--train_len", type=int, help="Train length")
parser.add_argument("--test_len", type=int, help="Test length", required=True)
parser.add_argument("--dataset", type=str, help="Dataset", required=True)
parser.add_argument("--freq", type=str, help="Whether it is high or low frequency", required=True)
parser.add_argument("--cut_off", type=float, default=15.0, help="cutoff_frequencey used", required=False)
parser.add_argument("--num_cutoffs", type=int, default=1, help="Number of frequency components", required=False)
parser.add_argument("--alpha", type=float, default=0.6,
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

# # # temporarily add the values
# train_len = 48
# test_len = 48
# dataset = 'electricity'
# freq = 'low'
# cut_off = float(2.5)
# alpha = 0.6
# num_cutoffs = 1
# num_freq_comps = 1
# num_feat = 1
# spec_feat = 0
# max_tokens = 200
# model_name = 'GPT-4o-mini'
# limit = 50
# exp = 'run_individual_feat'


file_name = f'{exp}_{dataset}_{model_name}_{train_len}_{test_len}_{num_cutoffs}_{alpha}_{num_feat}_{spec_feat}_{max_tokens}_{freq}'
path_data_in = path_pre_process_nn + '/' + file_name + '.csv'

# Load dataset
dataset = CSVArrayDataset(path_data_in, num_feat)

data_csv = pd.read_csv(path_data_in)
prompts = np.unique(data_csv['prompt_id'].values)
train_prompt_ids = int(np.percentile(prompts, 70))

df_train = data_csv.loc[data_csv['prompt_id'] <= train_prompt_ids]
df_test = data_csv.loc[data_csv['prompt_id'] > train_prompt_ids]

total_data = len(data_csv.groupby(['prompt_id', 'batch_id']))
train_data = len(df_train.groupby(['prompt_id', 'batch_id']))
test_data = len(df_test.groupby(['prompt_id', 'batch_id']))

# Condition: Split based on 'id' column (Example: Train if id < 500, Test otherwise)
train_indices = list(np.arange(0, train_data))
test_indices = list(np.arange(train_data, total_data))

# Create subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # No shuffle
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)  # No shuffle

# Define model, loss, and optimizer
input_size = len(dataset[0][0])  # Infer data_1_input_data_for_ablation size from first row
model = SimpleNN(input_size=input_size,
                 output_size=input_size,
                 num_feat=num_feat).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 128

for epoch in range(epochs):
    for inputs, targets, meta in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = inputs.permute(0, 2, 1)

        optimizer.zero_grad()
        outputs = model(inputs)

        outputs = outputs.permute(0, 2, 1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluation
model.eval()
test_loss = 0

list_pred_gt = []

i = 0

with torch.no_grad():
    for inputs, targets, meta in test_loader:
        i += 1

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = inputs.permute(0, 2, 1)

        outputs = model(inputs)

        outputs = outputs.permute(0, 2, 1)

        loss = criterion(outputs, targets)
        test_loss += loss.item()

        ouput_arr = outputs.cpu().detach().numpy()
        gt_arr = targets.cpu().detach().numpy()
        meta_arr = meta.numpy()

        df = get_df(ouput_arr, gt_arr, meta_arr, round=i, num_feat=num_feat)
        list_pred_gt.append(df)

final_df = pd.concat(list_pred_gt, axis=0)

final_df.to_csv(path_nn_out + '/' + file_name + '.csv',
                index=False, )

print(f"Test Loss: {test_loss / len(test_loader)}")
print("Training complete.")
