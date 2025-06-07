import numpy as np
import pandas as pd
from pathlib import Path


def read_data(path_in, path_max, dataset, dataset_max, size, feat, num_of_feat, alpha, model, freq):
    if dataset in ['weather', 'electricity'] and model in ['llama_7b', 'deepseek_7b'] and size == '96':
        num_of_feat = 6

    file_name = 'run_multivariate_' + dataset + '_' + model + '_' + size + '_' + size + '_1_' + str(
        alpha) + '_' + str(num_of_feat) + '_' + str(feat) + '_200_' + freq

    path_f = path_in + '/' + file_name + '.csv'

    file_name_max = dataset_max + '_' + str(alpha) + '_' + freq + '_max.csv'
    path_max_f = path_max + '/' + dataset + '/' + size + '_' + size + '/' + file_name_max

    # read the path_f
    df = pd.read_csv(path_f)

    # read max data
    df_max = pd.read_csv(path_max_f)

    return df, df_max, file_name


def denormalize(df, df_max, num_of_feat):
    prompt_ids = df['prompt_id'].unique()

    list_df = []
    for p_id in prompt_ids:
        df_sub = df[(df['prompt_id'] == p_id)]
        for feat in range(1, num_of_feat + 1):
            constant_max = df_max.at[int(p_id), 'F' + str(int(feat))]

            df_sub['pred_' + str(feat)] = df_sub['pred_' + str(feat)] * constant_max
            df_sub['gt_' + str(feat)] = df_sub['gt_' + str(feat)] * constant_max

        list_df.append(df_sub)

    final_df = pd.concat(list_df)

    return final_df


def get_min_len(a, b):
    min_len = min(len(a), len(b))

    # Truncate both arrays
    a_trunc = a[:min_len]
    b_trunc = b[:min_len]

    return a_trunc, b_trunc


def combine_taces(df_l, df_h, num_of_feat):
    # start from df_l
    unique_prompts = df_l['prompt_id'].unique()

    list_dfs = []
    for pr in unique_prompts:
        df_l_pr = df_l[(df_l['prompt_id'] == pr)]
        df_h_pr = df_h[(df_h['prompt_id'] == pr)]

        unique_batches = df_l_pr['batch_id'].unique()
        for ba in unique_batches:
            df_l_pr_ba = df_l_pr[(df_l_pr['batch_id'] == ba)]
            df_h_pr_ba = df_h_pr[(df_h_pr['batch_id'] == ba)]

            for feat in range(1, num_of_feat + 1):

                if df_h_pr_ba.shape[0] > 0 and df_l_pr_ba.shape[0] > 0:
                    gt_l = df_l_pr_ba['gt_' + str(feat)].values
                    gt_h = df_h_pr_ba['gt_' + str(feat)].values
                    gt_l, gt_h = get_min_len(gt_l, gt_h)
                    gt_c = gt_l + gt_h

                    pred_l = df_l_pr_ba['pred_' + str(feat)].values
                    pred_h = df_h_pr_ba['pred_' + str(feat)].values
                    pred_l, pred_h = get_min_len(pred_l, pred_h)
                    pred_c = pred_l + pred_h

                    data = np.concatenate((pred_c.reshape([-1, 1]),
                                           gt_c.reshape([-1, 1])),
                                          axis=1)

                    if feat == 1:
                        df_new = pd.DataFrame(columns=['pred_' + str(feat), 'gt_' + str(feat)],
                                              data=data)
                        df_new['pompt_id'] = pr
                        df_new['batch_id'] = ba
                    else:
                        df_new['pred_' + str(feat)] = pred_c
                        df_new['gt_' + str(feat)] = gt_c

                    if feat == num_of_feat:
                        list_dfs.append(df_new)

    final_df = pd.concat(list_dfs)

    return final_df


def measure_metrics():
    return


if __name__ == '__main__':

    path_curr = Path.cwd()
    path_parent = path_curr.parent.parent.__str__()
    path_in_mlp = path_curr + '/data_multi/data_5_nn_out'
    path_in_gauss = path_curr + '/data_multi/data_6_gaussian_transform'
    path_max = path_curr + '/data_multi/data_1_input_data'
    path_out = path_curr + '/data_multi/data_7_combine'

    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity']
    datasets_max = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity']
    num_of_feats = [6, 6, 6, 6, 9, 9]
    models = ['llama_3b', 'llama_7b', 'deepseek_7b', 'GPT-4o-mini']
    alpha = 0.7

    # read data
    for d, dataset in enumerate(datasets):
        print(dataset)
        for s, size in enumerate(['48', '96']):
            for model in models:
                # for f, feat in enumerate(feats):
                feat_data_l, max_data_l, file_name = read_data(path_in=path_in_mlp,
                                                               path_max=path_max,
                                                               dataset=dataset,
                                                               dataset_max=datasets_max[d],
                                                               size=size,
                                                               feat=100,
                                                               num_of_feat=num_of_feats[d],
                                                               alpha=0.7,
                                                               model=model,
                                                               freq='low')

                feat_data_h, max_data_h, file_name = read_data(path_in=path_in_gauss,
                                                               path_max=path_max,
                                                               dataset=dataset,
                                                               dataset_max=datasets_max[d],
                                                               size=size,
                                                               feat=100,
                                                               num_of_feat=num_of_feats[d],
                                                               alpha=0.7,
                                                               model=model,
                                                               freq='high')

                if dataset in ['weather', 'electricity'] and model in ['llama_7b', 'deepseek_7b'] and size == '96':
                    num_of_feat = 6
                else:
                    num_of_feat = num_of_feats[d]

                # renormalize and  combined data
                df_data_l = denormalize(feat_data_l, max_data_l, num_of_feat)
                df_data_h = denormalize(feat_data_h, max_data_h, num_of_feat)

                # combine two dfs together
                final_df = combine_taces(df_data_l, df_data_h, num_of_feat)
                file_name = file_name[:-4] + 'comb'
                if '_9_' in file_name:
                    new_file_name = file_name.replace('_9_', '_6_')
                else:
                    new_file_name = file_name
                final_df.to_csv(path_out + '/' + new_file_name + '.csv',
                                index=False)
