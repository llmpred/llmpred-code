# llmpred-code
This is the repository for LLMPred paper and **currently being updated**.

# Abastract of LLMPred
Time-series prediction or forecasting is critical across many real-world dynamic systems, and recent studies have proposed using Large Language Models (LLMs) for this task due to their strong generalization capabilities and ability to perform well without extensive pre-training.
However, their effectiveness in handling complex, noisy, and multivariate time-series data remains underexplored. To address this, we propose LLMPred which enhances LLM-based time-series prediction by converting time-series sequences into text and feeding them to LLMs for zero-shot prediction along with two main data pre-processing techniques. First, we apply time-series sequence decomposition to facilitate accurate prediction on complex and noisy univariate sequences. Second, we extend this univariate prediction capability to multivariate data using a lightweight prompt-processing strategy. Extensive experiments with smaller LLMs such as Llama 2 7B, Llama 3.2 3B, GPT-4o-mini, and DeepSeek 7B demonstrate that LLMPred achieves competitive or superior performance compared to state-of-the-art baselines. Additionally, a thorough ablation study highlights the importance of the key components proposed in LLMPred. 

# Overview of LLMpred process
<img src="overview_revised.jpg" alt="My Image" width="1000"/>

# Dataset
Intermediate datasets including input to the LLM and other models and their outputs can be found at [Dataset](https://drive.google.com/drive/folders/1z3gY4nfkHPGF0JJcoKwRQCS3fYjstGkj?usp=sharing). To run the code download the dataset and keep it in data folder in the parent folder.

# Requirements
`pip install -r requirements.txt`

# Run the code

## Univariate analysis

Use the following sequence of commands to run the scripts for an individual feature experiment: 

We have set:  
- `train` and `test` (prediction length as well) to 48  
- `dataset` to ETTm2  
- `freq` (frequency) component generated to `high`  
- `num_feat` as one  
- `spec_feat` to 3 to generate the third feature of the dataset  
- `max_tokens` to 200 unless it is overridden by the input token length inside the code  
- `model_name` to llama_3b indicating Llama 3.2 - 3B model
- `limit` of number of training prompts to 50  
- `num_samples` to 6

### Python scripts to run ###
- Step 1: This step correponds converting the numerical sequences to the text data. This converted data is already provided in the dataset folder as **data_1_input_data**

- Step 2: Run the LLM model for initial generation. Input data is from **data_1_input_data** and output will be stored in **data_2_ouput_data**
```
python step_2_run_gen.py --exp run_individual --train_len 48 --test_len 48 --dataset ETTm2 --freq high --alpha 0.7 --num_feat 1 --spec_feat 3 --max_tokens 200 --model_name llama_3b --limit 50 --num_samples 6
```

- Step 3: Run the post-processing steps to validate the output from the LLM model. Input data is from **data_2_ouput_data** and the output will be stored in **data_3_llm_post_processed**.
```
python step_3_post_process_llm_output.py --train_len 48 --test_len 48 --dataset ETTm2 --freq high --num_feat 1 --spec_feat 3 --max_tokens 200 --model_name llama_3b --limit 50 --num_samples 6
```

- Step 4: Pre-process data to be run by the nn model. Input data is from **data_3_llm_post_processed** and the output will be stored in **data_4_pre_process_nn**. 
```
python step_4_pre_process_for_nn_2.py --train_len 48 --test_len 48 --dataset ETTm2 --freq low --num_feat 1 --spec_feat 3 --max_tokens 200 --model_name llama_3b --limit 50 --num_samples 6
```

- Step 5: Run post-processing MLP on the `low` freq components. Input data is from **data_4_pre_process_nn** and output is stored in **data_5_nn_out**. This step is run only for `low` frequencey components
```
python step_5_run_nn.py --train_len 48 --test_len 48 --dataset ETTm2 --freq low --num_feat 1 --spec_feat 3 --max_tokens 200 --model_name llama_3b --limit 50 --num_samples 6
```

- Step 6: Run post-processing MLP on the `high` freq components. Input data is from **data_4_pre_process_nn** and output is stored in **data_6_gaussian_tranform**. This step is run only for `high` frequencey components
```
python step_6_gaussian_transform.py --train_len 48 --test_len 48 --dataset ETTm2 --freq high --num_feat 1 --spec_feat 3 --max_tokens 200 --model_name llama_3b --limit 50 --num_samples 6
```

## Multivariate analysis
Considering the similar parameters in the univariate analysis, users can run the following code segements

```
python step_2_run_gen.py --exp run_multivariate --train_len 48 --test_len 48 --dataset ETTm2 --freq high --alpha 0.7 --num_feat 6 --spec_feat 100 --max_tokens 200 --model_name llama_3b --limit 50 --num_samples 6
```





