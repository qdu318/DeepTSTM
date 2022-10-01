import numpy as np
import pandas as pd
from utils.utils import Z_Score
from utils.utils import generate_dataset

import pywt


def Data_load(dataset,config, timesteps_input, timesteps_output):

    X = pd.read_csv("./dataset/50nodes/V_flow_50_2.csv", header=None).to_numpy(np.float32)
    W_nodes = np.load("./dataset/50nodes/noSparsity_50.npy").astype(np.float32)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    cAX1,(cHX,cDX,cVX)=pywt.dwt2(X,'haar')


    cAX, X_mean, X_std = Z_Score(X)

    cHX,XH_mean,XH_std= Z_Score(cDX)

    index_1 = int(cAX.shape[2] * 0.8)
    index_2 = int(cAX.shape[2])

    indexH_1=int(cHX.shape[2]*0.8)
    indexH_2 = int(cHX.shape[2])

    train_original_data = cAX

    train_original_data_CH=cHX

    val_original_data = cAX[:, :, index_1:index_2]

    val_original_data_CH = cHX[:, :, indexH_1:indexH_2]

    train_input, train_target = generate_dataset(train_original_data,
                                                 num_timesteps_input=timesteps_input,
                                                 num_timesteps_output=timesteps_output,
                                                 step=288)
    train_input_CH, train_target_CH = generate_dataset(train_original_data_CH,
                                                 num_timesteps_input=int(timesteps_input/2),
                                                 num_timesteps_output=int(timesteps_output/2),
                                                       step=144)


    evaluate_input, evaluate_target = generate_dataset(val_original_data,
                                                       num_timesteps_input=timesteps_input,
                                                       num_timesteps_output=timesteps_output,
                                                       step=288)
    evaluate_input_CH, evaluate_target_CH = generate_dataset(val_original_data_CH,
                                                       num_timesteps_input=int(timesteps_input/2),
                                                       num_timesteps_output=int(timesteps_output/2),
                                                             step=144)

    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set[
        'eval_target'],  data_set['X_mean'], data_set['X_std'],data_set['train_input_CH'],data_set[
        'train_target_CH'], data_set['evaluate_input_CH'],data_set['evaluate_target_CH'],data_set['XH_mean'],data_set['XH_std']\
        = train_input, train_target, evaluate_input, evaluate_target, X_mean, X_std,train_input_CH,train_target_CH,evaluate_input_CH,evaluate_target_CH,XH_mean,XH_std
    return W_nodes, data_set

