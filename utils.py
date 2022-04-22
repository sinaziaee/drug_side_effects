import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_top_cells(df, least_score=0.85):
    col_names = np.array(list(df.columns)[1:])
    row_names = np.array(list(df.iloc[:, 0:1].to_numpy().reshape((df.shape[0],))))
    temp = df.iloc[:, 1:]
    temp = temp[temp >= least_score]
    table = []
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            score = temp.iloc[i, j]
            if not str(score) == 'nan':
                table.append([row_names[i], row_names[j], score])
    table = np.array(table)
    return table

def train_result_plotting(values, name):
    plt.plot(values)
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.show()

def plot_different_results(loss_list, auc_list, aupr_list, min_score):
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ####################
    plt.subplot(132)
    plt.plot(auc_list)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    ####################
    plt.subplot(133)
    plt.plot(aupr_list)
    plt.xlabel('Epoch')
    plt.ylabel('AUPR')
    ####################
    plt.suptitle(f'Training on drugs with at least {min_score} similarity')
    plt.show()