import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import torch
from torch_geometric.data import Data, Batch
import pathlib

# loading only the edges with scores higher than least scores: list of [d1, d2, score]
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

def train_and_test(data, GNN, epochs=101, with_auc=True, with_aupr=True, with_f1=False, plot_training=False, print_logs=True, index=0):
    # Hyper parameter selection
    in_channels = data.num_features
    hid_channels = 128
    out_channels = data.y.shape[1]
    # Model creation
    model = GNN(in_channels=in_channels, hid_channels=hid_channels, out_channels=out_channels)
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Optimizer function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # selecting CPU and GPU depending on the environment
    def device_finder():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    # training the model
    losses = []
    auc_score_list = []
    f1score_list = []
    aupr_score_list = []
    f1 = 'N/A'
    aupr = 'N/A'
    auc_score = 'N/A'
    for epoch in range(epochs):
        output = model(data.x, data.edge_index, index=index)
        loss = criterion(output, data.y.float())
        loss_value = float("{:.3f}".format(loss.item()))
        losses.append(loss_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp_outputs = output.clone().detach().numpy()
        temp_labels = np.array(data.y)
        try:
            # auc score calculation
            if with_auc == True:
                auc_score = roc_auc_score(temp_labels, temp_outputs, average='weighted')
                auc_score = float("{:.3f}".format(auc_score))
                auc_score_list.append(auc_score)
            # f1 score calculation
            if with_f1 == True:
                f1 = f1_score(temp_labels, temp_outputs, average='weighted')
                f1 = float("{:.3f}".format(f1))
                f1score_list.append(f1)
            # aupr score calculation
            if with_aupr == True:
                aupr = average_precision_score(temp_labels, temp_outputs, average='weighted')
                aupr = float("{:.3f}".format(aupr))
                aupr_score_list.append(aupr)
        except Exception as e:
            print(e)
        if epoch % 10 == 0 and print_logs == True:
            print(f'Epoch: {epoch}, Loss: {loss_value}, AUC Score: {auc_score}, F1 Score: {f1}, AUPR Score: {aupr}')
    # plotting results
    if plot_training == True:
        train_result_plotting(losses, 'Loss')
        if with_auc == True:
            train_result_plotting(auc_score_list, 'AUC scores')
        if with_aupr == True:
            train_result_plotting(aupr_score_list, 'AUPR scores')
        if with_f1 == True:
            train_result_plotting(f1score_list, 'F1 scores')

    return losses, auc_score_list, aupr_score_list

def create_graph_of_input_drugs(df, se_df, node_features_df, least_score=0.8, consider_edge_attrs=True):
    # change least_score for changing minimum similarity score
    table = get_top_cells(df, least_score=least_score)
    # check to see if nodes in the dataset have node features
    node_features_drug_list = list(node_features_df['DCC_ID'])
    temp_edges = []
    for row in table:
        d1 = row[0]
        d2 = row[1]
        score = row[2]
        if d1 in node_features_drug_list and d2 in node_features_drug_list:
            temp_edges.append([d1, d2, score])
    table = np.array(temp_edges)
    # loading all edges and scores
    all_edges = []
    all_edges_scores = []
    nodes_list = list()
    nodes_dict = dict()
    reverse_node_data_dict = dict()
    count = 0
    # assigning numbers to nodes and their features
    for row in table:
        score = row[2]
        d1 = row[0]
        d2 = row[1]
        edge = [row[0], row[1]]
        reverse_edge = [row[1], row[0]]
        if d1 in node_features_drug_list and d2 in node_features_drug_list:
            if edge not in all_edges and reverse_edge not in all_edges:
                # adding edges with DCC code
                all_edges.append(edge)
                all_edges.append(reverse_edge)
                # adding edges scores
                all_edges_scores.append(float(score))
                all_edges_scores.append(float(score))
            if d1 not in nodes_list:
                nodes_list.append(d1)
                nodes_dict[d1] = count
                reverse_node_data_dict[count] = d1
                count += 1
            if d2 not in nodes_list:
                nodes_list.append(d2)
                nodes_dict[d2] = count
                reverse_node_data_dict[count] = d2
                count += 1
    # assigning new integer codes to use in pytorch geometric graph construction
    all_edges_coded = []
    for row in table:
        d1 = nodes_dict[row[0]]
        d2 = nodes_dict[row[1]]
        edge = [d1, d2]
        reverse_edge = [d2, d1]
        if edge not in all_edges_coded and reverse_edge not in all_edges_coded:
            all_edges_coded.append(edge)
            all_edges_coded.append(reverse_edge)

    # Loading drug nodes data
    nodes_data = []
    # print(len(nodes_dict))
    for each in list(nodes_dict.keys()):
        vecs = list(node_features_df[node_features_df['DCC_ID'] == each].values[0][1:])
        nodes_data.append(np.array(vecs))
    nodes_data = np.array(nodes_data, dtype=np.float32)
    # print("nodes_data.shape", nodes_data.shape)
    # Loading targets data (data.y)
    targets = []
    for row in se_df.iloc[:, :se_df.shape[1]].values:
        drug = np.squeeze(row[se_df.shape[1]-1:])
        if drug in list(nodes_dict.keys()):
            targets.append(list(row[1:se_df.shape[1]-1]))
    targets = np.array(targets)
    # Selecting only the features (columns) that both 1 and 0 exists
    cols = []
    for i in range(targets.shape[1]):
        col = targets[:, i]
        if 1 in col:
            cols.append(col)
    cols = np.array(cols)
    targets = cols.T
    # converting numpy arrays to tensors to feed GNN
    targets = torch.from_numpy(np.array(targets))
    edges_data = torch.from_numpy(np.array(all_edges_coded))
    edges_attr = torch.from_numpy(np.array(all_edges_scores))
    nodes_data = torch.from_numpy(nodes_data)

    # data = Data(x=nodes_data, edge_index=edges_data.T, edge_attr=edges_attr)
    if consider_edge_attrs == True:
        data = Data(y=targets, edge_index=edges_data.T, edge_attr=edges_attr, x=nodes_data)
    else:
        data = Data(y=targets, edge_index=edges_data.T, x=nodes_data)
    # # Automatically creating 5 node features
    # ldp = LocalDegreeProfile()
    # data = ldp(data)
    print(data)
    return data

def load_node_features_datasets():
    # finding the current path
    base_dir = pathlib.Path().resolve()
    node_target_df = pd.read_csv(f'{base_dir}/datasets/node_feature_datasets/Drugs_Targets_Onehot.csv', index_col='DCC_ID')
    node_indication_df = pd.read_csv(f'{base_dir}/datasets/node_feature_datasets/indicationsVec.csv', index_col='DCC_ID')

    node_w2v_df = pd.read_csv(f'{base_dir}/datasets/node_feature_datasets/word2vec.csv')

    word2vecs = np.random.randn(node_w2v_df.shape[0], 200)
    for i, row in enumerate(node_w2v_df.values):
        drug = row[0]
        vectors = row[1].replace('[', '').replace(']', '').replace('\n', ' ').split(' ')
        word_vectors = []
        for each in vectors:
            if each != '':
                word_vectors.append(float(each))
        for j, each in enumerate(word_vectors):
            word2vecs[i, j] = each
    del node_w2v_df['vector']
    for i in range(1, word2vecs.shape[1]+1):
        node_w2v_df[f'c{i}'] = list(word2vecs[:, i-1])

    node_w2v_df.set_index('DCC_ID', inplace=True)

    # preprocessing node2vec dataframe
    node_n2v_df = pd.read_csv(f'{base_dir}/datasets/node_feature_datasets/Node2Vec_DCC.csv')

    node2vecs = np.random.randn(node_n2v_df.shape[0], 128)
    for row in node_n2v_df.values:
        drug = row[0]
        vectors = row[1].replace('[', '').replace(']', '').replace("'", ' ').split(', ')
        node_vectors = []
        for each in vectors:
            if each != '':
                node_vectors.append(float(each))
        for j, each in enumerate(node_vectors):
            node2vecs[i, j] = each
    del node_n2v_df['Node2Vec']
    for i in range(1, node2vecs.shape[1]+1):
        node_n2v_df[f'c{i}'] = list(node2vecs[:, i-1])
    node_n2v_df.set_index('DCC_ID', inplace=True)
    df = pd.read_csv(f'{base_dir}/similarities/ATCSimilarity.csv', index_col='DCC_ID')
    se_df = pd.read_csv(f'{base_dir}/datasets/sideEffectsfillterd.csv', index_col='DCC_ID')
    del se_df['Unnamed: 0']

    node_fin_df = pd.read_csv(f'{base_dir}/datasets/node_feature_datasets/Drug_finger.csv', index_col='DCC_ID')
    del node_fin_df['Unnamed: 0']
    fin_list = []
    for row in node_fin_df.values:
        rows = str(row).replace('[', '').replace(']', '').split(', ')
        temp = []
        for each in rows:
            each = each.replace("'", '')
            temp.append(int(each))
        fin_list.append(temp)
    fin_list = np.array(fin_list)
    cols_title_list = [f'c{i+1}' for i in range(len(fin_list[0]))]

    for i in range(len(cols_title_list)):
        node_fin_df[cols_title_list[i]] = fin_list[:, i]
    del node_fin_df['fingerPrints']

    return node_n2v_df, node_w2v_df, node_target_df, node_indication_df, node_fin_df

def load_edges_df():
    base_dir = pathlib.Path().resolve()
    similarity_datasets = ['ATCSimilarity', 'fingerSimilarity', 'maxSimWithoutside', 'nod2vecSimilarity', 'TargetSimilarity', 'word2vecSimilarity']
    edge_df = []
    for i in range(len(similarity_datasets)):
        # loading the finger similarity into dataframe
        try:
            df = pd.read_csv(f'{base_dir}/similarities/{similarity_datasets[i]}.csv', index_col='DCC_ID')
        except Exception as e:
            df = pd.read_csv(f'{base_dir}/similarities/{similarity_datasets[i]}.csv', index_col='Unnamed: 0')
        edge_df.append(df)
    return edge_df

def create_graph(df, se_df, nodes_df, least_score=0.8):
    node_features_df = nodes_df
    se_drug_names_list = list(se_df.index)
    node_feature_names_list = list(node_features_df.index)
    df_edge_names_list = list(df.index)
    values = df.values

    drug_edges = []

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = float(values[i, j])
            d_row = df_edge_names_list[i]
            d_col = df_edge_names_list[j]
            if value >= least_score:
                if d_row in se_drug_names_list and d_row in node_feature_names_list and d_col in se_drug_names_list and d_col in node_feature_names_list:
                    drug_edges.append([d_row, d_col, value])
                else:
                    pass
            else:
                pass

    nodes_dict = dict()
    reverse_node_data_dict = dict()
    count = 0
    for row in drug_edges:
        d1 = row[0]
        d2 = row[1]
        value = row[2]
        if d1 not in list(nodes_dict.keys()):
            nodes_dict[d1] = count
            reverse_node_data_dict[count] = d1
            count+=1
        if d2 not in list(nodes_dict.keys()):
            nodes_dict[d2] = count
            reverse_node_data_dict[count] = d2
            count+=1
    edges_encoded = []
    edge_attrs = []
    for each in drug_edges:
        d1 = each[0]
        d2 = each[1]
        score = each[2]
        d1 = nodes_dict[d1]
        d2 = nodes_dict[d2]
        edge_attrs.append(score)
        edges_encoded.append([d1, d2])
    edges_encoded = np.array(edges_encoded)
    edge_attrs = np.array(edge_attrs, dtype=np.float32)
    node_features = []
    targets = []
    for key in list(nodes_dict.keys()):
        if key in node_feature_names_list:
            vecs = list(node_features_df.loc[key])
            node_features.append(vecs)
            side_effects = se_df.loc[key]
            targets.append(side_effects)
        else:
            print(key)
    node_features = np.array(node_features, dtype=np.float32)
    targets = np.array(targets, np.int32)
    values = []
    for i in range(targets.shape[1]):
        temp = targets[:, i]
        if temp.min() != temp.max():
            values.append(temp)
    values = np.array(values)
    targets = values.T
    targets.shape
    edge_index = edges_encoded.T

    targets = torch.from_numpy(targets)
    # targets = torch.Tensor(targets)
    edge_index = torch.from_numpy(edge_index)
    edges_attr = torch.from_numpy(edge_attrs)
    node_features = torch.from_numpy(node_features)

    data = Data(x=node_features, edge_index=edge_index, y=targets, edge_attr=edges_attr)
    return data