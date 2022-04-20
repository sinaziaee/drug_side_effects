import numpy as np

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


