import numpy as np
import pandas as pd

def get_data(file_name):
    data = pd.read_csv('./data/{}.csv'.format(file_name), header=None, index_col=None)
    data_X = np.array(data[0])
    data_Y = np.array(data[1])

    return data_X, data_Y