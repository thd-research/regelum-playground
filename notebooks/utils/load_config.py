import os
import yaml
import pandas as pd


def load_exp_config(exp_dpath):
    os.listdir(exp_dpath)
    with open(os.path.join(exp_dpath, '.rehydra', 'config.yaml')) as f:
        exp_config = yaml.safe_load(f)
        
    return exp_config


def get_value_from_fields(configs, fields, index=0):
    if index == len(fields) - 1:
        return configs[fields[index]]
    else:
        return get_value_from_fields(configs[fields[index]], fields, index + 1)

def get_df_historical_data(exp_path, chosen_name):
    file_path = os.path.join(exp_path, ".callbacks/HistoricalDataCallback", f"{chosen_name}.h5")

    return pd.read_hdf(file_path, key="data") 
