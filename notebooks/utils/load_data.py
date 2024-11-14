import os
from datetime import datetime
import pathlib
import numpy as np
import sys, traceback
import pandas as pd

from utils.load_config import (
    get_df_historical_data,
    get_list_historical_data,
    load_exp_config
    )

from glob import glob
from yaml import safe_load


ROOT_DIR = "./regelum_data/outputs/"

def correct_column_name(df):
    replacements = {
        "x_rob"               : "x [m]", 
        "y_rob"               : "y [m]",
        "vartheta"            : "angle [rad]",
        "v"                   : "velocity [m/s]",
        "omega"               : "angular velocity [rad/s]",
        "steering angle [rad]": "steering angle [rad]"
    }

    # print("old columns:", df.columns.values)
    new_columns = []
    if "x [m]" not in df.columns.values:
        for col in df.columns.values:
            new_columns.append(col if col not in replacements else replacements[col])

        df.columns = new_columns
    
    return df


def cal_obj_df(row, objective_function):
    observation = np.expand_dims(np.array(row.loc[["x [m]", "y [m]", "angle [rad]"]].values, dtype=np.float32), axis=0)

    if "angular velocity [rad/s]" in row:
        action = np.expand_dims(np.array(row.loc[["velocity [m/s]", "angular velocity [rad/s]"]].values, dtype=np.float32), axis=0)
    else:
        action = np.expand_dims(np.array(row.loc[["velocity [m/s]", "steering angle [rad]"]].values, dtype=np.float32), axis=0)

    try:
        return objective_function(observation, action)
    except Exception as err:
        print("Error:", err)
        traceback.print_exc(file=sys.stdout)
        
        raise err


def is_df_valid(df):
    old_position = None
    old_timestamp = 0
    for idx, data in df.iterrows():
        position = np.array([data["x [m]"], data["y [m]"]])
        if old_position is not None:
            position_change = np.linalg.norm(position - old_position)
            delta_t = data["time"] - old_timestamp
            # print(position_change)
            if (position_change > delta_t*df["velocity [m/s]"].abs().max()):
                print("At", data["time"], position_change)
                return False

        old_timestamp = data["time"]
        old_position = position

    return True


def get_df_from_datetime_range(start_datetime_str, 
                               end_datetime_str, 
                               objective_function=None,
                               date_format='%Y-%m-%d %H-%M-%S', 
                               decay_rate=1,
                               max_iter=100,
                               reload=False,
                               validity_check=True,
                               backup_dir="./backup-data"
                               ):
    start_date_time = datetime.strptime(start_datetime_str, date_format)
    end_date_time = datetime.strptime(end_datetime_str, date_format)
    

    backup_file_name = "_".join([c.replace(" ", "_") for c in ["data", start_datetime_str, end_datetime_str]]) + ".pkl"
    bk_path = os.path.join(backup_dir, backup_file_name)

    if not reload and os.path.exists(bk_path):
        return pd.read_pickle(bk_path)

    date_folder = os.listdir(ROOT_DIR)

    valid_paths = []
    for d in date_folder:
        for t in os.listdir(os.path.join(ROOT_DIR, d)):
            tmp_datetime = datetime.strptime(f"{d} {t}", date_format)
            if tmp_datetime < start_date_time or end_date_time < tmp_datetime:
                continue

            valid_paths.append(str(pathlib.Path(os.path.join(ROOT_DIR, d, t)).absolute()))

    path_hierachy = {}
    for p in valid_paths:
        path_hierachy[p] = get_list_historical_data(p)

    print("Load path:", "\n".join(path_hierachy))
    total_dfs = []
    for exp_path in path_hierachy:
        exp_dfs = []
        for iteration_path in path_hierachy[exp_path]:
            tmp_df = get_df_historical_data(absolute_path=iteration_path)
            
            tmp_df = correct_column_name(tmp_df)

            if validity_check and not is_df_valid(tmp_df):
                continue

            tmp_df["absolute_path"] = iteration_path
            config = load_exp_config(exp_path)
            tmp_df.loc[:, "exp_config"] = [config] * len(tmp_df)
            
            if objective_function is not None:
                tmp_df["objective_value"] = tmp_df.apply(lambda x: cal_obj_df(x, objective_function), axis=1)
                # tmp_df["accumulative_objective"] = tmp_df["objective_value"].apply(lambda x: x*0.1).cumsum()
                tmp_df["accumulative_objective"] = tmp_df.apply(lambda x: x["objective_value"]*0.1*decay_rate**x["time"], axis=1).cumsum()

            exp_dfs.append(tmp_df)
        if len(exp_dfs) == 0:
            continue
        
        exp_df = pd.concat(exp_dfs)
        exp_df.sort_values(by=["iteration_id", "time"], inplace=True)
        exp_df["experiment_path"] = exp_path
        
        total_dfs.append(exp_df)

    total_df = pd.concat(total_dfs)

    # Post process
    total_df = total_df[total_df.iteration_id <= max_iter]

    os.makedirs(backup_dir, exist_ok=True)
    total_df.to_pickle(bk_path)
    
    return total_df

def get_mlruns_info(start_datetime_str, 
                    end_datetime_str,
                    date_format='%Y-%m-%d %H-%M-%S',
                    backup_dir="./backup-data",
                    reload=False):
    
    backup_file_name = "_".join([c.replace(" ", "_") for c in ["mlruns_actorloss_", start_datetime_str, end_datetime_str]]) + ".pkl"
    bk_path = os.path.join(backup_dir, backup_file_name)

    if not reload and os.path.exists(bk_path):
        return pd.read_pickle(bk_path)

    MLRUN_DIR = "./regelum_data/mlruns"
    mlruns_yaml_files = glob(f"{MLRUN_DIR}/**/*.yaml", recursive=True)
    mlruns_folder_info = {}

    for fp in mlruns_yaml_files:
        with open(fp, "r") as f:
            data = safe_load(f)

        if not isinstance(data, dict):
            continue

        if "run_id" in data.keys():
            mlruns_folder_info[data["run_name"]] = os.path.join(MLRUN_DIR, data["experiment_id"], data["run_id"])

    start_date_time = datetime.strptime(start_datetime_str, date_format)
    end_date_time = datetime.strptime(end_datetime_str, date_format)

    date_folder = os.listdir(ROOT_DIR)

    valid_path = None
    for d in date_folder:
        for t in os.listdir(os.path.join(ROOT_DIR, d)):
            tmp_datetime = datetime.strptime(f"{d} {t}", date_format)
            if tmp_datetime < start_date_time or end_date_time < tmp_datetime:
                continue

            valid_path = str(pathlib.Path(os.path.join(ROOT_DIR, d, t)).absolute())
            break
        if valid_path is not None:
            break

    if valid_path is None:
        return pd.DataFrame()
    
    run_name = "{} {} 0".format(*pathlib.PurePath(valid_path).parts[-2:])
    actor_loss_path = mlruns_folder_info[run_name] + "/metrics/losses/actor_loss"
    if not os.path.exists(actor_loss_path):
        raise FileNotFoundError
    
    step_info = pd.read_table(actor_loss_path, delimiter=" ", names=["time", "actor_loss", "step_id"])
    step_info["run_name"] = run_name

    os.makedirs(backup_dir, exist_ok=True)
    step_info.to_pickle(bk_path)

    return step_info
