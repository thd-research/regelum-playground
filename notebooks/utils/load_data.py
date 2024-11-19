import os
from datetime import datetime
import pathlib
import numpy as np
import sys, traceback
import pandas as pd
from multiprocessing import Pool
from rich.progress import Progress

from utils.load_config import (
    get_df_historical_data,
    get_list_historical_data,
    load_exp_config
    )

from glob import glob
from yaml import safe_load

PROJECT_DIR = "."
PROJECT_DIR = "/home/robosrv/huyhoang/iclr-2025/regelum-playground-iclr"
ROOT_DIR = PROJECT_DIR + "/regelum_data/outputs/"

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


def load_mlrun_df(exp_path, mlruns_folder_info):
    run_name = "{} {} 0".format(*pathlib.PurePath(exp_path).parts[-2:])

    if run_name not in mlruns_folder_info:
        return pd.DataFrame()

    actor_loss_path = mlruns_folder_info[run_name] + "/metrics/losses/actor_loss"

    # print("actor_loss_path:", actor_loss_path)
    if not os.path.exists(actor_loss_path):
        return pd.DataFrame()
    
    step_info = pd.read_table(actor_loss_path, delimiter=" ", names=["time", "actor_loss", "step_id"])
    step_info["run_name"] = run_name
    step_info["experiment_path"] = exp_path

    return step_info


def load_iteration(iteration_path, exp_path, validity_check, objective_function, decay_rate):
    tmp_df = get_df_historical_data(absolute_path=iteration_path)

    if tmp_df.empty:
        return pd.DataFrame(), pd.DataFrame()
            
    tmp_df = correct_column_name(tmp_df)

    if validity_check and not is_df_valid(tmp_df):
        return pd.DataFrame()

    tmp_df["absolute_path"] = iteration_path
    config = load_exp_config(exp_path)
    tmp_df.loc[:, "exp_config"] = [config] * len(tmp_df)
    
    if objective_function is not None:
        tmp_df["objective_value"] = tmp_df.apply(lambda x: cal_obj_df(x, objective_function), axis=1)
        # tmp_df["accumulative_objective"] = tmp_df["objective_value"].apply(lambda x: x*0.1).cumsum()
        tmp_df["accumulative_objective"] = tmp_df.apply(lambda x: x["objective_value"]*0.1*decay_rate**x["time"], axis=1).cumsum()

    return tmp_df


def name_backup_file(prefix, start_datetime_str, end_datetime_str, backup_dir):
    backup_file_name = "_".join([c.replace(" ", "_") for c in [prefix, start_datetime_str, end_datetime_str]]) + ".pkl"
    bk_path = os.path.join(backup_dir, backup_file_name)
    return bk_path


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
    
    log_bk_path = name_backup_file("data", start_datetime_str, end_datetime_str, backup_dir)
    mlrun_bk_path = name_backup_file("mlruns_actorloss_", start_datetime_str, end_datetime_str, backup_dir)

    if not reload and os.path.exists(log_bk_path) and os.path.exists(mlrun_bk_path):
        return pd.read_pickle(log_bk_path), pd.read_pickle(mlrun_bk_path)

    date_folder = os.listdir(ROOT_DIR)
    mlruns_folder_info = get_mlruns_folder_info()

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

    print("Load path:", len(path_hierachy))
    total_dfs = []
    total_mlrun_dfs = []

    with Progress() as progress:
        task1 = progress.add_task("[red]Total loading...", total=len(path_hierachy)) # Just for visualization

        for exp_path in path_hierachy:
            progress.update(task1, advance=1) # Just for visualization
            mlrun_df = load_mlrun_df(exp_path, mlruns_folder_info)

            if mlrun_df.empty:
                continue

            total_mlrun_dfs.append(mlrun_df)

            task2 = progress.add_task("[green]Iteration loading...", total=len(path_hierachy)) # Just for visualization
            exp_dfs = []

            with Pool() as p:
                args = [(iteration_path, 
                         exp_path, 
                         validity_check, 
                         objective_function, 
                         decay_rate) 
                        for iteration_path in path_hierachy[exp_path]]
                for tmp_df in p.starmap(load_iteration, args):
                    if tmp_df.empty:
                        continue

                    exp_dfs.append(tmp_df)
                    progress.update(task2, advance=1) # Just for visualization
                    
            if len(exp_dfs) == 0:
                continue
            
            exp_df = pd.concat(exp_dfs)
            exp_df.sort_values(by=["iteration_id", "time"], inplace=True)
            exp_df["experiment_path"] = exp_path
            
            total_dfs.append(exp_df)

            progress.remove_task(task2)

    total_df = pd.concat(total_dfs)
    total_mlrun_df = pd.concat(total_mlrun_dfs)


    # Post process
    total_df = total_df[total_df.iteration_id <= max_iter]

    os.makedirs(backup_dir, exist_ok=True)
    total_df.to_pickle(log_bk_path)
    total_mlrun_df.to_pickle(mlrun_bk_path)
    
    return total_df, total_mlrun_df


def get_mlruns_folder_info():
    MLRUN_DIR = PROJECT_DIR + "/regelum_data/mlruns"
    mlruns_yaml_files = glob(f"{MLRUN_DIR}/**/*.yaml", recursive=True)
    mlruns_folder_info = {}

    for fp in mlruns_yaml_files:
        with open(fp, "r") as f:
            data = safe_load(f)

        if not isinstance(data, dict):
            continue

        if "run_id" in data.keys():
            mlruns_folder_info[data["run_name"]] = os.path.join(MLRUN_DIR, data["experiment_id"], data["run_id"])

    return mlruns_folder_info


def get_mlruns_info(start_datetime_str, 
                    end_datetime_str,
                    date_format='%Y-%m-%d %H-%M-%S',
                    backup_dir="./backup-data",
                    reload=False):
    
    backup_file_name = "_".join([c.replace(" ", "_") for c in ["mlruns_actorloss_", start_datetime_str, end_datetime_str]]) + ".pkl"
    bk_path = os.path.join(backup_dir, backup_file_name)

    if not reload and os.path.exists(bk_path):
        return pd.read_pickle(bk_path)

    mlruns_folder_info = get_mlruns_folder_info()

    start_date_time = datetime.strptime(start_datetime_str, date_format)
    end_date_time = datetime.strptime(end_datetime_str, date_format)

    date_folder = os.listdir(ROOT_DIR)

    valid_paths = []
    for d in date_folder:
        for t in os.listdir(os.path.join(ROOT_DIR, d)):
            tmp_datetime = datetime.strptime(f"{d} {t}", date_format)
            if tmp_datetime < start_date_time or end_date_time < tmp_datetime:
                continue

            valid_paths.append(str(pathlib.Path(os.path.join(ROOT_DIR, d, t)).absolute()))

    if len(valid_paths) == 0:
        return pd.DataFrame()
    
    final_df = None
    for p in valid_paths:
        run_name = "{} {} 0".format(*pathlib.PurePath(p).parts[-2:])

        if run_name not in mlruns_folder_info:
            continue

        actor_loss_path = mlruns_folder_info[run_name] + "/metrics/losses/actor_loss"

        # print("actor_loss_path:", actor_loss_path)
        if not os.path.exists(actor_loss_path):
            raise FileNotFoundError
        
        step_info = pd.read_table(actor_loss_path, delimiter=" ", names=["time", "actor_loss", "step_id"])
        step_info["run_name"] = run_name
        step_info["experiment_path"] = p

        if final_df is None:
            final_df = step_info
        else:
            final_df = pd.concat([final_df, step_info])

    if final_df is None:
        return pd.DataFrame()
    
    os.makedirs(backup_dir, exist_ok=True)
    final_df.to_pickle(bk_path)

    return final_df
