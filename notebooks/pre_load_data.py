
import os

if 'main_dir' in globals():
    os.chdir(main_dir)

import pandas as pd
from utils.load_config import load_exp_config, get_value_from_fields, get_df_historical_data
from utils.load_data import (
    get_df_from_datetime_range,
    get_mlruns_info
    )
import numpy as np

# %matplotlib inline

if os.getcwd().endswith("notebooks"):
    main_dir = os.getcwd()
    os.chdir("..")

def main():
    DECAY_RATE = 1

    # Only used for pivot table
    start_datetime_str = "2025-02-17 16-36-58" # Laptop
    end_datetime_str = "2025-02-17 16-44-57"
    df, actor_df = get_df_from_datetime_range(start_datetime_str, 
                                    end_datetime_str, 
                                    decay_rate=DECAY_RATE,
                                    max_iter=np.inf,
                                    validity_check=False,
                                    reload=True,
                                    # backup_dir="/media/tcc/F89F-5CC1/ICLR/backup-data"
                                    )

if __name__ == "__main__":
    main()
