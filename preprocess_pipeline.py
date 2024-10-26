import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def run(main_label, n_timesteps):
    label_name = ['eyebrows', 'left', 'right', 'both', 'teeth']
    trial_num = 10

    print(f">>>>>>>>> Filtering 5 label with {main_label} filter <<<<<<<<<<<<")
    os.makedirs(rf'./pipeline_{main_label}', exist_ok=True)
    for label in label_name:
        for position in range(3):
            for trial in range(0, trial_num):
                raw_df = pd.read_csv(rf'./data/raw_data_luc/{label}/{position}_{trial}.csv').drop(columns=['timestamps', 'Right AUX'])

                data = raw_df.to_numpy()
                for column in range(input.shape[1]):
                    x=np.array(data[:, column]) 
                    x = filter[main_label](x)
                    data[:, column] = x

                os.makedirs(rf'./pipeline_{main_label}/filtered/{label}', exist_ok=True)
                pd.DataFrame(data, columns=raw_df.columns).to_csv(rf'./pipeline_{main_label}/filtered/{label}/{position}_{trial}.csv')
    print("--------------- Done ---------------\n")


    print(f">>>>>>>>> Normalize 5 label with {main_label} filter <<<<<<<<<<<<")
    os.makedirs(rf'./pipeline_{main_label}/checkpoints', exist_ok=True)
    dfs = []
    for label in label_name:
        _path = rf'./pipeline_{main_label}/filtered'
        for position in range(3):
            for trial in range(0, trial_num):
                df = pd.read_csv(_path + rf'/{label}/{position}_{trial}.csv').drop(columns=['Unnamed: 0'])
                dfs.append(df)
    dfs = pd.concat(dfs)

    scaler = MinMaxScaler()
    scaler.fit(dfs)

    scaler_filename = rf"./pipeline_{main_label}/checkpoints/scaler.save"
    joblib.dump(scaler, scaler_filename) 
    print("--------------- Done ---------------\n")