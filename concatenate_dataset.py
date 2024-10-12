import os
import shutil

new_raw_data_folder = './data/raw_data'
new_roi_folder = './data/roi'

subjects = ['dang', 'luc']
labels = ['eyebrows', 'left', 'right', 'both', 'teeth']

for i, subject in enumerate(subjects):
    for label in labels:
        os.makedirs(f"{new_raw_data_folder}/{label}", exist_ok=True)
        os.makedirs(f"{new_roi_folder}/{label}", exist_ok=True)

        raw_data_folder = f'./data/raw_data_{subject}/{label}'
        roi_folder = f'./data/roi_{subject}/{label}'

        raw_data_files = os.listdir(raw_data_folder)
        roi_files = os.listdir(roi_folder)

        for raw_data_file in raw_data_files:
            shutil.copyfile(
                f"{raw_data_folder}/{raw_data_file}",
                f"{new_raw_data_folder}/{label}/{i}_{raw_data_file}",
            )

        for roi_file in roi_files:
            shutil.copyfile(
                f"{roi_folder}/{roi_file}",
                f"{new_roi_folder}/{label}/{i}_{roi_file}",
            )