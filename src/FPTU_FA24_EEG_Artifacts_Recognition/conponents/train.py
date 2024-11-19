from FPTU_FA24_EEG_Artifacts_Recognition.config import *
from FPTU_FA24_EEG_Artifacts_Recognition.constants import *
import joblib
import pandas as pd
from copy import deepcopy
import warnings
import time
import multiprocessing
import datasets
from .models import *
import shutil
import json

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Activation, Reshape, Concatenate


warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def initialize_preprocess_module(self):
        self.filters = get_all_filters()

        self.scalers = {}
        config = self.config.get_dataset_config()
        for label in config.label:
            self.scalers[label] = joblib.load(Path(
                os.path.join(
                    config.scaler_path,
                    label + config.scaler_extension
                )
            ))


    def load_data(self):
        dataset_config = self.config.get_dataset_config()
        self.data = {}
        for label in dataset_config.label:
            files = []
            for idx, _, position, trial in itter_dataset_file_by_label(dataset_config, label):
                filepath = dataset_config.filepath_format.format_map(
                    {
                        'subject_id': idx,
                        'label': label,
                        'position': position,
                        'trial': trial
                    }
                )

                raw_data_filepath = Path(
                    os.path.join(dataset_config.raw_data_path, filepath)
                )
                roi_filepath = Path(
                    dataset_config.output_roi_path,
                    filepath
                )
                files.append([raw_data_filepath, roi_filepath])
            print(len(files))
            self.data[label] = deepcopy(files)

    def transform_data(self):
        dataset_config = self.config.get_dataset_config()
        model_config = self.config.get_eeg_model_params()

        timestep = model_config.both.Samples

        jobs = []

        if dataset_config.skip_preprocess_data:
            return

        for label in dataset_config.label:
            p = multiprocessing.Process(
                target=run_data_process, 
                args=(
                    self.data, 
                    label,
                    timestep, 
                    dataset_config, 
                    self.filters, 
                    self.scalers
                )
            )
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()


    def train(self):
        # Prepare data
        dataset_config = self.config.get_dataset_config()
        ds_min = 9999999
        big_df = {}
        for label in dataset_config.label:
            ds = datasets.Dataset.load_from_disk(Path(os.path.join(
                dataset_config.output_data_path, label
            )))
            ds.set_format('tf')
            ds = ds.map(transform_data, num_proc=5)
            
            big_df[label] = ds
            if ds_min > len(ds):
                ds_min = len(ds)

        train_ds = []
        test_ds = []

        for label in dataset_config.label:
            big_df[label] = big_df[label].shuffle(seed=42)
            big_df[label] = big_df[label].select(range(ds_min))

            ds = big_df[label].train_test_split(test_size=0.2, shuffle=False)
            train_ds.append(ds['train'])
            test_ds.append(ds['test'])

        train_ds = datasets.concatenate_datasets(train_ds)
        test_ds = datasets.concatenate_datasets(test_ds)

        test_ds.save_to_disk(dataset_config.save_test_data)

        print(train_ds)
        print(test_ds)

        # Modelling
        model_params = self.config.get_eeg_model_params()
        
        backbone_left, backbone_left_featmap = get_backbone_and_feature_map(model_params.left, EEGNet, 'left')
        backbone_right, backbone_right_featmap = get_backbone_and_feature_map(model_params.right, EEGNet, 'right')
        backbone_both, backbone_both_featmap = get_backbone_and_feature_map(model_params.both, EEGNet, 'both')
        backbone_teeth, backbone_teeth_featmap = get_backbone_and_feature_map(model_params.teeth, EEGNet, 'teeth')
        backbone_eyebrows, backbone_eyebrows_featmap = get_backbone_and_feature_map(model_params.eyebrows, EEGNet, 'eyebrows')

        x = Concatenate()([
            backbone_left_featmap,
            backbone_right_featmap,
            backbone_both_featmap,
            backbone_teeth_featmap,
            backbone_eyebrows_featmap
        ])
        x = Dense(1024, activation='gelu')(x)
        x = Dense(512, activation='gelu')(x)
        x = Dense(model_params.eyebrows.Samples*6, activation='linear')(x)
        x = Reshape((model_params.eyebrows.Samples, 6))(x)
        x = Activation('softmax', name = 'softmax')(x)

        model = Model(
            inputs=[
                backbone_eyebrows.input,
                backbone_left.input,
                backbone_right.input,
                backbone_both.input,
                backbone_teeth.input,
            ], 
            outputs=x
        )

        model.summary()

        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=model_params.training.learning_rate
            ),
            metrics=[
                'sparse_categorical_accuracy', 
            ]
        )

        history = model.fit(
            [
                train_ds['eyebrows'],
                train_ds['left'],
                train_ds['right'],
                train_ds['both'],
                train_ds['teeth'],
            ],
            train_ds['label'],
            validation_data=(
                [
                    test_ds['eyebrows'],
                    test_ds['left'],
                    test_ds['right'],
                    test_ds['both'],
                    test_ds['teeth'],
                ],
                test_ds['label']
            ),
            batch_size=model_params.training.batch_size,
            epochs=model_params.training.epochs
        )


        model_config = self.config.get_eeg_model_config()

        # Save training history
        json.dump(
            history.history, 
            open(
                Path(os.path.join(
                    model_config.save_path,
                    model_config.save_name + model_config.history_extension
                )), 
                'w'
            )
        )

        # Save model checkpoint
        os.makedirs(model_config.save_path, exist_ok=True)
        model.save(Path(os.path.join(
            model_config.save_path,
            model_config.save_name + model_config.weight_extension
        )))

        # Save model params
        shutil.copyfile(
            PARAMS_FILE_PATH, 
            Path(os.path.join(
                model_config.save_path,
                model_config.save_name + model_config.config_extension
            ))
        )