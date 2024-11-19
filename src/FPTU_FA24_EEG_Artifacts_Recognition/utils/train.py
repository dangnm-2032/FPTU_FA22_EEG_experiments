from FPTU_FA24_EEG_Artifacts_Recognition.constants import *
from datasets import Dataset
import pandas as pd
import time
import tensorflow as tf

def run_data_process(
        data, 
        label,
        timestep, 
        dataset_config, 
        filters, 
        scalers,
    ):
    temp_input = {
        'eyebrows': [],
        'left': [],
        'right': [],
        'both': [],
        'teeth': [],
        'label': []
    }
    for i, (input_filepath, output_filepath) in enumerate(data[label]):
        input_df = pd.read_csv(input_filepath).drop(columns=['timestamps', 'Right AUX'])
        output_df = pd.read_csv(output_filepath, index_col=0)

        input_data = input_df.to_numpy()
        output_data = output_df.to_numpy()

        assert input_data.shape[1] == 4
        assert output_data.shape[0] == input_data.shape[0]

        for i in range(0, input_data.shape[0] - timestep):
            if 1 in output_data[i:i+timestep]:
                window_input = input_data[i:i+timestep]
                window_output = output_data[i:i+timestep]

                for _label in dataset_config.label:
                    filter = filters[_label]
                    scaler = scalers[_label]
                    _input = window_input.copy()
                    for col in range(window_input.shape[1]):
                        _input[:, col] = filter(window_input[:, col])
                    _input = scaler.transform(_input) # sample, channel (64, 4)

                    temp_input[_label].append(_input)
                
                temp_input['label'].append(window_output * LABEL2IDX[label])

    temp_input = Dataset.from_dict(temp_input)
    temp_input.save_to_disk(Path(os.path.join(
        dataset_config.output_data_path, label
    )))

def get_backbone_and_feature_map(
    model_params,
    backbone,
    label
):
    model = backbone(
        nb_classes=model_params.nb_classes,
        Chans=model_params.Chans,
        Samples=model_params.Samples,
        dropoutRate=model_params.dropoutRate,
        kernLength=model_params.kernLength,
        F1=model_params.F1,
        D=model_params.D,
        F2=model_params.F2,
        dropoutType=model_params.dropoutType,
    )

    model.layers[-3].name = label + '_flatten'
    feature_map = model.layers[-3].output

    return model, feature_map

def transform_data(ds):
    ds['eyebrows'] = tf.expand_dims(tf.transpose(ds['eyebrows'], (1, 0)), -1)
    ds['left'] = tf.expand_dims(tf.transpose(ds['left'], (1, 0)), -1)
    ds['right'] = tf.expand_dims(tf.transpose(ds['right'], (1, 0)), -1)
    ds['both'] = tf.expand_dims(tf.transpose(ds['both'], (1, 0)), -1)
    ds['teeth'] = tf.expand_dims(tf.transpose(ds['teeth'], (1, 0)), -1)
    
    return ds