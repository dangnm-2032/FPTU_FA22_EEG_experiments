import numpy as np
# Importing Pandas Library 
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
from utils import *

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Activation, Reshape, Concatenate

from models.EEGNet import *

from sklearn.metrics import confusion_matrix

from preprocessing import *

import warnings
warnings.filterwarnings("ignore")

label_name = [
    'eyebrows', 
    'left',
    'right',
    'both',
    'teeth'
]
models = []
scalers = {}

for label in label_name:
    _model = load_model(rf'./pipeline_{label}/checkpoints/checkpoint.keras')
    _model.trainable = False
    _model = Model(inputs=_model.input, outputs=_model.layers[-4].output, name=label)
    print(_model.summary())
    models.append(_model)

    scalers[label] = joblib.load(rf'./pipeline_{label}/checkpoints/scaler.save')


inputs = []
input_shape = {
    'teeth': (4, 128, 1),
    'left': (4, 128, 1),
    'right': (4, 128, 1),
    'eyebrows': (4, 128, 1),
    'both': (4, 128, 1),
}
for label in label_name:
    inputs.append(
        Input(shape=input_shape[label], name=f"{label} input")
    )


outs = []
for i in range(len(label_name)):
    model = models[i]
    input = inputs[i]

    outs.append(model(input))

x = Concatenate()(outs)
x = Dense(1024, activation='gelu')(x)
x = Dense(512, activation='gelu')(x)
x = Dense(128*6, activation='linear')(x)
x = Reshape((128, 6))(x)
x = Activation('softmax', name = 'softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)


raw_data = {}
trial_num = 10
for label in label_name:
    raw_data[label] = {}
    for position in range(3):
        for trial in range(trial_num):
            raw_data[label][len(raw_data[label])] = [
                rf'./data/raw_data_luc/{label}/{position}_{trial}.csv',
                rf'./data/roi_luc/{label}/{position}_{trial}.csv'
            ]
    

filters = {
    'left': filter_left,
    'right': filter_right,
    'both': filter_both,
    'teeth': filter_teeth,
    'eyebrows': filter_eyebrows,
}


def run_data_process(label_, num):
    print(f'{label_} - run process raw record')
    data, label = process_raw_record_20_features(raw_data[label_])

    print(f'{label_} - run create dataset')
    temp_data, temp_label = create_dataset_20_features(data, label, filters, scalers)
    print(label_, temp_data.shape, temp_label.shape)
    temp_label[temp_label == 1] = num

    temp_data, temp_label = unison_shuffled_copies(temp_data, temp_label)
    temp_data = temp_data[:103000]
    temp_label = temp_label[:103000]
    train_idx = int(temp_data.shape[0] * 0.8)

    print(
        label_, 
        temp_data[:train_idx].shape,
        temp_label[:train_idx].shape,
        temp_data[train_idx:].shape,
        temp_label[train_idx:].shape,
        sep=' --- '
    )

    np.savez_compressed(
        f'./running/{label_}.npz',
        train_x=temp_data[:train_idx], 
        train_y=temp_label[:train_idx], 
        test_x=temp_data[train_idx:], 
        test_y=temp_label[train_idx:]
    )

# if __name__ == '__main__':
#     jobs = []
#     for num, label in enumerate(label_name):
#         p = multiprocessing.Process(target=run_data_process, args=(label, num+1))
#         jobs.append(p)
#         p.start()

#     for proc in jobs:
#         proc.join()


train_x = []
train_y = []
test_x = []
test_y = []

for label in label_name:
    dataset = np.load(f'./running/{label}.npz')

    print(dataset['train_x'].shape, dataset['train_y'].shape, dataset['test_x'].shape, dataset['test_y'].shape)

    train_x.append(dataset['train_x'][:51500])
    train_y.append(dataset['train_y'][:51500])
    test_x.append(dataset['test_x'][:12900])
    test_y.append(dataset['test_y'][:12900])

train_x = np.concatenate(train_x)
train_y = np.concatenate(train_y)
test_x = np.concatenate(test_x)
test_y = np.concatenate(test_y)


train_x = train_x.transpose((0, 2, 1))
test_x = test_x.transpose((0, 2, 1))

train_x = np.expand_dims(train_x, axis=-1)
test_x = np.expand_dims(test_x, axis=-1)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train_x, train_y = unison_shuffled_copies(train_x, train_y)



history = model.fit(
    [
        train_x[:, :4], 
        train_x[:, 4:8], 
        train_x[:, 8:12],
        train_x[:, 12:16],
        train_x[:, 16:20]
    ], 
    train_y,
    epochs=20,
    validation_data=(
        [
            test_x[:, :4], 
            test_x[:, 4:8], 
            test_x[:, 8:12],
            test_x[:, 12:16],
            test_x[:, 16:20]
        ],
        test_y
    ),
)


plt.figure(figsize=(20, 10)).suptitle("All labels")
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.savefig('orthogonal_train_result.png')


y_pred = model.predict([
    test_x[:, :4], 
    test_x[:, 4:8], 
    test_x[:, 8:12],
    test_x[:, 12:16],
    test_x[:, 16:20]
])
y_true = test_y
y_pred = np.argmax(y_pred, 2)


cm_total = np.zeros((6, 6))

for y_t, y_p in zip(y_true, y_pred):
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2, 3, 4, 5])
    cm = np.array(cm)
    cm_total = cm_total + cm


result = []
for cls in range(6):
    tp = cm_total[cls, cls]
    fn = np.sum(np.delete(cm_total[cls, :], cls))
    fp = np.sum(np.delete(cm_total[:, cls], cls))
    tn = np.delete(cm_total, cls, axis=0)
    tn = np.sum(np.delete(tn, cls, axis=1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + fn + tn + fp)
    specifity = tn/(tn+fp)

    result.append([precision, recall, f1, acc, specifity])

result = np.array(result)

print(f'precision, recall, f1, acc, specifity\n{result}')

plt.figure(figsize=(20, 10))
plt.title("Confusion Matrix of All labels Detection Model")
plt.matshow(result, fignum=False)
plt.xticks([0, 1, 2, 3, 4], ['Positive Predictive\nValue (Precision)', 'True Positive\nRate (Recall)', 'F1 Score', 'Accuracy', 'True Negative\nRate (Specifity)'])
plt.yticks([0, 1, 2, 3, 4, 5], ['Not command', 'eyebrows', 'left', 'right', 'both', 'teeth'])
plt.xlabel("Metric")
plt.ylabel("Class")
for (i, j), z in np.ndenumerate(result):
    plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
plt.colorbar()

plt.savefig('orthogonal_cm.png')

model.save(r'./checkpoints/orthogonal.keras')