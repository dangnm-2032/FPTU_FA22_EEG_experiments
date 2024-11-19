from FPTU_FA24_EEG_Artifacts_Recognition.config import *
from tensorflow.keras.models import Model, load_model
import datasets
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Validator:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def load_test_data(self):
        dataset_config = self.config.get_dataset_config()
        self.test_data = datasets.load_from_disk(dataset_config.save_test_data)
    
    def load_model(self):
        model_config = self.config.get_eeg_model_config()
        self.model = load_model(Path(os.path.join(
            model_config.save_path,
            model_config.save_name + model_config.weight_extension
        )))

    def evaluate(self):
        model_config = self.config.get_eeg_model_config()
        os.makedirs(Path(os.path.join(
            RESULT_FOLDER_PATH,
            model_config.save_name
        )), exist_ok=True)

        # Plot training history
        history = json.load(
            open(
                Path(os.path.join(
                    model_config.save_path,
                    model_config.save_name + model_config.history_extension
                )), 
                'r'
            )
        )
        plt.figure(figsize=(20, 10)).suptitle("All labels")
        plt.subplot(121)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.subplot(122)
        plt.plot(history['sparse_categorical_accuracy'])
        plt.plot(history['val_sparse_categorical_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.savefig(
            Path(os.path.join(
                RESULT_FOLDER_PATH,
                model_config.save_name,
                'training_plot.png'
            ))
        )

        # Calculate each
        y_pred = self.model.predict([
            self.test_data['eyebrows'],
            self.test_data['left'],
            self.test_data['right'],
            self.test_data['both'],
            self.test_data['teeth'],
        ])
        y_true = self.test_data['label']
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

        plt.savefig(
            Path(os.path.join(
                RESULT_FOLDER_PATH,
                model_config.save_name,
                'metrics_of_each_label.png'
            ))
        )