# EEG Artifacts Recognition

## Introduction:
You want to control a robot car with your mind? I'm sorry that I cannot help you. 
But I can help you in control a robot car with facial movements!

## Features:
- Python application runs on Windows.
- Training code in Ubuntu/Windows, using Tensorflow.
- Open source dataset [Kaggle](https://www.kaggle.com/datasets/dangnguyenyuu/muse-2-eeg-facial-movements-dataset).

## How to use
### Installation
1. Clone this repo to your local machine.
2. Create virtual environment with **Python 3.10.14**.
3. `pip install -r requirements_app.txt` or `pip install -r requirement_dev.txt` whether you want to training or just run the app.
4. Download dataset and extract it here, it need to be an `artifacts` folder has the same level with this README.md.

### Training
1. Take a look into `./config/*.yaml`. There are three files, respectively: 
    - `config.yaml` for whether you want to skip preprocess or change the checkpoint's save name. 
    - `dataset.yaml` for information about the dataset, in future if you record your own dataset to train, you will need to define in this file.
    - `params.yaml` for model hyperparameters and train process.
2. After the configuration, you only need to run `python main.py`.
3. When complete the training, the model checkpoints and raw training history will be saved in `./artifacts/checkpoints/model/`. The training and evaluating figures will be saved in `./results/`.
### Inference
1. Open `./config/config.yaml`, in `inference_model`, copy the file path of your desired model checkpoint and paste directly to it.
2. Run `python app.py`.