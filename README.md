# D-Nikud

Welcome to the D-Nikud Diacritization Model main code repository! This repository is dedicated to the implementation of our innovative D-Nikud model, which use the TavBERT architecture and Bi-LSTM to predict and apply diacritics (nikud) to Hebrew text. Diacritics play a crucial role in accurately conveying pronunciation and interpretation, making our model an essential tool for enhancing the quality of Hebrew text analysis.

The code provided here encompasses various functionalities, including prediction, evaluation, and training of the D-Nikud diacritization model. 

## Prerequisites

Before running the script, make sure you have the following installed:

- Python 3.6 or higher
- `torch` library (PyTorch)
- `transformers` library
- Required Python packages (Install using `pip install -r requirements.txt`)

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
  - [Predict](#predict)
  - [Evaluate](#evaluate)
  - [Train](#train)
- [Requirements](#requirements)
- [License](#license)

## Introduction

Our D-Nikud model utilizes the TevBERT architecture and Bi-LSTM for diacritization (nikud) of Hebrew text. Diacritics (nikud) are essential for accurate pronunciation and interpretation of the text. This repository provides the core code for implementing and utilizing the D-Nikud model.

## Usage

Clone the repository:

   ```bash
   git clone https://github.com/NadavShaked/nlp-final-project.git
   cd d-nikud-prediction
   ```

### Predict

The "Predict" command enables the prediction of diacritics for input text files or folders containing un-diacritized text. It generates diacritization predictions using the specified diacritization model and saves the results to the specified output file. Optionally, you can choose to predict text for comparison with Nakdimon using the `-c/--compare` flag.

To predict diacritics for input text files or folders, use the following command:

```bash
python main.py predict <input_path> <output_path> [-c/--compare <compare_nakdimon>] [-ptmp/--pretrain_model_path <pretrain_model_path>]
```

- `<input_path>`: Path to the input file or folder containing text data.
- `<output_path>`: Path to the output file where the predicted diacritized text will be saved.
- `-c/--compare`: Optional. Set to `True` to predict text for comparison with Nakdimon.
- `-ptmp/--pretrain_model_path`: Optional. Path to the pre-trained model weights to be used for prediction. If not provided, the command will default to using our pre-trained D-Nikud model.

For example, to predict diacritics for a specific input text file and save the results to an output file, you can execute:

```bash
python main.py predict input.txt output.txt
```

If you wish to predict text for comparison with Nakdimon and specify a custom pre-trained model path, you can use:

```bash
python main.py predict input_folder output_folder -c True -ptmp path/to/pretrained/model.pth
```

Here, the command will predict diacritics for the texts in the `input_folder`, generate output files in the `output_folder`, and use the specified pre-trained model for prediction.

You can adapt the paths and options to suit your project's requirements. If the -ptmp parameter is omitted, the command will automatically employ our default pre-trained D-Nikud model for prediction.

### Evaluate

The "Evaluate" command assesses the performance of the diacritization model by computing accuracy metrics for specific diacritics elements: nikud, dagesh, sin, as well as overall letter and word accuracy. This evaluation process involves comparing the model's diacritization results with the original diacritics text, providing insights into the model's effectiveness in accurately predicting and applying diacritics.

To evaluate the diacritization model, you can use the following command:

```bash
python main.py evaluate <input_path> [-ptmp/--pretrain_model_path <pretrain_model_path>] [-df/--plots_folder <plots_folder>]
```

- `<input_path>`: Path to the input file or folder containing text data for evaluation.
- `-ptmp/--pretrain_model_path`: Optional. Path to the pre-trained model weights to be employed for evaluation. If this parameter is not specified, the command will default to using our pre-trained D-Nikud model.
- `-df/--plots_folder`: Optional. Path to the folder where evaluation plots will be saved. If not provided, the default plots folder will be used.

For example, to evaluate the diacritization model's performance on a specific dataset, you might run:

```bash
python main.py evaluate dataset_folder -ptmp path/to/pretrained/model.pth -df evaluation_plots
```

This command will evaluate the model's accuracy on the dataset found in the `dataset_folder`, using the specified pre-trained model weights and saving evaluation plots in the `evaluation_plots` folder.

### Train

The "Train" command enables the training of the diacritization model using your own dataset. This command supports fine-tuning a pre-trained model, adjusting hyperparameters such as learning rate and batch size, and specifying various training settings.

To train the diacritization model, use the following command:

```bash
python main.py train [--learning_rate <learning_rate>] [--batch_size <batch_size>]
                    [--n_epochs <n_epochs>] [--data_folder <data_folder>] [--checkpoints_frequency <checkpoints_frequency>]
                    [-df/--plots_folder <plots_folder>] [-ptmp/--pretrain_model_path <pretrain_model_path>]
```

- `--learning_rate`: Optional. Learning rate for training (default is 0.001).
- `--batch_size`: Optional. Batch size for training (default is 32).
- `--n_epochs`: Optional. Number of training epochs (default is 10).
- `--data_folder`: Optional. Path to the folder containing training data (default is "data").
- `--checkpoints_frequency`: Optional. Frequency of saving model checkpoints during training (default is 1).
- `-df/--plots_folder`: Optional. Path to the folder where training plots will be saved.
- `-ptmp/--pretrain_model_path`: Optional. Path to the pre-trained model weights to be used for training continuation. Use this only if you want to fine-tune a specific pre-trained model.

For instance, to initiate training with a specified learning rate, batch size, and number of epochs, you can execute:

```bash
python main.py train --learning_rate 0.001 --batch_size 16 --n_epochs 20
```

If you want to continue training from a pre-trained model and save model checkpoints every 3 epochs, you can use:

```bash
python main.py train --from_pretrain --checkpoints_frequency 3 -ptmp path/to/pretrained/model.pth
```

In this example, the command will resume training from the specified pre-trained model and save checkpoints every 3 epochs. Training logs and plots will be saved in the specified plots folder.

Remember to adjust the command options according to your training requirements and preferences. If you don't provide the `-ptmp` parameter, the command will start training from scratch using the default D-Nikud model architecture.

## Acknowledgments

This script utilizes the D-Nikud model developed by [Adi Rosenthal](https://github.com/Adirosenthal) and [Nadav Shaked](https://github.com/NadavShaked).


## License

This code is provided under the [MIT License](https://www.mit.edu/~amini/LICENSE.md). You are free to use, modify, and distribute the code according to the terms of the license.