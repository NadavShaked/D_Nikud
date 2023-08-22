# D-Nikud

This repository contains the main code for a diacritization (nikud) model using the TAVBERT architecture. The model is designed to predict diacritics (nikud) for Hebrew text. The code includes functionalities for predicting, evaluating, and training the diacritization model.

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

The main code in this repository implements a diacritization model using the TAVBERT architecture. The model aims to predict diacritics for Hebrew text, which improves the correct pronunciation and interpretation of the text.

## Usage

Clone the repository:

   ```bash
   git clone https://github.com/NadavShaked/nlp-final-project.git
   cd d-nikud-prediction
   ```

### Predict

To predict diacritics for input text files or folders, you can use the following command:

```bash
python main.py predict <input_path> <output_path> [-c/--compare <compare_nakdimon>]
```

- `<input_path>`: Path to the input file or folder containing text data.
- `<output_path>`: Path to the output file where the predicted diacritized text will be saved.
- `-c/--compare`: Optional. Set to `True` to predict text for comparison with Nakdimon.

### Evaluate

To evaluate the diacritization model, you can use the following command:

```bash
python main.py evaluate <input_path> [-df/--plots_folder <plots_folder>]
```

- `<input_path>`: Path to the input file or folder containing text data for evaluation.
- `-df/--plots_folder`: Optional. Path to the folder where evaluation plots will be saved.

### Train

To train the diacritization model, use the following command:

```bash
python main.py train [--from_pretrain] [--learning_rate <learning_rate>] [--batch_size <batch_size>]
                    [--n_epochs <n_epochs>] [--data_folder <data_folder>] [--checkpoints_frequency <checkpoints_frequency>]
                    [-df/--plots_folder <plots_folder>]
```

- `--from_pretrain`: Optional. Continue training from a pretrained model.
- `--learning_rate`: Optional. Learning rate for training (default is 0.001).
- `--batch_size`: Optional. Batch size for training (default is 32).
- `--n_epochs`: Optional. Number of training epochs (default is 10).
- `--data_folder`: Optional. Path to the folder containing training data (default is "data").
- `--checkpoints_frequency`: Optional. Frequency of saving model checkpoints during training (default is 1).
- `-df/--plots_folder`: Optional. Path to the folder where training plots will be saved.

## Acknowledgments

This script utilizes the D-Nikud model developed by [Adi Rosenthal](https://github.com/Adirosenthal) and [Nadav Shaked](https://github.com/NadavShaked).


## License

This code is provided under the [MIT License](https://www.mit.edu/~amini/LICENSE.md). You are free to use, modify, and distribute the code according to the terms of the license.