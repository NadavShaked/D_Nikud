# ML
import argparse
# DL
import copy
import logging
import os
import random
from logging.handlers import RotatingFileHandler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# visual
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, RobertaForMaskedLM
from transformers import TrainingArguments
import torch.nn.functional as F

from src.plot_helpers import plot_results
from src.utiles_data import NikudDataset, prepare_data, Nikud, Letters, DEBUG_MODE
from pathlib import Path

# DL
# HF

# TODO: EXTRACT TO models/utils.py
def save_model(model, path):
    model_state = model.state_dict()
    torch.save(model_state, path)

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda'
cols = ["precision", "recall", "f1-score", "support"]
SEED = 42
SAVE_MODEL_CHECKPOINT_EVERY_EPOCHS = 2

SAVED_MODEL_FOLDER = 'saved_models'
if not os.path.exists(SAVED_MODEL_FOLDER):
    os.makedirs(SAVED_MODEL_FOLDER)

# Set the random seed for Python
random.seed(SEED)

# Set the random seed for numpy
np.random.seed(SEED)

# Set the random seed for torch to SEED
torch.manual_seed(SEED)


# def model(model="imvladikon/alephbertgimmel-base-512"):
#     model = AutoModelForMaskedLM.from_pretrained("imvladikon/alephbertgimmel-base-512")
# DMtokenizer = AutoTokenizer.from_pretrained("imvladikon/alephbertgimmel-base-512")

def get_parameters(params):
    top_layer_params = []
    for name, param in params:
        print(name)
        if "lm_head" in name or name.startswith("LayerNorm") or name.startswith('classifier'):  # 'layer.11' in name or
            top_layer_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    return top_layer_params


class RobertaWithoutLMHead(RobertaForMaskedLM):
    def __init__(self, config):
        super(RobertaWithoutLMHead, self).__init__(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        # Call the forward method of the base class (RobertaForMaskedLM)
        outputs = super(RobertaWithoutLMHead, self).forward(input_ids, attention_mask=attention_mask,
                                                            token_type_ids=token_type_ids,
                                                            position_ids=position_ids,
                                                            head_mask=head_mask, output_hidden_states=True
                                                            )

        # Exclude the lm_head's output from the outputs
        last_hidden_states = outputs.hidden_states[-1]

        return last_hidden_states


class DiacritizationModel(nn.Module):
    def __init__(self, base_model_name):
        super(DiacritizationModel, self).__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        self.model = RobertaWithoutLMHead.from_pretrained(base_model_name).to(
            DEVICE)
        # self.model.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.classifier_nikud = nn.Linear(config.hidden_size, Nikud.LEN_NIKUD)
        self.classifier_sin = nn.Linear(config.hidden_size, Nikud.LEN_SIN)
        self.classifier_dagesh = nn.Linear(config.hidden_size, Nikud.LEN_DAGESH)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        try:
            last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
            normalized_hidden_states = self.LayerNorm(last_hidden_state)

            # Classifier for Nikud
            nikud_logits = self.classifier_nikud(normalized_hidden_states)
            nikud_probs = self.softmax(nikud_logits)

            # Classifier for Dagesh
            dagesh_logits = self.classifier_dagesh(normalized_hidden_states)
            dagesh_probs = self.softmax(dagesh_logits)

            # Classifier for Sin
            sin_logits = self.classifier_sin(normalized_hidden_states)
            sin_probs = self.softmax(sin_logits)
        except:
            a = 1
        # Return the probabilities for each diacritical mark
        return nikud_probs, dagesh_probs, sin_probs


def training(model, n_epochs, train_data, dev_data, criterion_nikud, criterion_dagesh, criterion_sin, logger,
             optimizer=None):
    best_accuracy = 0.0
    best_model_weights = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(torch.cuda.is_available())
    model = model.to(device)

    criterion_nikud.to(device)
    criterion_dagesh.to(device)
    criterion_sin.to(device)

    train_loader = train_data
    dev_loader = dev_data
    max_length = None
    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        train_loss = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        sum = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}

        for index_data, data in enumerate(train_loader):
            # if DEBUG_MODE and index_data > 100:
            #     break
            (inputs, attention_mask, labels) = data
            if max_length is None:
                max_length = labels.shape[1]
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

            # sep_id = 2
            # length_sentences = np.where(np.array(inputs.data) == sep_id)[1] - 1
            for i, (probs, name_class) in enumerate(
                    zip([nikud_probs, dagesh_probs, sin_probs], ["nikud", "dagesh", "sin"])):
                reshaped_tensor = torch.transpose(probs, 1, 2).contiguous().view(probs.shape[0],
                                                                                 probs.shape[2],
                                                                                 probs.shape[1])
                loss = criterion_nikud(reshaped_tensor, labels[:, :, i]).to(device)

                num_relevant = (labels[:, :, i] != -1).sum()
                train_loss[name_class] += loss.item() * num_relevant
                sum[name_class] += num_relevant

                loss.backward(retain_graph=True)

            optimizer.step()
            if (index_data + 1) % 100 == 0:
                msg = f'epoch: {epoch} , index_data: {index_data + 1}\n' \
                      f'mean loss train nikud: {(train_loss["nikud"] / (sum["nikud"]))}, ' \
                      f'mean loss train dagesh: {(train_loss["dagesh"] / (sum["dagesh"]))}, ' \
                      f'mean loss train sin: {(train_loss["sin"] / (sum["sin"]))}'
                logger.debug(msg)

        for name_class in train_loss.keys():
            train_loss[name_class] /= sum[name_class]

        msg = f"Epoch {epoch + 1}/{n_epochs}\n" \
              f'mean loss train nikud: {train_loss["nikud"]}, ' \
              f'mean loss train dagesh: {train_loss["dagesh"]}, ' \
              f'mean loss train sin: {train_loss["sin"]}'
        logger.debug(msg)

        model.eval()
        dev_loss = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        dev_accuracy = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        sum = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        correct_preds = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        masks = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        predictions = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        labels_class = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}

        correct_preds_letter = 0.0

        sum_all = 0.0
        with torch.no_grad():
            for index_data, data in enumerate(dev_loader):
                (inputs, attention_mask, labels) = data

                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

                for i, (probs, name_class) in enumerate(
                        zip([nikud_probs, dagesh_probs, sin_probs], ["nikud", "dagesh", "sin"])):
                    reshaped_tensor = torch.transpose(probs, 1, 2).contiguous().view(probs.shape[0],
                                                                                     probs.shape[2],
                                                                                     probs.shape[1])
                    loss = criterion_nikud(reshaped_tensor, labels[:, :, i]).to(device)
                    mask = labels[:, :, i] != -1
                    num_relevant = mask.sum()
                    sum[name_class] += num_relevant
                    _, preds = torch.max(nikud_probs, 2)
                    dev_loss[name_class] += loss.item() * num_relevant
                    correct_preds[name_class] += torch.sum(preds[mask] == labels[:, :, i][mask])
                    masks[name_class] = mask
                    predictions[name_class] = preds
                    labels_class[name_class] = labels[:, :, i]

                mask_all_or = torch.logical_or(torch.logical_or(masks["nikud"], masks["dagesh"]), masks["sin"])
                correct_preds_letter += torch.sum(
                    torch.logical_and(torch.logical_and(predictions["sin"][mask_all_or] == \
                                                        labels_class["sin"][mask_all_or],
                                                        predictions["dagesh"][mask_all_or] == \
                                                        labels_class["dagesh"][mask_all_or]),
                                      predictions["nikud"][mask_all_or] == \
                                      labels_class["nikud"][mask_all_or]))

                sum_all += mask_all_or.sum()

        for name_class in dev_loss.keys():
            dev_loss[name_class] /= sum[name_class]
            dev_accuracy[name_class] = correct_preds[name_class].double() / sum[name_class]

        dev_accuracy_letter = correct_preds_letter.double() / sum_all

        # tqdm.write(
        #     f"Epoch {epoch + 1}/{n_epochs}, "
        #     f"Dev Nikud Loss: {dev_loss['nikud']:.4f}, Dev Nikud Accuracy: {dev_accuracy['nikud']:.4f}"
        #     f", Dev dagesh Loss: {dev_loss['dagesh']:.4f}, Dev dagesh Accuracy: {dev_accuracy['dagesh']:.4f}"
        #     f", Dev sin Loss: {dev_loss['sin']:.4f}, Dev sin Accuracy: {dev_accuracy['sin']:.4f}"
        #     f"Dev letter Accuracy: {dev_accuracy_letter:.4f}")

        msg = f"Epoch {epoch + 1}/{n_epochs}\n" \
              f'mean loss Dev nikud: {train_loss["nikud"]}, ' \
              f'mean loss Dev dagesh: {train_loss["dagesh"]}, ' \
              f'mean loss Dev sin: {train_loss["sin"]}, ' \
              f'Dev letter Accuracy: {dev_accuracy_letter}'
        logger.debug(msg)

        # calc accuracy by letter

        if dev_accuracy_letter > best_accuracy:
            best_accuracy = dev_accuracy_letter

            save_model_path = os.path.join(SAVED_MODEL_FOLDER, 'best_model.pth')
            save_model(model, save_model_path)  # TODO: use this function in model class

        if epoch % SAVE_MODEL_CHECKPOINT_EVERY_EPOCHS == 0:
            save_model_path = os.path.join(SAVED_MODEL_FOLDER, f'checkpoint_model_o_nepoch_{epoch}.pth')
            save_model(model, save_model_path)  # TODO: use this function in model class

    a = 1
    # Load the weights of the best model
    # model.load_state_dict(best_model_weights) - TODO - MAKE IT WORK


def evaluate(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []
    predicted_labels_2_report = []
    masks = []
    reports = {}
    correct_preds = {"nikud": 0, "dagesh": 0, "sin": 0}
    sum = {"nikud": 0, "dagesh": 0, "sin": 0}

    word_level_correct = 0.0
    letter_level_correct = 0.0

    with torch.no_grad():
        for data in test_data:
            (inputs, attention_mask, labels) = data

            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

            for i, (probs, name_class) in enumerate(
                    zip([nikud_probs, dagesh_probs, sin_probs], ["nikud", "dagesh", "sin"])):
                mask = labels[:, :, i] != -1
                num_relevant = mask.sum()
                sum[name_class] += num_relevant
                _, preds = torch.max(probs, 2)
                correct_preds[name_class] += torch.sum(preds[mask] == labels[:, :, i][mask])
                predicted_labels.append(preds)
                masks.append(mask)
                true_labels.extend(labels[:, :, i][mask].cpu().numpy())
                predicted_labels_2_report.extend(preds[mask].cpu().numpy())

            mask_all_or = torch.logical_or(torch.logical_or(masks[0], masks[1]), masks[2])
            mask_correct_letter = torch.logical_and(torch.logical_and(predicted_labels[0][mask_all_or] == \
                                                                      labels[:, :, 0][mask_all_or],
                                                                      predicted_labels[1][mask_all_or] == \
                                                                      labels[:, :, 1][mask_all_or]),
                                                    predicted_labels[2][mask_all_or] == \
                                                    labels[:, :, 2][mask_all_or])
            letter_level_correct += torch.sum(mask_correct_letter)

    for i, name in enumerate(["nikud", "dagesh", "sin"]):
        report = classification_report(true_labels[i], predicted_labels_2_report[i],
                                       target_names=list(Nikud.label_2_id[name].keys()),
                                       output_dict=True)
        reports[name] = report

        cm = confusion_matrix(true_labels, predicted_labels)
        cm_df = pd.DataFrame(cm, index=list(Nikud.label_2_id[name].keys()), columns=list(Nikud.label_2_id[name].keys()))

        # Display confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.show()

    return reports, word_level_correct, letter_level_correct


OUTPUT_DIR = 'models/trained/latest'


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Save directory for model')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of train epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (train)')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Validation batch size')
    parser.add_argument('--save_strategy', '-lr', type=str, default='no',
                        help='Whether to save on every epoch ("epoch"/"no")')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help='Learning rate scheduler type ("linear"/"cosine"/"constant"/...')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='AdamW beta1 hyperparameter')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='AdamW beta2 hyperparameter')
    parser.add_argument('--weight_decay', type=float, default=0.15, help='Weight decay')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        help='How to validate (set to "no" for no validation)')
    parser.add_argument('--eval_steps', type=int, default=2000, help='Validate every N steps')
    parser.add_argument("-l", "--log", dest="loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default="DEBUG", help="Set the logging level")
    parser.add_argument("-ld", "--log_dir", dest="log_dir",
                        default=os.path.join(Path(__file__).parent.parent, "logging"), help="Set the logger path")
    parser.add_argument("-df", "--debug_folder", dest="debug_folder",
                        default=os.path.join(Path(__file__).parent.parent, "plots"), help="Set the debug folder")
    parser.add_argument("-dataf", "--data_folder", dest="data_folder",
                        default=os.path.join(Path(__file__).parent.parent, "data/hebrew_diacritized"),
                        help="Set the debug folder")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger = get_logger(args.loglevel, args.log_dir, args.debug_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    msg = f'Device detected: {device}'
    logger.info(msg)

    # training_args = TrainingArguments(**vars(args))  # vars: Namespace to dict

    msg = 'Loading data...'
    logger.debug(msg)

    dataset = NikudDataset(folder=args.data_folder, logger=logger)
    dataset.show_data_labels(debug_folder=debug_folder)
    dataset.calc_max_length()

    msg = f'Max length of data: {dataset.max_length}'
    logger.debug(msg)

    train, test = train_test_split(dataset.data, test_size=0.1, shuffle=True, random_state=SEED)
    train, dev = train_test_split(train, test_size=0.1, shuffle=True, random_state=SEED)

    msg = f'Num rows in train data: {len(train)}, ' \
          f'Num rows in dev data: {len(dev)}, ' \
          f'Num rows in test data: {len(test)}'
    logger.debug(msg)

    msg = 'Loading tokenizer and prepare data...'
    logger.debug(msg)

    DMtokenizer = AutoTokenizer.from_pretrained("tau/tavbert-he")
    mtb_train_dl = prepare_data(train, DMtokenizer, dataset.max_length, batch_size=32, name="train")
    mtb_dev_dl = prepare_data(dev, DMtokenizer, dataset.max_length, batch_size=32, name="dev")
    mtb_test_dl = prepare_data(test, DMtokenizer, dataset.max_length, batch_size=32, name="test")

    msg = 'Loading model...'
    logger.debug(msg)

    model_DM = DiacritizationModel("tau/tavbert-he").to(DEVICE)
    all_model_params_MTB = model_DM.named_parameters()
    top_layer_params = get_parameters(all_model_params_MTB)
    optimizer = torch.optim.Adam(top_layer_params, lr=args.learning_rate)

    msg = 'training...'
    logger.debug(msg)

    criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    training(model_DM, 5, mtb_train_dl, mtb_dev_dl, criterion_nikud, criterion_dagesh, criterion_sin, logger,
             optimizer=optimizer)

    report_dev, word_level_correct_dev, letter_level_correct_dev = evaluate(model_DM, mtb_dev_dl)
    report_test, word_level_correct_test, letter_level_correct_test = evaluate(model_DM, mtb_test_dl)

    msg = f"Diacritization Model\nDev dataset\nLetter level accuracy:{letter_level_correct_dev}\n" \
          f"Word level accuracy: {word_level_correct_dev}\n--------------------\nTest dataset\n" \
          f"Letter level accuracy: {letter_level_correct_test}\nWord level accuracy: {word_level_correct_test}"
    logger.debug(msg)

    plot_results(report_dev, report_filename="results_dev")
    plot_results(report_test, report_filename="results_test")

    msg = 'Done'
    logger.info(msg)


def get_logger(log_level, log_location, debug_folder):
    log_format = '%(asctime)s %(levelname)-8s Thread_%(thread)-6d ::: %(funcName)s(%(lineno)d) ::: %(message)s'
    logger = logging.getLogger("algo")
    logger.setLevel(getattr(logging, log_level))
    cnsl_log_formatter = logging.Formatter(log_format)
    cnsl_handler = logging.StreamHandler()
    cnsl_handler.setFormatter(cnsl_log_formatter)
    cnsl_handler.setLevel(log_level)
    logger.addHandler(cnsl_handler)

    if not os.path.exists(log_location):
        os.makedirs(log_location)

    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    file_location = os.path.join(log_location,
                                 'Diacritization_Model_DEBUG.log')
    file_log_formatter = logging.Formatter(log_format)

    SINGLE_LOG_SIZE = 2 * 1024 * 1024  # in Bytes
    MAX_LOG_FILES = 20
    file_handler = RotatingFileHandler(file_location,
                                       mode='a',
                                       maxBytes=SINGLE_LOG_SIZE,
                                       backupCount=MAX_LOG_FILES)
    file_handler.setFormatter(file_log_formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    main()
