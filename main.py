# general
import argparse
import os
import random
import sys
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ML
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

# DL
from src.models import DNikudModel, ModelConfig
from src.models_utils import training, evaluate, predict
from src.plot_helpers import generate_plot_by_nikud_dagesh_sin_dict, \
    generate_word_and_letter_accuracy_plot
from src.running_params import SEED, BEST_MODEL_PATH, BATCH_SIZE, MAX_LENGTH_SEN
from src.utiles_data import NikudDataset, Nikud, get_sub_folders_paths, create_missing_folders, \
    extract_text_to_compare_nakdimon

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda'

# TODO: DELETE SEEDS

# Set the random seed for Python
random.seed(SEED)

# Set the random seed for numpy
np.random.seed(SEED)

# Set the random seed for torch to SEED
torch.manual_seed(SEED)


# def train(use_pretrain=False):
#     args = parse_arguments()
#
#     output_model_dir, output_log_dir, output_dir_running, plots_folder = generate_folders(args,
#                                                                                           name_log=f"log_model_lr_{args.learning_rate}_bs_{BATCH_SIZE}")
#
#     logger = get_logger(args.loglevel, output_log_dir)
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     msg = f'Device detected: {device}'
#     logger.info(msg)
#
#     msg = 'Loading data...'
#     logger.debug(msg)
#
#     tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")
#
#     dataset_train = NikudDataset(tokenizer_tavbert,
#                                  folder=os.path.join(args.data_folder, "train"),
#                                  logger=logger,
#                                  max_length=MAX_LENGTH_SEN,
#                                  is_train=True)
#     dataset_dev = NikudDataset(tokenizer=tokenizer_tavbert,
#                                folder=os.path.join(args.data_folder, "dev"),
#                                logger=logger,
#                                max_length=dataset_train.max_length,
#                                is_train=True)
#     dataset_test = NikudDataset(tokenizer=tokenizer_tavbert,
#                                 folder=os.path.join(args.data_folder, "test"),
#                                 logger=logger,
#                                 max_length=dataset_train.max_length,
#                                 is_train=True)
#
#     dataset_train.show_data_labels(plots_folder=plots_folder)
#
#     msg = f'Max length of data: {dataset_train.max_length}'
#     logger.debug(msg)
#
#     msg = f'Num rows in train data: {len(dataset_train.data)}, ' \
#           f'Num rows in dev data: {len(dataset_dev.data)}, ' \
#           f'Num rows in test data: {len(dataset_test.data)}'
#     logger.debug(msg)
#
#     msg = 'Loading tokenizer and prepare data...'
#     logger.debug(msg)
#
#     dataset_train.prepare_data(name="train")
#     dataset_dev.prepare_data(name="dev")
#     dataset_test.prepare_data(name="test")
#
#     mtb_train_dl = torch.utils.data.DataLoader(dataset_train.prepered_data, batch_size=BATCH_SIZE)
#     mtb_dev_dl = torch.utils.data.DataLoader(dataset_dev.prepered_data, batch_size=BATCH_SIZE)
#     mtb_test_dl = torch.utils.data.DataLoader(dataset_test.prepered_data, batch_size=BATCH_SIZE)
#
#     msg = 'Loading model...'
#     logger.debug(msg)
#
#     base_model_name = "tau/tavbert-he"
#     config = AutoConfig.from_pretrained(base_model_name)
#     model_DM = DNikudModel(config,
#                            len(Nikud.label_2_id["nikud"]),
#                            len(Nikud.label_2_id["dagesh"]),
#                            len(Nikud.label_2_id["sin"]),
#                            pretrain_model=base_model_name,
#                            device=DEVICE
#                            ).to(DEVICE)
#
#     if use_pretrain:
#         # load last best model:
#         state_dict_model = model_DM.state_dict()
#         state_dict_model.update(
#             torch.load(BEST_MODEL_PATH))
#         model_DM.load_state_dict(state_dict_model)
#
#     dir_model_config = os.path.join(output_model_dir, "config.yml")
#
#     if not os.path.isfile(dir_model_config):
#         our_model_config = ModelConfig(dataset_train.max_length)
#         our_model_config.save_to_file(dir_model_config)
#
#     optimizer = torch.optim.Adam(model_DM.parameters(), lr=args.learning_rate)
#
#     msg = 'training...'
#     logger.debug(msg)
#
#     criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
#     criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
#     criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
#
#     training_params = {"n_epochs": args.n_epochs, "checkpoints_frequency": args.checkpoints_frequency}
#     (best_model_details, best_accuracy, epochs_loss_train_values, steps_loss_train_values, loss_dev_values,
#      accuracy_dev_values) = training(
#         model_DM,
#         mtb_train_dl,
#         mtb_dev_dl,
#         criterion_nikud,
#         criterion_dagesh,
#         criterion_sin,
#         training_params,
#         logger,
#         output_dir_running,
#         optimizer,
#         device=DEVICE
#     )
#
#     generate_plot_by_nikud_dagesh_sin_dict(epochs_loss_train_values, "Train epochs loss", "Loss", plots_folder)
#     generate_plot_by_nikud_dagesh_sin_dict(steps_loss_train_values, "Train steps loss", "Loss", plots_folder)
#     generate_plot_by_nikud_dagesh_sin_dict(loss_dev_values, "Dev epochs loss", "Loss", plots_folder)
#     generate_plot_by_nikud_dagesh_sin_dict(accuracy_dev_values, "Dev accuracy", "Accuracy", plots_folder)
#     generate_word_and_letter_accuracy_plot(accuracy_dev_values, plots_folder)
#
#     report_dev, word_level_correct_dev, letter_level_correct_dev = evaluate(model_DM, mtb_dev_dl, plots_folder,
#                                                                             device=DEVICE)
#     report_test, word_level_correct_test, letter_level_correct_test = evaluate(model_DM, mtb_test_dl, plots_folder,
#                                                                                device=DEVICE)
#
#     msg = f"Diacritization Model\nDev dataset\nLetter level accuracy:{letter_level_correct_dev}\n" \
#           f"Word level accuracy: {word_level_correct_dev}\n--------------------\nTest dataset\n" \
#           f"Letter level accuracy: {letter_level_correct_test}\nWord level accuracy: {word_level_correct_test}"
#     logger.debug(msg)
#
#     plot_results(logger, report_dev, report_filename="results_dev")
#     plot_results(logger, report_test, report_filename="results_test")
#
#     msg = 'Done'
#     logger.info(msg)


def get_logger(log_level, name_func, date_time=datetime.now().strftime('%d_%m_%y__%H_%M')):
    log_location = os.path.join(os.path.join(Path(__file__).parent, "logging"), f"log_model_{name_func}_{date_time}")
    create_missing_folders(log_location)

    log_format = '%(asctime)s %(levelname)-8s Thread_%(thread)-6d ::: %(funcName)s(%(lineno)d) ::: %(message)s'
    logger = logging.getLogger("algo")
    logger.setLevel(getattr(logging, log_level))
    cnsl_log_formatter = logging.Formatter(log_format)
    cnsl_handler = logging.StreamHandler()
    cnsl_handler.setFormatter(cnsl_log_formatter)
    cnsl_handler.setLevel(log_level)
    logger.addHandler(cnsl_handler)

    create_missing_folders(log_location)

    file_location = os.path.join(log_location, 'Diacritization_Model_DEBUG.log')
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


# def hyperparams_checker(use_pretrain=False):
#     args = parse_arguments()
#
#     plots_folder = args.plots_folder
#     create_missing_folders(plots_folder)
#
#     tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")
#
#     dataset_train = NikudDataset(tokenizer_tavbert, folder=os.path.join(args.data_folder, "train"), logger=None,
#                                  max_length=MAX_LENGTH_SEN, is_train=True)
#     dataset_dev = NikudDataset(tokenizer=tokenizer_tavbert, folder=os.path.join(args.data_folder, "dev"), logger=None,
#                                max_length=dataset_train.max_length, is_train=True)
#     dataset_test = NikudDataset(tokenizer=tokenizer_tavbert, folder=os.path.join(args.data_folder, "test"),
#                                 logger=None, max_length=dataset_train.max_length, is_train=True)
#
#     dataset_train.prepare_data(name="train")
#     dataset_dev.prepare_data(name="dev")
#     dataset_test.prepare_data(name="test")
#
#     # hyperparameters search space
#     lr_values = np.logspace(-6, -1, num=6)  # learning rates between 1e-6 and 1e-1
#     num_freeze_layers = list(range(1, 10, 2))  # learning rates between 1e-6 and 1e-1
#     batch_size_values = [2 ** i for i in range(3, 7)]  # batch sizes between 32 and 512
#
#     # number of random combinations to test
#     num_combinations = 20
#
#     # best hyperparameters and their performance
#     best_accuracy = 0.0
#     best_hyperparameters = None
#
#     training_params = {"n_epochs": args.n_epochs, "checkpoints_frequency": args.checkpoints_frequency}
#
#     for _ in range(num_combinations):
#         torch.cuda.empty_cache()
#         lr = np.random.choice(lr_values)
#         nfl = np.random.choice(num_freeze_layers)
#         batch_size = int(np.random.choice(batch_size_values))
#
#         output_model_dir, output_log_dir, output_dir_running, plots_folder = generate_folders(args,
#                                                                                               name_log=f"log_model_lr_{lr}_bs_{batch_size}_nfl_{nfl}")
#         logger = get_logger(args.loglevel, output_log_dir)
#
#         msg = 'Loading model...'
#         logger.debug(msg)
#
#         base_model_name = "tau/tavbert-he"
#         config = AutoConfig.from_pretrained(base_model_name)
#
#         model_DM = DNikudModel(config,
#                                len(Nikud.label_2_id["nikud"]),
#                                len(Nikud.label_2_id["dagesh"]),
#                                len(Nikud.label_2_id["sin"]),
#                                device=DEVICE
#                                ).to(DEVICE)
#         if use_pretrain:
#             # load last best model:
#             state_dict_model = model_DM.state_dict()
#             state_dict_model.update(
#                 torch.load(BEST_MODEL_PATH))
#             model_DM.load_state_dict(state_dict_model)
#
#         # set these hyperparameters in your optimizer
#         optimizer = torch.optim.Adam(model_DM.parameters(), lr=args.learning_rate)
#
#         # redefine your data loaders with the new batch size
#         mtb_train_dl = torch.utils.data.DataLoader(dataset_train.prepered_data, batch_size=batch_size)
#         mtb_dev_dl = torch.utils.data.DataLoader(dataset_dev.prepered_data, batch_size=batch_size)
#
#         criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
#         criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
#         criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
#
#         # call your training function and get the dev accuracy
#         (best_model_details,
#          _,
#          epochs_loss_train_values,
#          steps_loss_train_values,
#          loss_dev_values,
#          accuracy_dev_values) = training(model_DM,
#                                          mtb_train_dl,
#                                          mtb_dev_dl,
#                                          criterion_nikud,
#                                          criterion_dagesh,
#                                          criterion_sin,
#                                          training_params,
#                                          logger,
#                                          output_dir_running,
#                                          optimizer,
#                                          device=DEVICE)
#
#         # if these hyperparameters are better, store them
#         if accuracy_dev_values["all_nikud_letter"] > best_accuracy:
#             best_accuracy = accuracy_dev_values["all_nikud_letter"]
#             best_hyperparameters = (lr, batch_size)
#
#     # print the best hyperparameters
#     print(best_hyperparameters)
#

def evaluate_text(path, model_DM, tokenizer_tavbert, logger, plots_folder, batch_size=BATCH_SIZE):
    path_name = os.path.basename(path)

    msg = f"evaluate text: {path_name} on D-nikud Model"
    logger.debug(msg)

    if os.path.isfile(path):
        dataset = NikudDataset(tokenizer_tavbert, file=path, logger=logger, max_length=MAX_LENGTH_SEN, is_train=True)
    elif os.path.isdir(path):
        dataset = NikudDataset(tokenizer_tavbert, folder=path, logger=logger, max_length=MAX_LENGTH_SEN,
                               is_train=True)
    else:
        raise Exception("input path doesnt exist")

    dataset.prepare_data(name="evaluate")
    mtb_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=batch_size)

    word_level_correct, letter_level_correct_dev = evaluate(model_DM, mtb_dl, plots_folder, device=DEVICE)

    msg = f"Dnikud Model\n{path_name} evaluate\nLetter level accuracy:{letter_level_correct_dev}\n" \
          f"Word level accuracy: {word_level_correct}"
    logger.debug(msg)


def predict_text(text_file, tokenizer_tavbert, output_file, logger, model_DM, compare_nakdimon=False):
    dataset = NikudDataset(tokenizer_tavbert, file=text_file, logger=logger, max_length=MAX_LENGTH_SEN)

    dataset.prepare_data(name="prediction")
    mtb_prediction_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=BATCH_SIZE)
    all_labels = predict(model_DM, mtb_prediction_dl, DEVICE)
    text_data_with_labels = dataset.back_2_text(labels=all_labels)

    if output_file is None:
        for line in text_data_with_labels:
            print(line)
    else:
        with open(output_file, "w", encoding='utf-8') as f:
            if compare_nakdimon:
                f.write(extract_text_to_compare_nakdimon(text_data_with_labels))
            else:
                f.write(text_data_with_labels)


def predict_folder(folder, output_folder, logger, tokenizer_tavbert, model_DM, compare_nakdimon=False):
    create_missing_folders(output_folder)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if filename.lower().endswith('.txt') and os.path.isfile(file_path):
            output_file = os.path.join(output_folder, filename)
            predict_text(file_path,
                         output_file=output_file,
                         logger=logger,
                         tokenizer_tavbert=tokenizer_tavbert,
                         model_DM=model_DM, compare_nakdimon=compare_nakdimon)
        elif os.path.isdir(file_path) and filename != ".git":
            sub_folder = file_path
            sub_folder_output = os.path.join(output_folder, filename)
            predict_folder(sub_folder, sub_folder_output, logger, tokenizer_tavbert, model_DM,
                           compare_nakdimon=compare_nakdimon)


def update_compare_folder(folder, output_folder):
    create_missing_folders(output_folder)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if filename.lower().endswith('.txt') and os.path.isfile(file_path):
            output_file = os.path.join(output_folder, filename)
            with open(file_path, "r", encoding='utf-8') as f:
                text_data_with_labels = f.read()
            with open(output_file, "w", encoding='utf-8') as f:
                f.write(extract_text_to_compare_nakdimon(text_data_with_labels))
        elif os.path.isdir(file_path) and filename != ".git":
            sub_folder = file_path
            sub_folder_output = os.path.join(output_folder, filename)
            update_compare_folder(sub_folder, sub_folder_output)


def check_files_excepted(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if filename.lower().endswith('.txt') and os.path.isfile(file_path):
            try:
                x = NikudDataset(None, file=file_path)
            except:
                print(f"failed in file: {filename}")
        elif os.path.isdir(file_path) and filename != ".git":
            check_files_excepted(file_path)


def do_predict(input_path, output_path, tokenizer_tavbert, logger, model_DM, compare_nakdimon):
    if os.path.isdir(input_path):
        predict_folder(input_path, output_path, logger, tokenizer_tavbert, model_DM, compare_nakdimon=compare_nakdimon)
    elif os.path.isfile(input_path):
        predict_text(input_path,
                     output_file=output_path,
                     logger=logger,
                     tokenizer_tavbert=tokenizer_tavbert,
                     model_DM=model_DM, compare_nakdimon=compare_nakdimon)
    else:
        raise Exception("Input file not exist")


def do_evaluate(input_path, logger, model_DM, tokenizer_tavbert, plots_folder):
    msg = f'evaluate all_data: {input_path}'
    logger.info(msg)

    evaluate_text(input_path,
                  model_DM=model_DM,
                  tokenizer_tavbert=tokenizer_tavbert,
                  logger=logger,
                  plots_folder=plots_folder,
                  batch_size=BATCH_SIZE)

    msg = f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n'
    logger.info(msg)

    for sub_folder_name in os.listdir(input_path):
        sub_folder = os.path.join(input_path, sub_folder_name)

        if not os.path.isdir(sub_folder) or sub_folder == ".git":
            continue

        msg = f'evaluate sub folder: {sub_folder}'
        logger.info(msg)

        evaluate_text(sub_folder,
                      model_DM=model_DM,
                      tokenizer_tavbert=tokenizer_tavbert,
                      logger=logger,
                      plots_folder=plots_folder,
                      batch_size=BATCH_SIZE)

        msg = f'\n***************************************\n'
        logger.info(msg)

        folders = get_sub_folders_paths(sub_folder)
        for folder in folders:
            msg = f'evaluate sub folder: {folder}'
            logger.info(msg)

            evaluate_text(folder,
                          model_DM=model_DM,
                          tokenizer_tavbert=tokenizer_tavbert,
                          logger=logger,
                          plots_folder=plots_folder,
                          batch_size=args.batch_size)

            msg = f'\n---------------------------------------\n'
            logger.info(msg)


def do_train(logger, plots_folder, dir_model_config, tokenizer_tavbert, model_DM, output_trained_model_dir, data_folder,
             n_epochs, checkpoints_frequency, learning_rate, batch_size):
    msg = 'Loading data...'
    logger.debug(msg)

    dataset_train = NikudDataset(tokenizer_tavbert,
                                 folder=os.path.join(data_folder, "train"),
                                 logger=logger,
                                 max_length=MAX_LENGTH_SEN,
                                 is_train=True)
    dataset_dev = NikudDataset(tokenizer=tokenizer_tavbert,
                               folder=os.path.join(data_folder, "dev"),
                               logger=logger,
                               max_length=dataset_train.max_length,
                               is_train=True)
    dataset_test = NikudDataset(tokenizer=tokenizer_tavbert,
                                folder=os.path.join(data_folder, "test"),
                                logger=logger,
                                max_length=dataset_train.max_length,
                                is_train=True)

    dataset_train.show_data_labels(plots_folder=plots_folder)

    msg = f'Max length of data: {dataset_train.max_length}'
    logger.debug(msg)

    msg = f'Num rows in train data: {len(dataset_train.data)}, ' \
          f'Num rows in dev data: {len(dataset_dev.data)}, ' \
          f'Num rows in test data: {len(dataset_test.data)}'
    logger.debug(msg)

    msg = 'Loading tokenizer and prepare data...'
    logger.debug(msg)

    dataset_train.prepare_data(name="train")
    dataset_dev.prepare_data(name="dev")
    dataset_test.prepare_data(name="test")

    mtb_train_dl = torch.utils.data.DataLoader(dataset_train.prepered_data, batch_size=batch_size)
    mtb_dev_dl = torch.utils.data.DataLoader(dataset_dev.prepered_data, batch_size=batch_size)

    if not os.path.isfile(dir_model_config):
        our_model_config = ModelConfig(dataset_train.max_length)
        our_model_config.save_to_file(dir_model_config)

    optimizer = torch.optim.Adam(model_DM.parameters(), lr=learning_rate)

    msg = 'training...'
    logger.debug(msg)

    criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
    criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
    criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)

    training_params = {"n_epochs": n_epochs, "checkpoints_frequency": checkpoints_frequency}
    (best_model_details, best_accuracy, epochs_loss_train_values, steps_loss_train_values, loss_dev_values,
     accuracy_dev_values) = training(
        model_DM,
        mtb_train_dl,
        mtb_dev_dl,
        criterion_nikud,
        criterion_dagesh,
        criterion_sin,
        training_params,
        logger,
        output_trained_model_dir,
        optimizer,
        device=DEVICE
    )

    generate_plot_by_nikud_dagesh_sin_dict(epochs_loss_train_values, "Train epochs loss", "Loss", plots_folder)
    generate_plot_by_nikud_dagesh_sin_dict(steps_loss_train_values, "Train steps loss", "Loss", plots_folder)
    generate_plot_by_nikud_dagesh_sin_dict(loss_dev_values, "Dev epochs loss", "Loss", plots_folder)
    generate_plot_by_nikud_dagesh_sin_dict(accuracy_dev_values, "Dev accuracy", "Accuracy", plots_folder)
    generate_word_and_letter_accuracy_plot(accuracy_dev_values, plots_folder)

    msg = 'Done'
    logger.info(msg)


if __name__ == '__main__':
    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Predict D-nikud""")
    parser.add_argument("-l", "--log", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default="DEBUG", help="Set the logging level")
    parser.add_argument("-m", "--output_model_dir", type=str, default='models', help='Save directory for model')
    subparsers = parser.add_subparsers(help='sub-command help', dest="command", required=True)

    parser_predict = subparsers.add_parser('predict', help='diacritize a text files ')
    parser_predict.add_argument('input_path', help='input file or folder')
    parser_predict.add_argument('output_path', help='output file')
    parser_predict.add_argument("-c", "--compare", dest="compare_nakdimon",
                                default="False", help="predict text for comparing with Nakdimon")
    parser_predict.set_defaults(func=do_predict)

    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate D-nikud')
    parser_evaluate.add_argument('input_path', help='input file or folder')
    parser_evaluate.add_argument("-df", "--plots_folder", dest="plots_folder",
                        default=os.path.join(Path(__file__).parent, "plots"), help="Set the debug folder")
    parser_evaluate.set_defaults(func=do_evaluate)

    parser_train = subparsers.add_parser('train', help='train D-nikud')
    parser_train.add_argument('--from_pretrain', default=False, help='continue from pretrained')
    parser_train.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser_train.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser_train.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser_train.add_argument("--data_folder", dest="data_folder",
                        default=os.path.join(Path(__file__).parent, "data"), help="Set the debug folder")
    parser_train.add_argument('--checkpoints_frequency', type=int, default=1,
                        help='checkpoints frequency for save the model')
    parser_train.add_argument("-df", "--plots_folder", dest="plots_folder",
                        default=os.path.join(Path(__file__).parent, "plots"), help="Set the debug folder")
    parser_train.set_defaults(func=do_train)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    logger = get_logger(kwargs["log_level"], args.command, date_time)

    del kwargs['log_level']

    kwargs['tokenizer_tavbert'] = tokenizer_tavbert
    kwargs['logger'] = logger

    msg = 'Loading model...'
    logger.debug(msg)

    if args.command in ["evaluate", "predict"] or (args.command == "train" and args.from_pretrain):
        dir_model_config = os.path.join(kwargs["output_model_dir"], "config.yml")
        config = ModelConfig.load_from_file(dir_model_config)

        model_DM = DNikudModel(config, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
                               len(Nikud.label_2_id["sin"]), device=DEVICE).to(DEVICE)
        state_dict_model = model_DM.state_dict()
        state_dict_model.update(torch.load(BEST_MODEL_PATH))
        model_DM.load_state_dict(state_dict_model)

    else:
        base_model_name = "tau/tavbert-he"
        config = AutoConfig.from_pretrained(base_model_name)
        model_DM = DNikudModel(config,
                               len(Nikud.label_2_id["nikud"]),
                               len(Nikud.label_2_id["dagesh"]),
                               len(Nikud.label_2_id["sin"]),
                               pretrain_model=base_model_name,
                               device=DEVICE
                               ).to(DEVICE)

    if args.command == "train":
        output_trained_model_dir = os.path.join(kwargs['output_model_dir'], "latest", f"output_models_{date_time}")
        create_missing_folders(output_trained_model_dir)
        dir_model_config = os.path.join(kwargs['output_model_dir'], "config.yml")
        kwargs['dir_model_config'] = dir_model_config
        kwargs['output_trained_model_dir'] = output_trained_model_dir
        del kwargs['from_pretrain']
    del kwargs['output_model_dir']
    kwargs['model_DM'] = model_DM

    del kwargs['command']
    del kwargs['func']
    args.func(**kwargs)

    sys.exit(0)

    # evaluate  "C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\female2\Dnikud_v44"
    # folder = r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\test"
    # # data = []
    # for sub_folder in os.listdir(folder):
    #     # if sub_folder != "law":
    #     #     continue
    #     print(sub_folder)
    #     sub_folder_path = os.path.join(folder, sub_folder)
    #     # num_files, num_letters = info_folder(sub_folder_path, 0, 0)
    #     evaluate_text(sub_folder_path)
    #     # data.append(sub_data)
    # # print(data)

    # predict "C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\female2\expected" "C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\female2\Dnikud_v4"
    # orgenize_data(main_folder=r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized")
    # evaluate_text(r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\WikipediaHebrewWithVocalization.txt")
    # predict_text(
    #     r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\WikipediaHebrewWithVocalization-WithMetegToMarkMatresLectionis.txt")
    # train(use_pretrain=False)
    # hyperparams_checker()
    # test_by_folders(main_folder=r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\test_modern")
    # test_by_folders(main_folder=r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\test")
    # test_by_folders(
    #     main_folder=r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized\male_female\male_not_use")
    # predict_folder_flow(r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized\dicta\male",
    #                     output_folder=r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized\dicta\male_nakdimon")
    # predict_folder_flow(r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\new\expected",
    #                     output_folder=r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\new\Dnikud_v8",
    #                     compare_nakdimon=True)
    # update_compare_folder(r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\new\Dnikud_v8",
    #                     output_folder=r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\new\Dnikud_v82")
    # check_files_excepted(r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data")
    # check_files_excepted(r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\haser\expected\haser")
# predict "C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\female2\expected" "C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\female2\Dnikud_v44"