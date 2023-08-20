# general
import argparse
import os
import random
import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re

# ML
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

# DL
from src.models import DNikudModel, ModelConfig
from src.models_utils import training, evaluate, predict
from src.plot_helpers import plot_results, generate_plot_by_nikud_dagesh_sin_dict, \
    generate_word_and_letter_accuracy_plot
from src.running_params import SEED, BEST_MODEL_PATH, BATCH_SIZE, MAX_LENGTH_SEN
from src.utiles_data import NikudDataset, Nikud, Letters, get_sub_folders_paths, create_folder_if_not_exist

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda'
cols = ["precision", "recall", "f1-score", "support"]

# TODO: DELETE SEEDS

# Set the random seed for Python
random.seed(SEED)

# Set the random seed for numpy
np.random.seed(SEED)

# Set the random seed for torch to SEED
torch.manual_seed(SEED)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_model_dir', type=str, default='models', help='Save directory for model')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (train)')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Validation batch size')
    parser.add_argument('--save_strategy', '-lr', type=str, default='no',
                        help='Whether to save on every epoch ("epoch"/"no")')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help='Learning rate scheduler type ("linear"/"cosine"/"constant"/...')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--checkpoints_frequency', type=int, default=1, help='checkpoints frequency for save the model')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        help='How to validate (set to "no" for no validation)')
    parser.add_argument('--eval_steps', type=int, default=2000, help='Validate every N steps')
    parser.add_argument("-l", "--log", dest="loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default="DEBUG", help="Set the logging level")
    parser.add_argument("-ld", "--log_dir", dest="log_dir",
                        default=os.path.join(Path(__file__).parent, "logging"), help="Set the logger path")
    parser.add_argument("-df", "--debug_folder", dest="debug_folder",
                        default=os.path.join(Path(__file__).parent, "plots"), help="Set the debug folder")
    parser.add_argument("-dataf", "--data_folder", dest="data_folder",
                        default=os.path.join(Path(__file__).parent, "data"),
                        help="Set the debug folder")
    return parser.parse_args()


def generate_folders(args, name_log):
    output_model_dir = args.output_model_dir

    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')

    create_folder_if_not_exist(args.debug_folder)

    debug_folder = os.path.join(args.debug_folder, f"debug_plots_{date_time}")
    create_folder_if_not_exist(debug_folder)

    output_dir_running = os.path.join(output_model_dir, "trained", "latest", f"output_models_{date_time}")
    create_folder_if_not_exist(output_dir_running)

    output_log_dir = os.path.join(args.log_dir, f"{name_log}_{date_time}")
    create_folder_if_not_exist(output_log_dir)

    return output_model_dir, output_log_dir, output_dir_running, debug_folder


def train(use_pretrain=False):
    args = parse_arguments()

    output_model_dir, output_log_dir, output_dir_running, debug_folder = generate_folders(args,
                                                                                          name_log=f"log_model_lr_{args.learning_rate}_bs_{BATCH_SIZE}")

    logger = get_logger(args.loglevel, output_log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    msg = f'Device detected: {device}'
    logger.info(msg)

    msg = 'Loading data...'
    logger.debug(msg)

    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    dataset_train = NikudDataset(tokenizer_tavbert,
                                 folder=os.path.join(args.data_folder, "train"),
                                 logger=logger,
                                 max_length=MAX_LENGTH_SEN,
                                 is_train=True)
    dataset_dev = NikudDataset(tokenizer=tokenizer_tavbert,
                               folder=os.path.join(args.data_folder, "dev"),
                               logger=logger,
                               max_length=dataset_train.max_length,
                               is_train=True)
    dataset_test = NikudDataset(tokenizer=tokenizer_tavbert,
                                folder=os.path.join(args.data_folder, "test"),
                                logger=logger,
                                max_length=dataset_train.max_length,
                                is_train=True)

    dataset_train.show_data_labels(debug_folder=debug_folder)

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

    mtb_train_dl = torch.utils.data.DataLoader(dataset_train.prepered_data, batch_size=BATCH_SIZE)
    mtb_dev_dl = torch.utils.data.DataLoader(dataset_dev.prepered_data, batch_size=BATCH_SIZE)
    mtb_test_dl = torch.utils.data.DataLoader(dataset_test.prepered_data, batch_size=BATCH_SIZE)

    msg = 'Loading model...'
    logger.debug(msg)

    base_model_name = "tau/tavbert-he"
    config = AutoConfig.from_pretrained(base_model_name)
    model_DM = DNikudModel(config,
                           len(Nikud.label_2_id["nikud"]),
                           len(Nikud.label_2_id["dagesh"]),
                           len(Nikud.label_2_id["sin"]),
                           pretrain_model=base_model_name,
                           device=DEVICE
                           ).to(DEVICE)

    if use_pretrain:
        # load last best model:
        state_dict_model = model_DM.state_dict()
        state_dict_model.update(
            torch.load(BEST_MODEL_PATH))
        model_DM.load_state_dict(state_dict_model)

    dir_model_config = os.path.join(output_model_dir, "config.yml")

    if not os.path.isfile(dir_model_config):
        our_model_config = ModelConfig(dataset_train.max_length)
        our_model_config.save_to_file(dir_model_config)

    optimizer = torch.optim.Adam(model_DM.parameters(), lr=args.learning_rate)

    msg = 'training...'
    logger.debug(msg)

    criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
    criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
    criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)

    training_params = {"n_epochs": args.n_epochs, "checkpoints_frequency": args.checkpoints_frequency}
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
        output_dir_running,
        optimizer,
        device=DEVICE
    )

    generate_plot_by_nikud_dagesh_sin_dict(epochs_loss_train_values, "Train epochs loss", "Loss", debug_folder)
    generate_plot_by_nikud_dagesh_sin_dict(steps_loss_train_values, "Train steps loss", "Loss", debug_folder)
    generate_plot_by_nikud_dagesh_sin_dict(loss_dev_values, "Dev epochs loss", "Loss", debug_folder)
    generate_plot_by_nikud_dagesh_sin_dict(accuracy_dev_values, "Dev accuracy", "Accuracy", debug_folder)
    generate_word_and_letter_accuracy_plot(accuracy_dev_values, debug_folder)

    model_DM.load_state_dict(best_model_details['model_state_dict'])

    report_dev, word_level_correct_dev, letter_level_correct_dev = evaluate(model_DM, mtb_dev_dl, debug_folder,
                                                                            device=DEVICE)
    report_test, word_level_correct_test, letter_level_correct_test = evaluate(model_DM, mtb_test_dl, debug_folder,
                                                                               device=DEVICE)

    msg = f"Diacritization Model\nDev dataset\nLetter level accuracy:{letter_level_correct_dev}\n" \
          f"Word level accuracy: {word_level_correct_dev}\n--------------------\nTest dataset\n" \
          f"Letter level accuracy: {letter_level_correct_test}\nWord level accuracy: {word_level_correct_test}"
    logger.debug(msg)

    plot_results(logger, report_dev, report_filename="results_dev")
    plot_results(logger, report_test, report_filename="results_test")

    msg = 'Done'
    logger.info(msg)


def get_logger(log_level, log_location):
    log_format = '%(asctime)s %(levelname)-8s Thread_%(thread)-6d ::: %(funcName)s(%(lineno)d) ::: %(message)s'
    logger = logging.getLogger("algo")
    logger.setLevel(getattr(logging, log_level))
    cnsl_log_formatter = logging.Formatter(log_format)
    cnsl_handler = logging.StreamHandler()
    cnsl_handler.setFormatter(cnsl_log_formatter)
    cnsl_handler.setLevel(log_level)
    logger.addHandler(cnsl_handler)

    create_folder_if_not_exist(log_location)

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


def hyperparams_checker(use_pretrain=False):
    args = parse_arguments()
    debug_folder = args.debug_folder
    create_folder_if_not_exist(debug_folder)

    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    dataset_train = NikudDataset(tokenizer_tavbert, folder=os.path.join(args.data_folder, "train"), logger=None,
                                 max_length=MAX_LENGTH_SEN, is_train=True)
    dataset_dev = NikudDataset(tokenizer=tokenizer_tavbert, folder=os.path.join(args.data_folder, "dev"), logger=None,
                               max_length=dataset_train.max_length, is_train=True)
    dataset_test = NikudDataset(tokenizer=tokenizer_tavbert, folder=os.path.join(args.data_folder, "test"),
                                logger=None, max_length=dataset_train.max_length, is_train=True)

    dataset_train.prepare_data(name="train")
    dataset_dev.prepare_data(name="dev")
    dataset_test.prepare_data(name="test")

    # hyperparameters search space
    lr_values = np.logspace(-6, -1, num=6)  # learning rates between 1e-6 and 1e-1
    num_freeze_layers = list(range(1, 10, 2))  # learning rates between 1e-6 and 1e-1
    batch_size_values = [2 ** i for i in range(3, 7)]  # batch sizes between 32 and 512

    # number of random combinations to test
    num_combinations = 20

    # best hyperparameters and their performance
    best_accuracy = 0.0
    best_hyperparameters = None

    training_params = {"n_epochs": args.n_epochs, "checkpoints_frequency": args.checkpoints_frequency}

    for _ in range(num_combinations):
        torch.cuda.empty_cache()
        lr = np.random.choice(lr_values)
        nfl = np.random.choice(num_freeze_layers)
        batch_size = int(np.random.choice(batch_size_values))

        output_model_dir, output_log_dir, output_dir_running, debug_folder = generate_folders(args,
                                                                                              name_log=f"log_model_lr_{lr}_bs_{batch_size}_nfl_{nfl}")
        logger = get_logger(args.loglevel, output_log_dir)

        msg = 'Loading model...'
        logger.debug(msg)

        base_model_name = "tau/tavbert-he"
        config = AutoConfig.from_pretrained(base_model_name)

        model_DM = DNikudModel(config,
                               len(Nikud.label_2_id["nikud"]),
                               len(Nikud.label_2_id["dagesh"]),
                               len(Nikud.label_2_id["sin"]),
                               device=DEVICE
                               ).to(DEVICE)
        if use_pretrain:
            # load last best model:
            state_dict_model = model_DM.state_dict()
            state_dict_model.update(
                torch.load(BEST_MODEL_PATH))
            model_DM.load_state_dict(state_dict_model)

        # set these hyperparameters in your optimizer
        optimizer = torch.optim.Adam(model_DM.parameters(), lr=args.learning_rate)

        # redefine your data loaders with the new batch size
        mtb_train_dl = torch.utils.data.DataLoader(dataset_train.prepered_data, batch_size=batch_size)
        mtb_dev_dl = torch.utils.data.DataLoader(dataset_dev.prepered_data, batch_size=batch_size)

        criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
        criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
        criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)

        # call your training function and get the dev accuracy
        (best_model_details,
         _,
         epochs_loss_train_values,
         steps_loss_train_values,
         loss_dev_values,
         accuracy_dev_values) = training(model_DM,
                                         mtb_train_dl,
                                         mtb_dev_dl,
                                         criterion_nikud,
                                         criterion_dagesh,
                                         criterion_sin,
                                         training_params,
                                         logger,
                                         output_dir_running,
                                         optimizer,
                                         device=DEVICE)

        # if these hyperparameters are better, store them
        if accuracy_dev_values["all_nikud_letter"] > best_accuracy:
            best_accuracy = accuracy_dev_values["all_nikud_letter"]
            best_hyperparameters = (lr, batch_size)

    # print the best hyperparameters
    print(best_hyperparameters)


def evaluate_text(path, model_DM=None, tokenizer_tavbert=None, logger=None, batch_size=32, config=None):
    path_name = os.path.basename(path)
    args = parse_arguments()
    if logger is None:
        date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
        output_log_dir = os.path.join(args.log_dir,
                                      f"evaluate_{path_name}_{date_time}")

        create_folder_if_not_exist(output_log_dir)
        logger = get_logger(args.loglevel, output_log_dir)

    msg = f"evaluate text: {path_name} on D-nikud Model"
    logger.debug(msg)

    if model_DM is None:
        dir_model_config = os.path.join(args.output_model_dir, "config.yml")
        config = ModelConfig.load_from_file(dir_model_config)
        model_DM = DNikudModel(config,
                               len(Nikud.label_2_id["nikud"]),
                               len(Nikud.label_2_id["dagesh"]),
                               len(Nikud.label_2_id["sin"]),
                               device=DEVICE
                               ).to(DEVICE)
        state_dict_model = model_DM.state_dict()
        state_dict_model.update(
            torch.load(BEST_MODEL_PATH))
        model_DM.load_state_dict(state_dict_model)

    if tokenizer_tavbert is None:
        tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    if os.path.isfile(path):
        dataset = NikudDataset(tokenizer_tavbert, file=path, logger=logger, max_length=config.max_length, is_train=True)
    elif os.path.isdir(path):
        dataset = NikudDataset(tokenizer_tavbert, folder=path, logger=logger, max_length=config.max_length,
                               is_train=True)

    dataset.prepare_data(name="evaluate")
    mtb_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=batch_size)

    report, word_level_correct, letter_level_correct_dev = evaluate(model_DM, mtb_dl, device=DEVICE)

    msg = f"Dnikud Model\n{path_name} evaluate\nLetter level accuracy:{letter_level_correct_dev}\n" \
          f"Word level accuracy: {word_level_correct}"
    logger.debug(msg)


def extract_text_to_compare_nakdimon(text):
    res = text.replace('|', '')
    res = res.replace(chr(Nikud.nikud_dict["KUBUTZ"]) + 'ו' + chr(Nikud.nikud_dict["METEG"]),
                      'ו' + chr(Nikud.nikud_dict['DAGESH OR SHURUK']))
    res = res.replace(chr(Nikud.nikud_dict["HOLAM"]) + 'ו' + chr(Nikud.nikud_dict["METEG"]),
                      'ו')
    res = res.replace("ו" + chr(Nikud.nikud_dict["HOLAM"]) + chr(Nikud.nikud_dict["KAMATZ"]),
                      'ו' + chr(Nikud.nikud_dict["KAMATZ"]))
    res = res.replace(chr(Nikud.nikud_dict["METEG"]), '')
    res = res.replace(chr(Nikud.nikud_dict["KAMATZ"]) + chr(Nikud.nikud_dict["HIRIK"]),
                      chr(Nikud.nikud_dict["KAMATZ"]) + 'י' + chr(Nikud.nikud_dict["HIRIK"]))
    res = res.replace(chr(Nikud.nikud_dict["PATAKH"]) + chr(Nikud.nikud_dict["HIRIK"]),
                      chr(Nikud.nikud_dict["PATAKH"]) + 'י' + chr(Nikud.nikud_dict["HIRIK"]))
    res = res.replace(chr(Nikud.nikud_dict["PUNCTUATION MAQAF"]), '')
    res = res.replace(chr(Nikud.nikud_dict["PUNCTUATION PASEQ"]), '')
    res = res.replace(chr(Nikud.nikud_dict["KAMATZ_KATAN"]), chr(Nikud.nikud_dict["KAMATZ"]))

    res = re.sub(chr(Nikud.nikud_dict["KUBUTZ"]) + 'ו' + '(?=[א-ת])', 'ו',
                 res)
    res = res.replace(chr(Nikud.nikud_dict["REDUCED_KAMATZ"]) + 'ו', 'ו')

    res = res.replace(chr(Nikud.nikud_dict["DAGESH OR SHURUK"]) * 2, chr(Nikud.nikud_dict["DAGESH OR SHURUK"]))
    res = res.replace('\u05be', '-')
    res = res.replace('יְהוָֹה', 'יהוה')

    return res


def predict_text(text_file, tokenizer_tavbert=None, output_file=None, logger=None, model_DM=None):
    dir_model_config = "models/config.yml"
    config = ModelConfig.load_from_file(dir_model_config)

    if tokenizer_tavbert is None:
        tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")
    if logger is None:
        args = parse_arguments()
        date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
        output_log_dir = os.path.join(args.log_dir, f"log_model_predict_{date_time}")

        create_folder_if_not_exist(output_log_dir)
        logger = get_logger(args.loglevel, output_log_dir)

    dataset = NikudDataset(tokenizer_tavbert,
                           file=text_file,
                           logger=logger, max_length=MAX_LENGTH_SEN)

    if model_DM is None:
        model_DM = DNikudModel(config,
                               len(Nikud.label_2_id["nikud"]),
                               len(Nikud.label_2_id["dagesh"]),
                               len(Nikud.label_2_id["sin"]),
                               device=DEVICE
                               ).to(DEVICE)

        state_dict_model = model_DM.state_dict()
        state_dict_model.update(torch.load(BEST_MODEL_PATH))
        model_DM.load_state_dict(state_dict_model)

    dataset.prepare_data(name="prediction")
    mtb_prediction_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=BATCH_SIZE)
    all_labels = predict(model_DM, mtb_prediction_dl, DEVICE)
    text_data_with_labels = dataset.back_2_text(labels=all_labels)

    if output_file is None:
        for line in text_data_with_labels:
            print(line)
    else:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(extract_text_to_compare_nakdimon(text_data_with_labels))


# TODO DELETE
def organize_data(main_folder):
    args = parse_arguments()
    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    output_log_dir = os.path.join(args.log_dir, f"log_orgenize_data_{date_time}")

    create_folder_if_not_exist(output_log_dir)
    logger = get_logger(args.loglevel, output_log_dir)

    x = NikudDataset(None)
    x.delete_files(os.path.join(Path(main_folder).parent, "train"))
    x.delete_files(os.path.join(Path(main_folder).parent, "dev"))
    x.delete_files(os.path.join(Path(main_folder).parent, "test"))
    x.split_data(main_folder, main_folder_name=os.path.basename(main_folder), logger=logger)


def test_by_folders(main_folder):
    args = parse_arguments()

    dir_model_config = os.path.join(args.output_model_dir, "config.yml")
    config = ModelConfig.load_from_file(dir_model_config)

    model_DM = DNikudModel(config,
                           len(Nikud.label_2_id["nikud"]),
                           len(Nikud.label_2_id["dagesh"]),
                           len(Nikud.label_2_id["sin"]),
                           device=DEVICE
                           ).to(DEVICE)

    state_dict_model = model_DM.state_dict()
    state_dict_model.update(torch.load(BEST_MODEL_PATH))
    model_DM.load_state_dict(state_dict_model)
    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    output_log_dir = os.path.join(args.log_dir, f"evaluate_{os.path.basename(main_folder)}_{date_time}")
    create_folder_if_not_exist(output_log_dir)
    logger = get_logger(args.loglevel, output_log_dir)

    msg = f'evaluate all_data: {main_folder}'
    logger.info(msg)

    evaluate_text(main_folder,
                  model_DM=model_DM,
                  tokenizer_tavbert=tokenizer_tavbert,
                  logger=logger,
                  batch_size=args.batch_size,
                  config=config)

    msg = f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n'
    logger.info(msg)

    for sub_folder_name in os.listdir(main_folder):
        sub_folder = os.path.join(main_folder, sub_folder_name)

        if not os.path.isdir(sub_folder) or sub_folder == ".git":
            continue

        msg = f'evaluate sub folder: {sub_folder}'
        logger.info(msg)

        evaluate_text(sub_folder,
                      model_DM=model_DM,
                      tokenizer_tavbert=tokenizer_tavbert,
                      logger=logger,
                      batch_size=args.batch_size,
                      config=config)

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
                          batch_size=args.batch_size,
                          config=config)

            msg = f'\n---------------------------------------\n'
            logger.info(msg)


def predict_folder_flow(folder, output_folder):
    args = parse_arguments()

    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    output_log_dir = os.path.join(args.log_dir, f"log_orgenize_data_{date_time}")
    create_folder_if_not_exist(output_log_dir)
    logger = get_logger(args.loglevel, output_log_dir)

    msg = f"prepare data in folder - {os.path.basename(folder)}"
    logger.debug(msg)

    start_time = time.time()

    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")
    dir_model_config = os.path.join(args.output_model_dir, "config.yml")
    config = ModelConfig.load_from_file(dir_model_config)
    model_DM = DNikudModel(config,
                           len(Nikud.label_2_id["nikud"]),
                           len(Nikud.label_2_id["dagesh"]),
                           len(Nikud.label_2_id["sin"]),
                           device=DEVICE
                           ).to(DEVICE)

    state_dict_model = model_DM.state_dict()
    state_dict_model.update(
        torch.load(BEST_MODEL_PATH))
    model_DM.load_state_dict(state_dict_model)
    predict_folder(folder, output_folder, logger, tokenizer_tavbert, model_DM)

    end_time = time.time()

    elapsed_time = end_time - start_time

    msg = f"dnikud predict took {elapsed_time} seconds to run."
    logger.debug(msg)


def predict_folder(folder, output_folder, logger, tokenizer_tavbert, model_DM):
    create_folder_if_not_exist(output_folder)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.lower().endswith('.txt') and os.path.isfile(file_path):
            output_file = os.path.join(output_folder, filename)
            predict_text(file_path,
                         output_file=output_file,
                         logger=logger,
                         tokenizer_tavbert=tokenizer_tavbert,
                         model_DM=model_DM)
        elif os.path.isdir(file_path) and filename != ".git":
            sub_folder = file_path
            sub_folder_output = os.path.join(output_folder, filename)
            predict_folder(sub_folder, sub_folder_output, logger, tokenizer_tavbert, model_DM)


def update_compare_folder(folder, output_folder):
    create_folder_if_not_exist(output_folder)

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


def info_folder(folder, num_files, num_hebrew_letters):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.lower().endswith('.txt') and os.path.isfile(file_path):
            num_files += 1
            dataset = NikudDataset(None, file=file_path)
            for line in dataset.data:
                for c in line[0]:
                    if c in Letters.hebrew:
                        num_hebrew_letters += 1

        elif os.path.isdir(file_path) and filename != ".git":
            sub_folder = file_path
            n1, n2 = info_folder(sub_folder, num_files, num_hebrew_letters)
            num_files += n1
            num_hebrew_letters += n2
    return num_files, num_hebrew_letters


def do_predict(input_path, output_path, log_level="DEBUG"):
    dir_model_config = "models/config.yml"
    config = ModelConfig.load_from_file(dir_model_config)

    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    output_log_dir = os.path.join(os.path.join(Path(__file__).parent, "logging"), f"log_model_predict_{date_time}")
    create_folder_if_not_exist(output_log_dir)
    logger = get_logger(log_level, output_log_dir)

    model_DM = DNikudModel(config,
                           len(Nikud.label_2_id["nikud"]),
                           len(Nikud.label_2_id["dagesh"]),
                           len(Nikud.label_2_id["sin"]),
                           device=DEVICE
                           ).to(DEVICE)
    state_dict_model = model_DM.state_dict()
    state_dict_model.update(torch.load(BEST_MODEL_PATH))

    if os.path.isdir(input_path):
        predict_folder(input_path, output_path, logger, tokenizer_tavbert, model_DM)
    elif os.path.isfile(input_path):
        predict_text(input_path,
                     output_file=output_path,
                     logger=logger,
                     tokenizer_tavbert=tokenizer_tavbert,
                     model_DM=model_DM)
    else:
        raise Exception("Input file not exist")


if __name__ == '__main__':
    train(use_pretrain=False)
    # predict
    # "C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\test\law\law.txt" "C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\test\law\law.txt"
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    #     description="""Predict D-nikud""",
    # )
    # subparsers = parser.add_subparsers(help='sub-command help', dest="command", required=True)
    #
    # parser_predict = subparsers.add_parser('predict', help='diacritize a text file')
    # parser_predict.add_argument('input_path', help='input file')
    # parser_predict.add_argument('output_path', help='output file')
    # parser_predict.add_argument("-l", "--log", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    #                     default="DEBUG", help="Set the logging level")
    # parser_predict.set_defaults(func=do_predict)
    #
    #
    # args = parser.parse_args()
    #
    # kwargs = vars(args).copy()
    # del kwargs['command']
    # del kwargs['func']
    # args.func(**kwargs)
    #
    # sys.exit(0)

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
    # predict_folder_flow(r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\female\expected",
    #                     output_folder=r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\female\Dnikud_v6")
    # update_compare_folder(r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\dnikud_test\expected",
    #                     output_folder=r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\dnikud_test\expected2")
    # check_files_excepted(r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data")
    # check_files_excepted(r"C:\Users\adir\Desktop\studies\nlp\nakdimon\tests\haser\expected\haser")
