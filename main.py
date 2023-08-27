# general
import argparse
import os
import sys
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ML
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

# DL
from src.models import DNikudModel, ModelConfig
from src.models_utils import training, evaluate, predict
from src.plot_helpers import generate_plot_by_nikud_dagesh_sin_dict, \
    generate_word_and_letter_accuracy_plot
from src.running_params import BATCH_SIZE, MAX_LENGTH_SEN
from src.utiles_data import NikudDataset, Nikud, create_missing_folders, \
    extract_text_to_compare_nakdimon

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda'


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


def evaluate_text(path, dnikud_model, tokenizer_tavbert, logger, plots_folder=None, batch_size=BATCH_SIZE):
    path_name = os.path.basename(path)

    msg = f"evaluate text: {path_name} on D-nikud Model"
    logger.debug(msg)

    if os.path.isfile(path):
        dataset = NikudDataset(tokenizer_tavbert, file=path, logger=logger, max_length=MAX_LENGTH_SEN)
    elif os.path.isdir(path):
        dataset = NikudDataset(tokenizer_tavbert, folder=path, logger=logger, max_length=MAX_LENGTH_SEN)
    else:
        raise Exception("input path doesnt exist")

    dataset.prepare_data(name="evaluate")
    mtb_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=batch_size)

    word_level_correct, letter_level_correct_dev = evaluate(dnikud_model, mtb_dl, plots_folder, device=DEVICE)

    msg = f"Dnikud Model\n{path_name} evaluate\nLetter level accuracy:{letter_level_correct_dev}\n" \
          f"Word level accuracy: {word_level_correct}"
    logger.debug(msg)


def predict_text(text_file, tokenizer_tavbert, output_file, logger, dnikud_model, compare_nakdimon=False):
    dataset = NikudDataset(tokenizer_tavbert, file=text_file, logger=logger, max_length=MAX_LENGTH_SEN)

    dataset.prepare_data(name="prediction")
    mtb_prediction_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=BATCH_SIZE)
    all_labels = predict(dnikud_model, mtb_prediction_dl, DEVICE)
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


def predict_folder(folder, output_folder, logger, tokenizer_tavbert, dnikud_model, compare_nakdimon=False):
    create_missing_folders(output_folder)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if filename.lower().endswith('.txt') and os.path.isfile(file_path):
            output_file = os.path.join(output_folder, filename)
            predict_text(file_path,
                         output_file=output_file,
                         logger=logger,
                         tokenizer_tavbert=tokenizer_tavbert,
                         dnikud_model=dnikud_model, compare_nakdimon=compare_nakdimon)
        elif os.path.isdir(file_path) and filename != ".git" and filename != "README.md":
            sub_folder = file_path
            sub_folder_output = os.path.join(output_folder, filename)
            predict_folder(sub_folder, sub_folder_output, logger, tokenizer_tavbert, dnikud_model,
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


def do_predict(input_path, output_path, tokenizer_tavbert, logger, dnikud_model, compare_nakdimon):
    if os.path.isdir(input_path):
        predict_folder(input_path, output_path, logger, tokenizer_tavbert, dnikud_model,
                       compare_nakdimon=compare_nakdimon)
    elif os.path.isfile(input_path):
        predict_text(input_path,
                     output_file=output_path,
                     logger=logger,
                     tokenizer_tavbert=tokenizer_tavbert,
                     dnikud_model=dnikud_model, compare_nakdimon=compare_nakdimon)
    else:
        raise Exception("Input file not exist")


def evaluate_folder(folder_path, logger, dnikud_model, tokenizer_tavbert, plots_folder):
    msg = f'evaluate sub folder: {folder_path}'
    logger.info(msg)

    evaluate_text(folder_path,
                  dnikud_model=dnikud_model,
                  tokenizer_tavbert=tokenizer_tavbert,
                  logger=logger,
                  plots_folder=plots_folder,
                  batch_size=BATCH_SIZE)

    msg = f'\n***************************************\n'
    logger.info(msg)

    for sub_folder_name in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)

        if (not os.path.isdir(sub_folder_path)
                or sub_folder_path == ".git"
                or "not_use" in sub_folder_path
                or "NakdanResults" in sub_folder_path):
            continue

        evaluate_folder(sub_folder_path, logger, dnikud_model, tokenizer_tavbert, plots_folder)


def do_evaluate(input_path, logger, dnikud_model, tokenizer_tavbert, plots_folder, eval_sub_folders=False):
    msg = f'evaluate all_data: {input_path}'
    logger.info(msg)

    evaluate_text(input_path,
                  dnikud_model=dnikud_model,
                  tokenizer_tavbert=tokenizer_tavbert,
                  logger=logger,
                  plots_folder=plots_folder,
                  batch_size=BATCH_SIZE)

    msg = f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n'
    logger.info(msg)

    if eval_sub_folders:
        for sub_folder_name in os.listdir(input_path):
            sub_folder_path = os.path.join(input_path, sub_folder_name)

            if (not os.path.isdir(sub_folder_path)
                    or sub_folder_path == ".git"
                    or "not_use" in sub_folder_path
                    or "NakdanResults" in sub_folder_path):
                continue

            evaluate_folder(sub_folder_path, logger, dnikud_model, tokenizer_tavbert, plots_folder)


def do_train(logger, plots_folder, dir_model_config, tokenizer_tavbert, dnikud_model, output_trained_model_dir,
             data_folder, n_epochs, checkpoints_frequency, learning_rate, batch_size):
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

    optimizer = torch.optim.Adam(dnikud_model.parameters(), lr=learning_rate)

    msg = 'training...'
    logger.debug(msg)

    criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
    criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)
    criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD_OR_IRRELEVANT).to(DEVICE)

    training_params = {"n_epochs": n_epochs, "checkpoints_frequency": checkpoints_frequency}
    (best_model_details, best_accuracy, epochs_loss_train_values, steps_loss_train_values, loss_dev_values,
     accuracy_dev_values) = training(
        dnikud_model,
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
    generate_word_and_letter_accuracy_plot(accuracy_dev_values, "Accuracy", plots_folder)

    msg = 'Done'
    logger.info(msg)


if __name__ == '__main__':
    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Predict D-nikud""")
    parser.add_argument('-l', '--log', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='DEBUG', help='Set the logging level')
    parser.add_argument('-m', '--output_model_dir', type=str, default='models', help='save directory for model')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command', required=True)

    parser_predict = subparsers.add_parser('predict', help='diacritize a text files ')
    parser_predict.add_argument('input_path', help='input file or folder')
    parser_predict.add_argument('output_path', help='output file')
    parser_predict.add_argument('-ptmp', '--pretrain_model_path', type=str,
                                default=os.path.join(Path(__file__).parent, 'models', 'prod', 'Dnikud_best_model.pth'),
                                help='pre-train model path - use only if you want to use trained model weights')
    parser_predict.add_argument('-c', '--compare', dest='compare_nakdimon',
                                default=False, help='predict text for comparing with Nakdimon')
    parser_predict.set_defaults(func=do_predict)

    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate D-nikud')
    parser_evaluate.add_argument('input_path', help='input file or folder')
    parser_evaluate.add_argument('-ptmp', '--pretrain_model_path', type=str,
                                default=os.path.join(Path(__file__).parent, 'models', 'prod', 'Dnikud_best_model.pth'),
                                help='pre-train model path - use only if you want to use trained model weights')
    parser_evaluate.add_argument('-df', '--plots_folder', dest='plots_folder',
                                 default=os.path.join(Path(__file__).parent, 'plots'), help='set the debug folder')
    parser_evaluate.add_argument('-es', '--eval_sub_folders', dest='eval_sub_folders',
                                 default=False, help='accuracy calculation includes the evaluation of sub-folders '
                                                     'within the input_path folder, providing independent assessments '
                                                     'for each subfolder.')
    parser_evaluate.set_defaults(func=do_evaluate)

    # train --n_epochs 20

    parser_train = subparsers.add_parser('train', help='train D-nikud')
    parser_train.add_argument('-ptmp', '--pretrain_model_path', type=str,
                                default=None,
                                help='pre-train model path - use only if you want to use trained model weights')
    parser_train.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser_train.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser_train.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser_train.add_argument('--data_folder', dest='data_folder',
                              default=os.path.join(Path(__file__).parent, 'data'), help='Set the debug folder')
    parser_train.add_argument('--checkpoints_frequency', type=int, default=1,
                              help='checkpoints frequency for save the model')
    parser_train.add_argument('-df', '--plots_folder', dest='plots_folder',
                              default=os.path.join(Path(__file__).parent, 'plots'), help='Set the debug folder')
    parser_train.set_defaults(func=do_train)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    logger = get_logger(kwargs['log_level'], args.command, date_time)

    del kwargs['log_level']

    kwargs['tokenizer_tavbert'] = tokenizer_tavbert
    kwargs['logger'] = logger

    msg = 'Loading model...'
    logger.debug(msg)

    if args.command in ["evaluate", "predict"] or (args.command == "train" and args.pretrain_model_path is not None):
        dir_model_config = os.path.join("models", "config.yml")
        config = ModelConfig.load_from_file(dir_model_config)

        dnikud_model = DNikudModel(config, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
                                   len(Nikud.label_2_id["sin"]), device=DEVICE).to(DEVICE)
        state_dict_model = dnikud_model.state_dict()
        state_dict_model.update(torch.load(args.pretrain_model_path))
        dnikud_model.load_state_dict(state_dict_model)
    else:
        base_model_name = "tau/tavbert-he"
        config = AutoConfig.from_pretrained(base_model_name)
        dnikud_model = DNikudModel(config,
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
    del kwargs['pretrain_model_path']
    del kwargs['output_model_dir']
    kwargs['dnikud_model'] = dnikud_model

    del kwargs['command']
    del kwargs['func']
    args.func(**kwargs)

    sys.exit(0)
