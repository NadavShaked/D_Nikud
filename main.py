# ML
import argparse
import os
import random
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from transformers import AutoConfig
import numpy as np
# visual
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
# from torchviz import make_dot
from transformers import AutoTokenizer, AutoModelForMaskedLM

# DL
import logging
from src.models import DiacritizationModel, BaseModel, CharClassifierTransformer, ModelConfig
from src.models_utils import get_model_parameters, training, evaluate, freeze_model_parameters, predict
from src.plot_helpers import plot_results, plot_steps_info
from src.running_params import SEED
from src.utiles_data import NikudDataset, Nikud, Letters

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda'
cols = ["precision", "recall", "f1-score", "support"]

# Set the random seed for Python
random.seed(SEED)

# Set the random seed for numpy
np.random.seed(SEED)

# Set the random seed for torch to SEED
torch.manual_seed(SEED)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_model_dir', type=str, default='models', help='Save directory for model')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of train epochs')
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
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='AdamW beta1 hyperparameter')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='AdamW beta2 hyperparameter')
    parser.add_argument('--weight_decay', type=float, default=0.15, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--num_freeze_layers', type=int, default=5, help='number of freeze layers')
    parser.add_argument('--checkpoints_frequency', type=int, default=1, help='checkpoints saving frequency')
    parser.add_argument('--only_nikud', type=bool, default=False,
                        help='Want to train only on nikud classification (not dagesh/sin)')
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
                        help="Set the debug folder")  # "data/hebrew_diacritized"
    return parser.parse_args()


def main():
    args = parse_arguments()
    debug_folder = args.debug_folder
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    output_model_dir = args.output_model_dir
    batch_size = args.batch_size
    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    output_dir_running = os.path.join(output_model_dir, "trained", "latest", f"output_models_{date_time}")
    if not os.path.exists(output_dir_running):
        os.makedirs(output_dir_running)

    output_log_dir = os.path.join(args.log_dir,
                                  f"log_model_lr_{args.learning_rate}_bs_{batch_size}_{date_time}")
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)

    logger = get_logger(args.loglevel, output_log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    msg = f'Device detected: {device}'
    logger.info(msg)

    # training_args = TrainingArguments(**vars(args))  # vars: Namespace to dict

    msg = 'Loading data...'
    logger.debug(msg)

    tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    dataset_train = NikudDataset(tokenizer_tavbert, folder=os.path.join(args.data_folder, "train"), logger=logger)
    dataset_train.calc_max_length()
    dataset_dev = NikudDataset(tokenizer=tokenizer_tavbert, folder=os.path.join(args.data_folder, "dev"), logger=logger,
                               max_length=dataset_train.max_length)
    dataset_test = NikudDataset(tokenizer=tokenizer_tavbert, folder=os.path.join(args.data_folder, "test"),
                                logger=logger, max_length=dataset_train.max_length)
    # dataset.show_data_labels(debug_folder=debug_folder)

    msg = f'Max length of data: {dataset_train.max_length}'
    logger.debug(msg)

    msg = f'Num rows in train data: {len(dataset_train.data)}, ' \
          f'Num rows in dev data: {len(dataset_dev.data)}, ' \
          f'Num rows in test data: {len(dataset_test.data)}'
    logger.debug(msg)

    msg = 'Loading tokenizer and prepare data...'
    logger.debug(msg)

    dataset_train.prepare_data(name="train")  # , with_label=True)
    dataset_dev.prepare_data(name="dev")  # , with_label=True)
    dataset_test.prepare_data(name="test")  # , with_label=True)

    mtb_train_dl = torch.utils.data.DataLoader(dataset_train.prepered_data, batch_size=batch_size)
    mtb_dev_dl = torch.utils.data.DataLoader(dataset_dev.prepered_data, batch_size=batch_size)
    mtb_test_dl = torch.utils.data.DataLoader(dataset_test.prepered_data, batch_size=batch_size)
    msg = 'Loading model...'
    logger.debug(msg)

    # model_DM = BaseModel(400, Letters.vocab_size, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
    #                      len(Nikud.label_2_id["sin"])).to(DEVICE)
    # model_DM = CharClassifierTransformer(Letters.vocab_size, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
    #                      len(Nikud.label_2_id["sin"])).to(DEVICE)
    base_model_name = "tau/tavbert-he"
    config = AutoConfig.from_pretrained(base_model_name)
    model_DM = DiacritizationModel(config, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
                                   len(Nikud.label_2_id["sin"])).to(DEVICE)

    dir_model_config = os.path.join(output_model_dir, "config.yml")
    if not os.path.isfile(dir_model_config):
        our_model_config = ModelConfig(dataset_train.max_length)
        our_model_config.save_to_file()
    # all_model_params_MTB = model_DM.named_parameters()
    # top_layer_params = freeze_model_parameters(all_model_params_MTB)(all_model_params_MTB)  # args.num_freeze_layers)
    # top_layer_params = get_model_parameters(all_model_params_MTB, logger, 0)  # args.num_freeze_layers)
    optimizer = torch.optim.Adam(model_DM.parameters(), lr=args.learning_rate)

    msg = 'training...'
    logger.debug(msg)

    criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)

    # TODO: EXTRACT TO FUNCTION - PRINT MODEL
    if 0:
        dataloader = prepare_data([train[0]], DMtokenizer, dataset.max_length, batch_size=1, name="model_architecture")
        for (inputs, attention_mask, labels) in dataloader:
            y = model_DM(inputs, attention_mask).to(DEVICE)
            make_dot(y.mean(), params=dict(model_DM.named_parameters())).render("D-Nikud_model_architecture",
                                                                                format="png")

    training_params = {"n_epochs": args.n_epochs, "checkpoints_frequency": args.checkpoints_frequency}
    best_model_details, best_accuracy, loss_train_values, loss_dev_values, accuracy_dev_values = training(model_DM,
                                                                                                          mtb_train_dl,
                                                                                                          mtb_dev_dl,
                                                                                                          criterion_nikud,
                                                                                                          criterion_dagesh,
                                                                                                          criterion_sin,
                                                                                                          training_params,
                                                                                                          logger,
                                                                                                          output_dir_running,
                                                                                                          optimizer,
                                                                                                          args.only_nikud)

    # best_model = model_DM.named_parameters()#BaseModel(400, Letters.vocab_size, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
    #                     # len(Nikud.label_2_id["sin"])).to(DEVICE)
    model_DM.load_state_dict(best_model_details['model_state_dict'])

    report_dev, word_level_correct_dev, letter_level_correct_dev = evaluate(model_DM, mtb_dev_dl, debug_folder)
    report_test, word_level_correct_test, letter_level_correct_test = evaluate(model_DM, mtb_test_dl, debug_folder)

    msg = f"Diacritization Model\nDev dataset\nLetter level accuracy:{letter_level_correct_dev}\n" \
          f"Word level accuracy: {word_level_correct_dev}\n--------------------\nTest dataset\n" \
          f"Letter level accuracy: {letter_level_correct_test}\nWord level accuracy: {word_level_correct_test}"
    logger.debug(msg)

    plot_results(report_dev, report_filename="results_dev")
    plot_results(report_test, report_filename="results_test")

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

    if not os.path.exists(log_location):
        os.makedirs(log_location)

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


# def hyperparams_checker():
#     args = parse_arguments()
#     debug_folder = args.debug_folder
#     if not os.path.exists(debug_folder):
#         os.makedirs(debug_folder)
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     msg = f'Device detected: {device}'
#     # logger.info(msg)
#
#     # training_args = TrainingArguments(**vars(args))  # vars: Namespace to dict
#
#     msg = 'Loading data...'
#     # logger.debug(msg)
#
#     dataset = NikudDataset(folder=args.data_folder)  # , logger=logger)
#     dataset.show_data_labels(debug_folder=debug_folder)
#     dataset.calc_max_length()
#
#     msg = f'Max length of data: {dataset.max_length}'
#     # logger.debug(msg)
#
#     train, test = train_test_split(dataset.data, test_size=0.1, shuffle=True, random_state=SEED)
#     train, dev = train_test_split(train, test_size=0.1, shuffle=True, random_state=SEED)
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
#     log_dir = args.log_dir
#     output_dir = args.output_dir
#
#     # DMtokenizer = AutoTokenizer.from_pretrained("tau/tavbert-he")
#
#     # dataset_train = prepare_data_base(train, dataset.max_length, name="train")
#     # dataset_dev = prepare_data_base(dev, dataset.max_length, name="dev")
#
#     for _ in range(num_combinations):
#         torch.cuda.empty_cache()
#         lr = np.random.choice(lr_values)
#         nfl = np.random.choice(num_freeze_layers)
#         batch_size = int(np.random.choice(batch_size_values))
#
#         date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
#         output_dir_running = os.path.join(output_dir, f"output_models_{date_time}")
#         if not os.path.exists(output_dir_running):
#             os.makedirs(output_dir_running)
#
#         output_log_dir = os.path.join(log_dir,
#                                       f"log_model_lr_{lr}_bs_{batch_size}_nfl_{nfl}_{date_time}")
#         if not os.path.exists(output_log_dir):
#             os.makedirs(output_log_dir)
#
#         logger = get_logger(args.loglevel, output_log_dir)
#
#         msg_data = f'Num rows in train data: {len(train)}, ' \
#                    f'Num rows in dev data: {len(dev)}, ' \
#                    f'Num rows in test data: {len(test)}'
#         logger.debug(msg_data)
#
#         msg = 'Loading tokenizer and prepare data...'
#         logger.debug(msg)
#
#         msg = 'Loading model...'
#         logger.debug(msg)
#
#         model_DM = DiacritizationModel("tau/tavbert-he").to(DEVICE)
#         all_model_params_MTB = model_DM.named_parameters()
#         top_layer_params = get_model_parameters(all_model_params_MTB, logger, num_freeze_layers=nfl)
#
#         # set these hyperparameters in your optimizer
#         optimizer = torch.optim.Adam(top_layer_params, lr=args.learning_rate)
#
#         # redefine your data loaders with the new batch size
#         mtb_train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
#         mtb_dev_dl = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size)
#         # mtb_test_dl = prepare_data(test, DMtokenizer, dataset.max_length, batch_size=batch_size, name="test")
#
#         criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
#         criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
#         criterion_sin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
#
#         # call your training function and get the dev accuracy
#         _, dev_accuracy = training(model_DM, mtb_train_dl, mtb_dev_dl, criterion_nikud, criterion_dagesh, criterion_sin,
#                                    training_params, logger, output_dir_running, optimizer, only_nikud=args.only_nikud)
#
#         # if these hyperparameters are better, store them
#         if dev_accuracy > best_accuracy:
#             best_accuracy = dev_accuracy
#             best_hyperparameters = (lr, batch_size)
#
#     # print the best hyperparameters
#     print(best_hyperparameters)

def predict_flow(tokenizer_tavbert=None):
    args = parse_arguments()
    date_time = datetime.now().strftime('%d_%m_%y__%H_%M')
    output_log_dir = os.path.join(args.log_dir,
                                  f"log_model_predict_{date_time}")
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)

    output_model_dir = args.output_model_dir
    dir_model_config = os.path.join(output_model_dir, "config.yml")
    config = ModelConfig.load_from_file(dir_model_config)

    if tokenizer_tavbert is None:
        tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")

    logger = get_logger(args.loglevel, output_log_dir)
    dataset = NikudDataset(tokenizer_tavbert,
                           folder=r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized\test_modern\books",
                           logger=logger, max_length=config.max_length)

    model_DM = DiacritizationModel(config, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
                                   len(Nikud.label_2_id["sin"])).to(DEVICE)
    state_dict_model = model_DM.state_dict()
    state_dict_model.update(
        torch.load(r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\models\trained\prod\dnikud_v1.pth"))

    dataset.prepare_data(name="prediction")
    mtb_prediction_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=args.batch_size)
    all_labels = predict(model_DM, mtb_prediction_dl)
    text_data_with_labels = dataset.back_2_text(labels=all_labels)


if __name__ == '__main__':
    predict_flow()
    # main()
    # hyperparams_checker()
