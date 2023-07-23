# ML
import argparse
# DL
import copy
import random

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
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers import TrainingArguments

from src.utiles_data import NikudDataset, prepare_data, Nikud

# DL
# HF

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda'
cols = ["precision", "recall", "f1-score", "support"]
SEED = 42

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
        if name.startswith('classifier') or name.startswith('bert.pooler') or 'layer.11' in name:
            top_layer_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    return top_layer_params


class DiacritizationModel(nn.Module):
    def __init__(self, base_model_name, vocab_size, num_labels):
        super(DiacritizationModel, self).__init__()
        config = AutoConfig.from_pretrained(base_model_name, num_labels=num_labels)
        self.model = AutoModelForMaskedLM.from_pretrained(base_model_name, num_labels=num_labels).bert.to(
            DEVICE)
        self.model.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.classifier_nikud = nn.Linear(config.hidden_size, num_labels)
        self.classifier_shin = nn.Linear(config.hidden_size, 2)
        self.classifier_dagesh = nn.Linear(config.hidden_size, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, e1_start, e2_start):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        batch_size = e1_start.size(0)
        e1_output = torch.stack([sequence_output[i][e1_start[i]] for i in range(batch_size)], dim=0)
        e2_output = torch.stack([sequence_output[i][e2_start[i]] for i in range(batch_size)], dim=0)
        e1_and_e2_output = torch.cat((e1_output, e2_output), dim=1)
        e1_and_e2_output = self.LayerNorm(e1_and_e2_output)

        output = self.classifier(e1_and_e2_output)
        return self.softmax(output)


def training(model, n_epochs, train_data, dev_data, optimizer=None, criterion=None, is_MTB=False):
    best_accuracy = 0.0
    best_model_weights = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    train_loader = train_data
    dev_loader = dev_data

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            if is_MTB:
                (inputs, attention_mask, labels, e1_starts, e2_starts) = data
                e1_starts = e1_starts.to(device)
                e2_starts = e2_starts.to(device)
            else:
                (inputs, attention_mask, labels) = data
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if is_MTB:
                outputs = model(inputs, attention_mask, e1_starts, e2_starts)
                loss = criterion(outputs, labels).to(device)
            else:
                outputs = model(inputs, attention_mask)
                loss = criterion(outputs[0], labels).to(device)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        tqdm.write(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}")

        model.eval()
        dev_loss = 0.0
        correct_preds = 0

        with torch.no_grad():
            for data in dev_loader:
                if is_MTB:
                    (inputs, attention_mask, labels, e1_starts, e2_starts) = data
                    e1_starts = e1_starts.to(device)
                    e2_starts = e2_starts.to(device)
                else:
                    (inputs, attention_mask, labels) = data

                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                if is_MTB:
                    outputs = model(inputs, attention_mask, e1_starts, e2_starts)
                    loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs, attention_mask)
                    loss = criterion(outputs[0], labels)

                dev_loss += loss.item() * inputs.size(0)
                if is_MTB:
                    _, preds = torch.max(outputs, 1)
                else:
                    _, preds = torch.max(outputs[0], 1)
                correct_preds += torch.sum(preds == labels.data)

        dev_loss /= len(dev_loader.dataset)
        dev_accuracy = correct_preds.double() / len(dev_loader.dataset)

        tqdm.write(f"Epoch {epoch + 1}/{n_epochs}, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load the weights of the best model
    model.load_state_dict(best_model_weights)


def evaluate(model, test_data, report_filename, is_MTB=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in test_data:
            if is_MTB:
                (inputs, attention_mask, labels, e1_starts, e2_starts) = data
                e1_starts = e1_starts.to(device)
                e2_starts = e2_starts.to(device)
            else:
                (inputs, attention_mask, labels) = data

            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            if is_MTB:
                outputs = model(inputs, attention_mask, e1_starts, e2_starts)
                _, preds = torch.max(outputs, 1)
            else:
                outputs = model(inputs, attention_mask=attention_mask)
                _, preds = torch.max(outputs[0], 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    report = classification_report(true_labels, predicted_labels, target_names=list(Nikud.nikud_2_id.keys()),
                                   output_dict=True)

    df = pd.DataFrame(report).transpose()
    df = df[cols]

    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))

    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=list(Nikud.nikud_2_id.keys()), columns=list(Nikud.nikud_2_id.keys()))

    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()

    # Save report to CSV
    df.to_csv(report_filename)

    print(f"Evaluation report saved to {report_filename}")


OUTPUT_DIR = 'models/trained/latest'


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Save directory for model')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of train epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (train)')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Validation batch size')
    parser.add_argument('--save_strategy', type=str, default='no', help='Whether to save on every epoch ("epoch"/"no")')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help='Learning rate scheduler type ("linear"/"cosine"/"constant"/...')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='AdamW beta1 hyperparameter')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='AdamW beta2 hyperparameter')
    parser.add_argument('--weight_decay', type=float, default=0.15, help='Weight decay')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        help='How to validate (set to "no" for no validation)')
    parser.add_argument('--eval_steps', type=int, default=2000, help='Validate every N steps')
    return parser.parse_args()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device detected:', device)

    args = parse_arguments()
    # training_args = TrainingArguments(**vars(args))  # vars: Namespace to dict

    print('Loading data...')
    dataset = NikudDataset()
    dataset.show_data_labels()
    dataset.calc_max_length()
    train, test = train_test_split(dataset.data, test_size=0.1, shuffle=True, random_state=SEED)
    train, dev = train_test_split(train, test_size=0.05, shuffle=True, random_state=SEED)

    print('Loading tokenizer...')
    DMtokenizer = AutoTokenizer.from_pretrained("imvladikon/alephbertgimmel-base-512")
    # prepare_data(dataset, DMtokenizer, Nikud.nikud_2_id, batch_size=8)
    mtb_train_dl = prepare_data(train, DMtokenizer, dataset.max_length, batch_size = 8)
    mtb_dev_dl = prepare_data(dev, DMtokenizer, dataset.max_length, batch_size=8)
    mtb_test_dl = prepare_data(test, DMtokenizer, dataset.max_length, batch_size=8)
    print('Loading model...')
    model_DM = DiacritizationModel("imvladikon/alephbertgimmel-base-512").to(DEVICE)
    all_model_params_MTB = model_DM.named_parameters()
    top_layer_params = get_parameters(all_model_params_MTB)
    optimizer = torch.optim.Adam(top_layer_params, lr=0.0001)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    print('Creating trainer...')
    training(model_DM, 5, mtb_train_dl, mtb_dev_dl, optimizer=optimizer, is_MTB=True)

    evaluate(model_DM, mtb_test_dl, f"316550797_312494925_part5.csv")
    print('Done')
    # trainer = Trainer(
    #     model=model_DM,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset
    # )
    #
    # print(f'Training... (on device: {device})')
    # trainer.train()
    #
    # print(f'Saving to: {OUTPUT_DIR}')
    # trainer.save_model(f'{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
