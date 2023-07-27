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
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, RobertaForMaskedLM
from transformers import TrainingArguments
import torch.nn.functional as F
from src.utiles_data import NikudDataset, prepare_data, Nikud, Letters, DEBUG_MODE

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
        self.classifier_shin = nn.Linear(config.hidden_size, Nikud.LEN_SIN)
        self.classifier_dagesh = nn.Linear(config.hidden_size, Nikud.LEN_DAGESH)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
        normalized_hidden_states = self.LayerNorm(last_hidden_state)

        # Classifier for Nikud
        nikud_logits = self.classifier_nikud(normalized_hidden_states)
        nikud_probs = self.softmax(nikud_logits)

        # Classifier for Dagesh
        dagesh_logits = self.classifier_dagesh(normalized_hidden_states)
        dagesh_probs = self.softmax(dagesh_logits)

        # Classifier for Shin
        shin_logits = self.classifier_shin(normalized_hidden_states)
        shin_probs = self.softmax(shin_logits)

        # Return the probabilities for each diacritical mark
        return nikud_probs, dagesh_probs, shin_probs


def training(model, n_epochs, train_data, dev_data, criterion_nikud, criterion_dagesh, criterion_shin, optimizer=None):
    best_accuracy = 0.0
    best_model_weights = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion_nikud.to(device)
    criterion_dagesh.to(device)
    criterion_shin.to(device)

    train_loader = train_data
    dev_loader = dev_data
    max_length = None
    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        train_loss_nikud = 0.0
        train_loss_dagesh = 0.0
        train_loss_shin = 0.0
        sum_nikud = 0.0
        sum_dagesh = 0.0
        sum_shin = 0.0

        for i, data in enumerate(train_loader):
            if DEBUG_MODE and i > 100:
                break
            (inputs, attention_mask, labels) = data
            if max_length is None:
                max_length = labels.shape[1]
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            nikud_probs, dagesh_probs, shin_probs = model(inputs, attention_mask)

            # sep_id = 2
            # length_sentences = np.where(np.array(inputs.data) == sep_id)[1] - 1

            reshaped_tensor_nikud = torch.transpose(nikud_probs, 1, 2).contiguous().view(nikud_probs.shape[0],
                                                                                         nikud_probs.shape[2],
                                                                                         nikud_probs.shape[1])
            nikud_labels = labels[:, :, 0]
            loss_nikud = criterion_nikud(reshaped_tensor_nikud, nikud_labels).to(device)

            num_relevant = (nikud_labels != -1).sum()
            train_loss_nikud += loss_nikud.item() * num_relevant
            sum_nikud += num_relevant

            dagesh_labels = labels[:, :, 1]
            reshaped_tensor_dagesh = torch.transpose(dagesh_probs, 1, 2).contiguous().view(dagesh_probs.shape[0],
                                                                                           dagesh_probs.shape[2],
                                                                                           dagesh_probs.shape[1])
            loss_dagesh = criterion_dagesh(reshaped_tensor_dagesh, dagesh_labels).to(device)

            num_relevant = (dagesh_labels != -1).sum()
            train_loss_dagesh += loss_dagesh.item() * num_relevant
            sum_dagesh += num_relevant

            shin_labels = labels[:, :, 2]
            reshaped_tensor_shin = torch.transpose(shin_probs, 1, 2).contiguous().view(shin_probs.shape[0],
                                                                                       shin_probs.shape[2],
                                                                                       shin_probs.shape[1])
            loss_shin = criterion_shin(reshaped_tensor_shin, shin_labels).to(device)

            num_relevant = (shin_labels != -1).sum()
            train_loss_shin += loss_shin.item() * num_relevant
            sum_shin += num_relevant

            loss_nikud.backward(retain_graph=True)
            loss_dagesh.backward(retain_graph=True)
            loss_shin.backward(retain_graph=True)

            optimizer.step()

        train_loss_nikud /= sum_nikud
        train_loss_dagesh /= sum_dagesh
        train_loss_shin /= sum_shin

        tqdm.write(
            f"Epoch {epoch + 1}/{n_epochs}, Train train_loss_nikud: {train_loss_nikud:.4f}, "
            f"Train train_loss_dagesh: {train_loss_dagesh:.4f}, Train train_loss_shin: {train_loss_shin:.4f}")

        model.eval()
        dev_loss_nikud = 0.0
        dev_loss_dagesh = 0.0
        dev_loss_shin = 0.0
        correct_preds_nikud = 0
        correct_preds_dagesh = 0
        correct_preds_shin = 0
        sum_nikud = 0.0
        sum_dagesh = 0.0
        sum_shin = 0.0
        correct_preds_letter = 0.0
        with torch.no_grad():
            for data in dev_loader:
                (inputs, attention_mask, labels) = data

                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                nikud_probs, dagesh_probs, shin_probs = model(inputs, attention_mask)

                reshaped_tensor_nikud = torch.transpose(nikud_probs, 1, 2).contiguous().view(nikud_probs.shape[0],
                                                                                             nikud_probs.shape[2],
                                                                                             nikud_probs.shape[1])
                loss_nikud = criterion_nikud(reshaped_tensor_nikud, nikud_labels).to(device)
                nikud_labels = labels[:, :, 0]
                mask_nikud = nikud_labels != -1
                num_relevant = (mask_nikud).sum()
                sum_nikud += num_relevant
                _, preds_nikud = torch.max(nikud_probs, 2)
                dev_loss_nikud += loss_nikud.item() * num_relevant
                correct_preds_nikud += torch.sum(preds_nikud[mask_nikud] == nikud_labels[mask_nikud])

                reshaped_tensor_dagesh = torch.transpose(dagesh_probs, 1, 2).contiguous().view(dagesh_probs.shape[0],
                                                                                               dagesh_probs.shape[2],
                                                                                               dagesh_probs.shape[1])
                loss_dagesh = criterion_dagesh(reshaped_tensor_dagesh, dagesh_labels).to(device)
                dagesh_labels = labels[:, :, 1]
                mask_dagesh = dagesh_labels != -1
                num_relevant = mask_dagesh.sum()
                sum_dagesh += num_relevant
                _, preds_dagesh = torch.max(dagesh_probs, 2)
                dev_loss_dagesh += loss_dagesh.item() * num_relevant
                correct_preds_dagesh += torch.sum(preds_dagesh[mask_dagesh] == dagesh_labels[mask_dagesh])

                reshaped_tensor_shin = torch.transpose(shin_probs, 1, 2).contiguous().view(shin_probs.shape[0],
                                                                                           shin_probs.shape[2],
                                                                                           shin_probs.shape[1])
                loss_shin = criterion_shin(reshaped_tensor_shin, shin_labels).to(device)
                shin_labels = labels[:, :, 2]
                mask_sin = shin_labels != -1
                num_relevant = mask_sin.sum()
                sum_shin += num_relevant
                _, preds_shin = torch.max(shin_probs, 2)
                dev_loss_shin += loss_shin.item() * num_relevant
                correct_preds_shin += torch.sum(preds_shin[mask_sin] == shin_labels[mask_sin])

                correct_preds_letter = 0

        dev_loss_nikud /= sum_nikud
        dev_loss_dagesh /= sum_dagesh
        dev_loss_shin /= sum_shin

        dev_accuracy_nikud = correct_preds_nikud.double() / sum_nikud
        dev_accuracy_dagesh = correct_preds_dagesh.double() / sum_dagesh
        dev_accuracy_shin = correct_preds_shin.double() / sum_shin

        tqdm.write(
            f"Epoch {epoch + 1}/{n_epochs}, Dev Nikud Loss: {dev_loss_nikud:.4f}, Dev Nikud Accuracy: {dev_accuracy_nikud:.4f}"
            f", Dev dagesh Loss: {dev_loss_dagesh:.4f}, Dev dagesh Accuracy: {dev_accuracy_dagesh:.4f}"
            f", Dev shin Loss: {dev_loss_shin:.4f}, Dev shin Accuracy: {dev_accuracy_shin:.4f}")

        # calc accuracy by letter


        if dev_accuracy_nikud > best_accuracy:
            best_accuracy = dev_accuracy_nikud
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load the weights of the best model
    model.load_state_dict(best_model_weights)


def evaluate(model, test_data, report_filename="results"):
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
    return parser.parse_args()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device detected:', device)

    args = parse_arguments()
    # training_args = TrainingArguments(**vars(args))  # vars: Namespace to dict

    print('Loading data...')
    dataset = NikudDataset()
    # dataset.show_data_labels()
    dataset.calc_max_length()
    train, test = train_test_split(dataset.data, test_size=0.1, shuffle=True, random_state=SEED)
    train, dev = train_test_split(train, test_size=0.1, shuffle=True, random_state=SEED)

    print('Loading tokenizer...')
    DMtokenizer = AutoTokenizer.from_pretrained("tau/tavbert-he")
    # prepare_data(dataset, DMtokenizer, Nikud.nikud_2_id, batch_size=8)
    mtb_train_dl = prepare_data(train, DMtokenizer, dataset.max_length, batch_size=8, name="train")
    mtb_dev_dl = prepare_data(dev, DMtokenizer, dataset.max_length, batch_size=8, name="dev")
    mtb_test_dl = prepare_data(test, DMtokenizer, dataset.max_length, batch_size=8, name="test")
    print('Loading model...')
    model_DM = DiacritizationModel("tau/tavbert-he").to(DEVICE)
    all_model_params_MTB = model_DM.named_parameters()
    top_layer_params = get_parameters(all_model_params_MTB)
    optimizer = torch.optim.Adam(top_layer_params, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    print('Creating trainer...')
    criterion_nikud = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    criterion_dagesh = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    criterion_shin = nn.CrossEntropyLoss(ignore_index=Nikud.PAD).to(DEVICE)
    training(model_DM, 5, mtb_train_dl, mtb_dev_dl, criterion_nikud, criterion_dagesh, criterion_shin,
             optimizer=optimizer)

    evaluate(model_DM, mtb_test_dl)
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
