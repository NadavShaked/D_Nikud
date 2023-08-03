# ML
# DL
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# visual
import seaborn as sns
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.running_params import DEBUG_MODE
from src.utiles_data import Nikud


def save_model(model, path):
    model_state = model.state_dict()
    torch.save(model_state, path)


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model


def get_model_parameters(params):
    top_layer_params = []
    for name, param in params:
        print(name)
        if "lm_head" in name or name.startswith("LayerNorm") or name.startswith('classifier'):  # 'layer.11' in name or
            top_layer_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    return top_layer_params


def training(model, train_data, dev_data, criterion_nikud, criterion_dagesh, criterion_sin, training_params, logger,
             output_model_path,
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

    output_checkpoints_path = os.path.join(output_model_path, "checkpoints")
    if not os.path.exists(output_checkpoints_path):
        os.makedirs(output_checkpoints_path)

    for epoch in tqdm(range(training_params["n_epochs"]), desc="Training"):
        model.train()
        train_loss = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        sum = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}

        for index_data, data in enumerate(train_loader):
            if DEBUG_MODE and index_data > 100:
                break
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

        msg = f"Epoch {epoch + 1}/{training_params['n_epochs']}\n" \
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
                if DEBUG_MODE and index_data > 100:
                    break
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

                correct_nikud = (torch.ones(mask_all_or.shape) == 1).to(device)
                correct_dagesh = (torch.ones(mask_all_or.shape) == 1).to(device)
                correct_sin = (torch.ones(mask_all_or.shape) == 1).to(device)

                correct_nikud[masks["nikud"]] = predictions["nikud"][masks["nikud"]] == labels_class["nikud"][
                    masks["nikud"]]
                correct_dagesh[masks["dagesh"]] = predictions["dagesh"][masks["dagesh"]] == labels_class["dagesh"][
                    masks["dagesh"]]
                correct_sin[masks["sin"]] = predictions["sin"][masks["sin"]] == labels_class["sin"][masks["sin"]]

                correct_preds_letter += torch.sum(
                    torch.logical_and(torch.logical_and(correct_sin[mask_all_or], correct_dagesh[mask_all_or]),
                                      correct_nikud[mask_all_or]))

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

        msg = f"Epoch {epoch + 1}/{training_params['n_epochs']}\n" \
              f'mean loss Dev nikud: {train_loss["nikud"]}, ' \
              f'mean loss Dev dagesh: {train_loss["dagesh"]}, ' \
              f'mean loss Dev sin: {train_loss["sin"]}, ' \
              f'Dev letter Accuracy: {dev_accuracy_letter}'
        logger.debug(msg)

        # calc accuracy by letter

        if dev_accuracy_letter > best_accuracy:
            best_accuracy = dev_accuracy_letter
            best_model = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }

        if epoch % training_params["checkpoints_frequency"] == 0:
            save_checkpoint_path = os.path.join(output_checkpoints_path, f'checkpoint_model_epoch_{epoch}.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint,
                       save_checkpoint_path)  # save_model(model, save_model_path)  # TODO: use this function in model class

    save_model_path = os.path.join(output_model_path, 'best_model.pth')
    torch.save(best_model, save_model_path)


def evaluate(model, test_data, debug_folder=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = {"nikud": 0, "dagesh": 0, "sin": 0}
    predictions = {"nikud": 0, "dagesh": 0, "sin": 0}
    predicted_labels_2_report = {"nikud": 0, "dagesh": 0, "sin": 0}
    masks = {"nikud": 0, "dagesh": 0, "sin": 0}
    reports = {}
    correct_preds = {"nikud": 0, "dagesh": 0, "sin": 0}
    sum = {"nikud": 0, "dagesh": 0, "sin": 0}
    labels_class = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}

    word_level_correct = 0.0
    letter_level_correct = 0.0

    with torch.no_grad():
        for index_data, data in enumerate(test_data):
            if DEBUG_MODE and index_data > 100:
                break
            (inputs, attention_mask, labels) = data

            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

            for i, (probs, name_class) in enumerate(
                    zip([nikud_probs, dagesh_probs, sin_probs], ["nikud", "dagesh", "sin"])):
                labels_class[name_class] = labels[:, :, i]
                mask = labels_class[name_class] != -1
                num_relevant = mask.sum()
                sum[name_class] += num_relevant
                _, preds = torch.max(probs, 2)
                correct_preds[name_class] += torch.sum(preds[mask] == labels_class[name_class][mask])
                predictions[name_class] = preds
                masks[name_class] = mask
                true_labels[name_class] = labels_class[name_class][mask].cpu().numpy()
                predicted_labels_2_report[name_class] = preds[mask].cpu().numpy()

            mask_all_or = torch.logical_or(torch.logical_or(masks["nikud"], masks["dagesh"]), masks["sin"])

            correct_nikud = (torch.ones(mask_all_or.shape) == 1).to(device)
            correct_dagesh = (torch.ones(mask_all_or.shape) == 1).to(device)
            correct_sin = (torch.ones(mask_all_or.shape) == 1).to(device)

            correct_nikud[masks["nikud"]] = predictions["nikud"][masks["nikud"]] == labels_class["nikud"][
                masks["nikud"]]
            correct_dagesh[masks["dagesh"]] = predictions["dagesh"][masks["dagesh"]] == labels_class["dagesh"][
                masks["dagesh"]]
            correct_sin[masks["sin"]] = predictions["sin"][masks["sin"]] == labels_class["sin"][masks["sin"]]

            letter_level_correct += torch.sum(
                torch.logical_and(torch.logical_and(correct_sin[mask_all_or], correct_dagesh[mask_all_or]),
                                  correct_nikud[mask_all_or]))

    for i, name in enumerate(["nikud", "dagesh", "sin"]):
        report = classification_report(true_labels[name], predicted_labels_2_report[name],
                                       output_dict=True)  # target_names=list(Nikud.label_2_id[name].keys()),

        reports[name] = report
        index_labels = np.unique(true_labels[name])
        cm = confusion_matrix(true_labels[name], predicted_labels_2_report[name], labels=index_labels)

        vowel_label = [Nikud.id_2_label[name][l] for l in index_labels]
        unique_vowels_names = [Nikud.sign_2_name[int(vowel)] for vowel in vowel_label if vowel!='WITHOUT']
        if "WITHOUT" in vowel_label:
            unique_vowels_names += ["WITHOUT"]
        cm_df = pd.DataFrame(cm, index=unique_vowels_names, columns=unique_vowels_names)

        # Display confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        if debug_folder is None:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_folder, F'Confusion_Matrix_{name}.jpg'))

    return reports, word_level_correct, letter_level_correct
