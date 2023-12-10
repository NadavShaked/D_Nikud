# general
import json
import os

# ML
import numpy as np
import pandas as pd
import torch

# visual
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.running_params import DEBUG_MODE
from src.utiles_data import Nikud, create_missing_folders

CLASSES_LIST = ["nikud", "dagesh", "sin"]


def calc_num_correct_words(input, letter_correct_mask):
    SPACE_TOKEN = 104
    START_SENTENCE_TOKEN = 1
    END_SENTENCE_TOKEN = 2

    correct_words_count = 0
    words_count = 0
    for index in range(input.shape[0]):
        input[index][np.where(input[index] == SPACE_TOKEN)[0]] = 0
        input[index][np.where(input[index] == START_SENTENCE_TOKEN)[0]] = 0
        input[index][np.where(input[index] == END_SENTENCE_TOKEN)[0]] = 0
        words_end_index = np.concatenate(
            (np.array([-1]), np.where(input[index] == 0)[0])
        )
        is_correct_words_array = [
            bool(
                letter_correct_mask[index][
                    list(range((words_end_index[s] + 1), words_end_index[s + 1]))
                ].all()
            )
            for s in range(len(words_end_index) - 1)
            if words_end_index[s + 1] - (words_end_index[s] + 1) > 1
        ]
        correct_words_count += np.array(is_correct_words_array).sum()
        words_count += len(is_correct_words_array)

    return correct_words_count, words_count


def predict(model, data_loader, device="cpu"):
    model.to(device)

    all_labels = None
    with torch.no_grad():
        for index_data, data in enumerate(data_loader):
            (inputs, attention_mask, labels_demo) = data
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels_demo = labels_demo.to(device)

            mask_cant_be_nikud = np.array(labels_demo.cpu())[:, :, 0] == -1
            mask_cant_be_dagesh = np.array(labels_demo.cpu())[:, :, 1] == -1
            mask_cant_be_sin = np.array(labels_demo.cpu())[:, :, 2] == -1

            nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

            pred_nikud = np.array(torch.max(nikud_probs, 2).indices.cpu()).reshape(
                inputs.shape[0], inputs.shape[1], 1
            )
            pred_dagesh = np.array(torch.max(dagesh_probs, 2).indices.cpu()).reshape(
                inputs.shape[0], inputs.shape[1], 1
            )
            pred_sin = np.array(torch.max(sin_probs, 2).indices.cpu()).reshape(
                inputs.shape[0], inputs.shape[1], 1
            )

            pred_nikud[mask_cant_be_nikud] = -1
            pred_dagesh[mask_cant_be_dagesh] = -1
            pred_sin[mask_cant_be_sin] = -1

            pred_labels = np.concatenate((pred_nikud, pred_dagesh, pred_sin), axis=2)

            if all_labels is None:
                all_labels = pred_labels
            else:
                all_labels = np.concatenate((all_labels, pred_labels), axis=0)

    return all_labels


def predict_single(model, data, device="cpu"):
    # model.to(device)

    all_labels = None
    with torch.no_grad():
        inputs = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)

        # mask_cant_be_nikud = np.array(labels_demo.cpu())[:, :, 0] == -1
        # mask_cant_be_dagesh = np.array(labels_demo.cpu())[:, :, 1] == -1
        # mask_cant_be_sin = np.array(labels_demo.cpu())[:, :, 2] == -1

        nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)
        print(nikud_probs, dagesh_probs, sin_probs)

        pred_nikud = np.array(torch.max(nikud_probs, 2).indices.cpu()).reshape(
            inputs.shape[0], inputs.shape[1], 1
        )
        pred_dagesh = np.array(torch.max(dagesh_probs, 2).indices.cpu()).reshape(
            inputs.shape[0], inputs.shape[1], 1
        )
        pred_sin = np.array(torch.max(sin_probs, 2).indices.cpu()).reshape(
            inputs.shape[0], inputs.shape[1], 1
        )

        # pred_nikud[mask_cant_be_nikud] = -1
        # pred_dagesh[mask_cant_be_dagesh] = -1
        # pred_sin[mask_cant_be_sin] = -1
        # print(pred_nikud, pred_dagesh, pred_sin)
        pred_labels = np.concatenate((pred_nikud, pred_dagesh, pred_sin), axis=2)
        print(pred_labels)
        if all_labels is None:
            all_labels = pred_labels
        else:
            all_labels = np.concatenate((all_labels, pred_labels), axis=0)

    return all_labels


def training(
    model,
    train_loader,
    dev_loader,
    criterion_nikud,
    criterion_dagesh,
    criterion_sin,
    training_params,
    logger,
    output_model_path,
    optimizer,
    device="cpu",
):
    max_length = None
    best_accuracy = 0.0

    logger.info(f"start training with training_params: {training_params}")
    model = model.to(device)

    criteria = {
        "nikud": criterion_nikud.to(device),
        "dagesh": criterion_dagesh.to(device),
        "sin": criterion_sin.to(device),
    }

    output_checkpoints_path = os.path.join(output_model_path, "checkpoints")
    create_missing_folders(output_checkpoints_path)

    train_steps_loss_values = {"nikud": [], "dagesh": [], "sin": []}
    train_epochs_loss_values = {"nikud": [], "dagesh": [], "sin": []}
    dev_loss_values = {"nikud": [], "dagesh": [], "sin": []}
    dev_accuracy_values = {
        "nikud": [],
        "dagesh": [],
        "sin": [],
        "all_nikud_letter": [],
        "all_nikud_word": [],
    }

    for epoch in tqdm(range(training_params["n_epochs"]), desc="Training"):
        model.train()
        train_loss = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        relevant_count = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}

        for index_data, data in enumerate(train_loader):
            (inputs, attention_mask, labels) = data

            if max_length is None:
                max_length = labels.shape[1]

            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

            for i, (probs, class_name) in enumerate(
                zip([nikud_probs, dagesh_probs, sin_probs], CLASSES_LIST)
            ):
                reshaped_tensor = (
                    torch.transpose(probs, 1, 2)
                    .contiguous()
                    .view(probs.shape[0], probs.shape[2], probs.shape[1])
                )
                loss = criteria[class_name](reshaped_tensor, labels[:, :, i]).to(device)

                num_relevant = (labels[:, :, i] != -1).sum()
                train_loss[class_name] += loss.item() * num_relevant
                relevant_count[class_name] += num_relevant

                loss.backward(retain_graph=True)

            for i, class_name in enumerate(CLASSES_LIST):
                train_steps_loss_values[class_name].append(
                    float(train_loss[class_name] / relevant_count[class_name])
                )

            optimizer.step()
            if (index_data + 1) % 100 == 0:
                msg = f"epoch: {epoch} , index_data: {index_data + 1}\n"
                for i, class_name in enumerate(CLASSES_LIST):
                    msg += f"mean loss train {class_name}: {float(train_loss[class_name] / relevant_count[class_name])}, "

                logger.debug(msg[:-2])

        for i, class_name in enumerate(CLASSES_LIST):
            train_epochs_loss_values[class_name].append(
                float(train_loss[class_name] / relevant_count[class_name])
            )

        for class_name in train_loss.keys():
            train_loss[class_name] /= relevant_count[class_name]

        msg = f"Epoch {epoch + 1}/{training_params['n_epochs']}\n"
        for i, class_name in enumerate(CLASSES_LIST):
            msg += f"mean loss train {class_name}: {train_loss[class_name]}, "
        logger.debug(msg[:-2])

        model.eval()
        dev_loss = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        dev_accuracy = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        relevant_count = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        correct_preds = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        un_masks = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        predictions = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}
        labels_class = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}

        all_nikud_types_correct_preds_letter = 0.0

        letter_count = 0.0
        correct_words_count = 0.0
        word_count = 0.0
        with torch.no_grad():
            for index_data, data in enumerate(dev_loader):
                (inputs, attention_mask, labels) = data
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

                for i, (probs, class_name) in enumerate(
                    zip([nikud_probs, dagesh_probs, sin_probs], CLASSES_LIST)
                ):
                    reshaped_tensor = (
                        torch.transpose(probs, 1, 2)
                        .contiguous()
                        .view(probs.shape[0], probs.shape[2], probs.shape[1])
                    )
                    loss = criteria[class_name](reshaped_tensor, labels[:, :, i]).to(
                        device
                    )
                    un_masked = labels[:, :, i] != -1
                    num_relevant = un_masked.sum()
                    relevant_count[class_name] += num_relevant
                    _, preds = torch.max(probs, 2)
                    dev_loss[class_name] += loss.item() * num_relevant
                    correct_preds[class_name] += torch.sum(
                        preds[un_masked] == labels[:, :, i][un_masked]
                    )
                    un_masks[class_name] = un_masked
                    predictions[class_name] = preds
                    labels_class[class_name] = labels[:, :, i]

                un_mask_all_or = torch.logical_or(
                    torch.logical_or(un_masks["nikud"], un_masks["dagesh"]),
                    un_masks["sin"],
                )

                correct = {
                    class_name: (torch.ones(un_mask_all_or.shape) == 1).to(device)
                    for class_name in CLASSES_LIST
                }

                for i, class_name in enumerate(CLASSES_LIST):
                    correct[class_name][un_masks[class_name]] = (
                        predictions[class_name][un_masks[class_name]]
                        == labels_class[class_name][un_masks[class_name]]
                    )

                letter_correct_mask = torch.logical_and(
                    torch.logical_and(correct["sin"], correct["dagesh"]),
                    correct["nikud"],
                )
                all_nikud_types_correct_preds_letter += torch.sum(
                    letter_correct_mask[un_mask_all_or]
                )

                letter_correct_mask[~un_mask_all_or] = True
                correct_num, total_words_num = calc_num_correct_words(
                    inputs.cpu(), letter_correct_mask
                )

                word_count += total_words_num
                correct_words_count += correct_num
                letter_count += un_mask_all_or.sum()

        for class_name in CLASSES_LIST:
            dev_loss[class_name] /= relevant_count[class_name]
            dev_accuracy[class_name] = float(
                correct_preds[class_name].double() / relevant_count[class_name]
            )

            dev_loss_values[class_name].append(float(dev_loss[class_name]))
            dev_accuracy_values[class_name].append(float(dev_accuracy[class_name]))

        dev_all_nikud_types_accuracy_letter = float(
            all_nikud_types_correct_preds_letter / letter_count
        )

        dev_accuracy_values["all_nikud_letter"].append(
            dev_all_nikud_types_accuracy_letter
        )

        word_all_nikud_accuracy = correct_words_count / word_count
        dev_accuracy_values["all_nikud_word"].append(word_all_nikud_accuracy)

        msg = (
            f"Epoch {epoch + 1}/{training_params['n_epochs']}\n"
            f'mean loss Dev nikud: {train_loss["nikud"]}, '
            f'mean loss Dev dagesh: {train_loss["dagesh"]}, '
            f'mean loss Dev sin: {train_loss["sin"]}, '
            f"Dev all nikud types letter Accuracy: {dev_all_nikud_types_accuracy_letter}, "
            f'Dev nikud letter Accuracy: {dev_accuracy["nikud"]}, '
            f'Dev dagesh letter Accuracy: {dev_accuracy["dagesh"]}, '
            f'Dev sin letter Accuracy: {dev_accuracy["sin"]}, '
            f"Dev word Accuracy: {word_all_nikud_accuracy}"
        )
        logger.debug(msg)

        save_progress_details(
            dev_accuracy_values,
            train_epochs_loss_values,
            dev_loss_values,
            train_steps_loss_values,
        )

        if dev_all_nikud_types_accuracy_letter > best_accuracy:
            best_accuracy = dev_all_nikud_types_accuracy_letter
            best_model = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }

        if epoch % training_params["checkpoints_frequency"] == 0:
            save_checkpoint_path = os.path.join(
                output_checkpoints_path, f"checkpoint_model_epoch_{epoch + 1}.pth"
            )
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
            torch.save(checkpoint["model_state_dict"], save_checkpoint_path)

    save_model_path = os.path.join(output_model_path, "best_model.pth")
    torch.save(best_model["model_state_dict"], save_model_path)
    return (
        best_model,
        best_accuracy,
        train_epochs_loss_values,
        train_steps_loss_values,
        dev_loss_values,
        dev_accuracy_values,
    )


def save_progress_details(
    accuracy_dev_values,
    epochs_loss_train_values,
    loss_dev_values,
    steps_loss_train_values,
):
    epochs_data_path = "epochs_data"
    create_missing_folders(epochs_data_path)

    save_dict_as_json(
        steps_loss_train_values, epochs_data_path, "steps_loss_train_values.json"
    )
    save_dict_as_json(
        epochs_loss_train_values, epochs_data_path, "epochs_loss_train_values.json"
    )
    save_dict_as_json(loss_dev_values, epochs_data_path, "loss_dev_values.json")
    save_dict_as_json(accuracy_dev_values, epochs_data_path, "accuracy_dev_values.json")


def save_dict_as_json(dict, file_path, file_name):
    json_data = json.dumps(dict, indent=4)
    with open(os.path.join(file_path, file_name), "w") as json_file:
        json_file.write(json_data)


def evaluate(model, test_data, plots_folder=None, device="cpu"):
    model.to(device)
    model.eval()

    true_labels = {"nikud": [], "dagesh": [], "sin": []}
    predictions = {"nikud": 0, "dagesh": 0, "sin": 0}
    predicted_labels_2_report = {"nikud": [], "dagesh": [], "sin": []}
    not_masks = {"nikud": 0, "dagesh": 0, "sin": 0}
    correct_preds = {"nikud": 0, "dagesh": 0, "sin": 0}
    relevant_count = {"nikud": 0, "dagesh": 0, "sin": 0}
    labels_class = {"nikud": 0.0, "dagesh": 0.0, "sin": 0.0}

    all_nikud_types_letter_level_correct = 0.0
    nikud_letter_level_correct = 0.0
    dagesh_letter_level_correct = 0.0
    sin_letter_level_correct = 0.0

    letters_count = 0.0
    words_count = 0.0
    correct_words_count = 0.0
    with torch.no_grad():
        for index_data, data in enumerate(test_data):
            if DEBUG_MODE and index_data > 100:
                break

            (inputs, attention_mask, labels) = data

            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            nikud_probs, dagesh_probs, sin_probs = model(inputs, attention_mask)

            for i, (probs, class_name) in enumerate(
                zip([nikud_probs, dagesh_probs, sin_probs], CLASSES_LIST)
            ):
                labels_class[class_name] = labels[:, :, i]
                not_masked = labels_class[class_name] != -1
                num_relevant = not_masked.sum()
                relevant_count[class_name] += num_relevant
                _, preds = torch.max(probs, 2)
                correct_preds[class_name] += torch.sum(
                    preds[not_masked] == labels_class[class_name][not_masked]
                )
                predictions[class_name] = preds
                not_masks[class_name] = not_masked

                if len(true_labels[class_name]) == 0:
                    true_labels[class_name] = (
                        labels_class[class_name][not_masked].cpu().numpy()
                    )
                else:
                    true_labels[class_name] = np.concatenate(
                        (
                            true_labels[class_name],
                            labels_class[class_name][not_masked].cpu().numpy(),
                        )
                    )

                if len(predicted_labels_2_report[class_name]) == 0:
                    predicted_labels_2_report[class_name] = (
                        preds[not_masked].cpu().numpy()
                    )
                else:
                    predicted_labels_2_report[class_name] = np.concatenate(
                        (
                            predicted_labels_2_report[class_name],
                            preds[not_masked].cpu().numpy(),
                        )
                    )

            not_mask_all_or = torch.logical_or(
                torch.logical_or(not_masks["nikud"], not_masks["dagesh"]),
                not_masks["sin"],
            )

            correct_nikud = (torch.ones(not_mask_all_or.shape) == 1).to(device)
            correct_dagesh = (torch.ones(not_mask_all_or.shape) == 1).to(device)
            correct_sin = (torch.ones(not_mask_all_or.shape) == 1).to(device)

            correct_nikud[not_masks["nikud"]] = (
                predictions["nikud"][not_masks["nikud"]]
                == labels_class["nikud"][not_masks["nikud"]]
            )
            correct_dagesh[not_masks["dagesh"]] = (
                predictions["dagesh"][not_masks["dagesh"]]
                == labels_class["dagesh"][not_masks["dagesh"]]
            )
            correct_sin[not_masks["sin"]] = (
                predictions["sin"][not_masks["sin"]]
                == labels_class["sin"][not_masks["sin"]]
            )

            letter_correct_mask = torch.logical_and(
                torch.logical_and(correct_sin, correct_dagesh), correct_nikud
            )
            all_nikud_types_letter_level_correct += torch.sum(
                letter_correct_mask[not_mask_all_or]
            )

            letter_correct_mask[~not_mask_all_or] = True
            total_correct_count, total_words_num = calc_num_correct_words(
                inputs.cpu(), letter_correct_mask
            )

            words_count += total_words_num
            correct_words_count += total_correct_count

            letters_count += not_mask_all_or.sum()

            nikud_letter_level_correct += torch.sum(correct_nikud[not_mask_all_or])
            dagesh_letter_level_correct += torch.sum(correct_dagesh[not_mask_all_or])
            sin_letter_level_correct += torch.sum(correct_sin[not_mask_all_or])

    for i, name in enumerate(CLASSES_LIST):
        index_labels = np.unique(true_labels[name])
        cm = confusion_matrix(
            true_labels[name], predicted_labels_2_report[name], labels=index_labels
        )

        vowel_label = [Nikud.id_2_label[name][l] for l in index_labels]
        unique_vowels_names = [
            Nikud.sign_2_name[int(vowel)] for vowel in vowel_label if vowel != "WITHOUT"
        ]
        if "WITHOUT" in vowel_label:
            unique_vowels_names += ["WITHOUT"]
        cm_df = pd.DataFrame(cm, index=unique_vowels_names, columns=unique_vowels_names)

        # Display confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        if plots_folder is None:
            plt.show()
        else:
            plt.savefig(os.path.join(plots_folder, f"Confusion_Matrix_{name}.jpg"))

    all_nikud_types_letter_level_correct = (
        all_nikud_types_letter_level_correct / letters_count
    )
    all_nikud_types_word_level_correct = correct_words_count / words_count
    nikud_letter_level_correct = nikud_letter_level_correct / letters_count
    dagesh_letter_level_correct = dagesh_letter_level_correct / letters_count
    sin_letter_level_correct = sin_letter_level_correct / letters_count
    print("\n")
    print(f"nikud_letter_level_correct = {nikud_letter_level_correct}")
    print(f"dagesh_letter_level_correct = {dagesh_letter_level_correct}")
    print(f"sin_letter_level_correct = {sin_letter_level_correct}")
    print(f"word_level_correct = {all_nikud_types_word_level_correct}")

    return all_nikud_types_word_level_correct, all_nikud_types_letter_level_correct
