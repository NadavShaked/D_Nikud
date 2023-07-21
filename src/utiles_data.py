import os
from typing import List, Tuple
import glob2
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Tkagg')

DEBUG_MODE = True


class Niqqud:
    niqud_dict = {"SHVA": '\u05B0', "REDUCED_SEGOL": '\u05B1', "REDUCED_PATAKH": '\u05B2', "REDUCED_KAMATZ": '\u05B3',
                  "HIRIK": '\u05B4', "TZEIRE": '\u05B5', "SEGOL": '\u05B6', "PATAKH": '\u05B7', "KAMATZ": '\u05B8',
                  "HOLAM": '\u05B9', "KUBUTZ": '\u05BB', "SHURUK": '\u05BC', "METEG": '\u05BD',
                  "DAGESH_LETTER": '\u05bc',
                  "RAFE": '\u05BF', "SHIN_YEMANIT": '\u05c1', "SHIN_SMALIT": '\u05c2'}
    sign_2_name = {sign: name for name, sign in niqud_dict.items()}
    # RAFE is for acronyms
    niqud = [niqud_dict["RAFE"]] + [chr(c) for c in range(0x05b0, 0x05bc + 1)] + ['\u05b7']
    niqud_sin = [niqud_dict["RAFE"], niqud_dict["SHIN_YEMANIT"], niqud_dict["SHIN_SMALIT"]]
    dagesh = [niqud_dict["RAFE"], niqud_dict["DAGESH_LETTER"]]  # note that DAGESH and SHURUK are one and the same
    any_niqud = niqud[1:] + niqud_sin[1:] + dagesh
    niqud2id = {label: i for i, label in enumerate(any_niqud)}
    niqud2name = {label: i for i, label in enumerate(any_niqud)}
    id2niqud = {i: label for i, label in enumerate(any_niqud)}


class Letters:
    hebrew = [chr(c) for c in range(0x05d0, 0x05ea + 1)]


def read_data(filepath: str) -> List[Tuple[str, list]]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        file_data = file.read()
    data_list = file_data.split("\n")
    for sen in tqdm(data_list):
        if sen == "":
            continue
        # split_sentences = sen.split('\n')
        labels = []
        ktiv_male = ""
        index = 0
        sentance_length = len(sen)
        while index < sentance_length:
            label = []
            ktiv_male += sen[index]
            if sen[index] in Letters.hebrew:
                index += 1
                while index < sentance_length and sen[index] in Niqqud.any_niqud:
                    label.append(sen[index])
                    index += 1
            else:
                index += 1
            labels.append(label)

        data.append((ktiv_male, labels))

    return data


def read_data_folder(folder_path: str):
    all_files = glob2.glob(f'{folder_path}/**/*.txt', recursive=True)
    all_data = []
    if DEBUG_MODE:
        all_files = all_files[:2]
    for file in all_files:
        all_data.extend(read_data(file))
    return all_data


def prepare_data(data, tokenizer, label2id, max_length, batch_size=8):
    dataset = []
    for index in range(len(data)):
        sentence, label = data[index]
        sentence = sentence.replace("<e1>", "")
        sentence = sentence.replace("</e1>", "")
        sentence = sentence.replace("<e2>", "")
        sentence = sentence.replace("</e2>", "")
        encoded_sequence = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        dataset.append((encoded_sequence['input_ids'][0], encoded_sequence['attention_mask'][0], label2id[label]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader


def show_data(all_data):
    vowels = [vowel for _, label_list in all_data for vowels in label_list for vowel in vowels]
    unique_vowels, label_counts = np.unique(vowels, return_counts=True)
    unique_vowels_names = [Niqqud.sign_2_name[vowel] for vowel in unique_vowels]
    fig, ax = plt.subplots(figsize=(16, 6))

    bar_positions = np.arange(len(unique_vowels))
    bar_width = 0.15
    ax.bar(bar_positions, list(label_counts), bar_width)

    ax.set_title("Distribution of labels in each dataset")
    ax.set_xlabel('Vowels')
    ax.set_ylabel('Count')
    ax.legend(loc='right', bbox_to_anchor=(1, 0.85))
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(unique_vowels_names, rotation=30, ha='right', fontsize=8)

    plt.show()


def prepare_data(data, tokenizer, label2id, max_length, batch_size=8):
    dataset = []
    for index in range(len(data)):
        sentence, label = data[index]
        sentence = sentence.replace("<e1>", "")
        sentence = sentence.replace("</e1>", "")
        sentence = sentence.replace("<e2>", "")
        sentence = sentence.replace("</e2>", "")
        encoded_sequence = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        dataset.append((encoded_sequence['input_ids'][0], encoded_sequence['attention_mask'][0], label2id[label]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader


def calc_max_length(data):
    max_length = 0
    for s, _ in data:
        if len(s) > max_length:
            max_length = len(s)
    return max_length


def main():
    folder_path = r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized"  # Replace with the root path of the folder containing sub-folders with .txt files
    all_data = read_data_folder(folder_path)
    show_data(all_data)
    max_length = calc_max_length(all_data)
    # prepare_data(data, tokenizer, Niqqud.label2id, max_length, batch_size=8)


if __name__ == '__main__':
    main()
