import os
from typing import List, Tuple
import glob2
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset
from transformers import CanineTokenizer, TrainingArguments, Trainer

matplotlib.use('Tkagg')

DEBUG_MODE = True


class Nikud:
    """
    1456 HEBREW POINT SHEVA
    1457 HEBREW POINT HATAF SEGOL
    1458 HEBREW POINT HATAF PATAH
    1459 HEBREW POINT HATAF QAMATS
    1460 HEBREW POINT HIRIQ
    1461 HEBREW POINT TSERE
    1462 HEBREW POINT SEGOL
    1463 HEBREW POINT PATAH
    1464 HEBREW POINT QAMATS
    1465 HEBREW POINT HOLAM
    1466 HEBREW POINT HOLAM HASER FOR VAV     ***EXTENDED***
    1467 HEBREW POINT QUBUTS
    1468 HEBREW POINT DAGESH OR MAPIQ
    1469 HEBREW POINT METEG                   ***EXTENDED***
    1470 HEBREW PUNCTUATION MAQAF             ***EXTENDED***
    1471 HEBREW POINT RAFE                    ***EXTENDED***
    1472 HEBREW PUNCTUATION PASEQ             ***EXTENDED***
    1473 HEBREW POINT SHIN DOT
    1474 HEBREW POINT SIN DOT
    """
    nikud_dict = {'SHVA': 1456,
                  'REDUCED_SEGOL': 1457,
                  'REDUCED_PATAKH': 1458,
                  'REDUCED_KAMATZ': 1459,
                  'HIRIK': 1460,
                  'TZEIRE': 1461,
                  'SEGOL': 1462,
                  'PATAKH': 1463,
                  'KAMATZ': 1464,
                  'HOLAM': 1465,
                  'HOLAM HASER VAV': 1466,
                  'KUBUTZ': 1467,
                  'DAGESH OR SHURUK': 1468,
                  'METEG': 1469,
                  'PUNCTUATION MAQAF': 1470,
                  'RAFE': 1471,
                  'PUNCTUATION PASEQ': 1472,
                  'SHIN_YEMANIT': 1473,
                  'SHIN_SMALIT': 1474}

    sign_2_name = {sign: name for name, sign in nikud_dict.items()}
    nikud_sin = [nikud_dict["RAFE"], nikud_dict["SHIN_YEMANIT"], nikud_dict["SHIN_SMALIT"]]
    dagesh = [nikud_dict["RAFE"], nikud_dict['DAGESH OR SHURUK']]  # note that DAGESH and SHURUK are one and the same
    nikud = [nikud_dict["RAFE"]] + [v for k, v in nikud_dict.items() if ((v not in nikud_sin) or (v not in dagesh))]
    all_nikud_ord = {v for v in nikud_dict.values()}
    all_nikud_chr = {chr(v) for v in nikud_dict.values()}

    nikud_2_id = {label: (i + 1) for i, label in enumerate(all_nikud_ord)}
    id_2_nikud = {(i + 1): label for i, label in enumerate(all_nikud_ord)}

    DAGESH_LETTER = nikud_dict['DAGESH OR SHURUK']
    RAFE = nikud_dict['RAFE']


class Letters:
    hebrew = [chr(c) for c in range(0x05d0, 0x05ea + 1)]
    niqud = Nikud()
    VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + hebrew
    SPECIAL_TOKENS = ['H', 'O', '5']
    ENDINGS_TO_REGULAR = dict(zip('ךםןףץ', 'כמנפצ'))

    def normalize(self, letter):
        if letter in self.VALID_LETTERS: return letter
        if letter in self.ENDINGS_TO_REGULAR: return self.ENDINGS_TO_REGULAR[letter]
        if letter in ['\n', '\t']: return ' '
        if letter in ['־', '‒', '–', '—', '―', '−']: return '-'
        if letter == '[': return '('
        if letter == ']': return ')'
        if letter in ['´', '‘', '’']: return "'"
        if letter in ['“', '”', '״']: return '"'
        if letter.isdigit(): return '5'
        if letter == '…': return ','
        if letter in ['ײ', 'װ', 'ױ']: return 'H'
        return 'O'

    def can_dagesh(self, letter):
        return letter in ('בגדהוזטיכלמנספצקשת' + 'ךף')

    def can_sin(self, letter):
        return letter == 'ש'

    def can_niqqud(self, letter):
        return letter in ('אבגדהוזחטיכלמנסעפצקרשת' + 'ךן')

    def get_label_letter(self, letter, labels):
        dagesh = True if self.can_dagesh(letter) else False
        sin = True if self.can_sin(letter) else False
        niqqud = True if self.can_niqqud(letter) else False
        normalized = self.normalize(letter)
        i = 0
        if dagesh and ord(labels[0]) == Nikud.DAGESH_LETTER:
            dagesh = labels[0]
            i += 1
        else:
            dagesh = ''
        if sin and labels[i] in Nikud.nikud_sin:
            sin = labels[i]
            i += 1
        else:
            sin =''
        if labels[i] in Nikud.nikud:
            nikud = labels[i]
            i += 1
        if letter == 'ו' and dagesh == Nikud.DAGESH_LETTER and niqqud == Nikud.RAFE:
            dagesh = Nikud.RAFE
            niqqud = Nikud.DAGESH_LETTER


def text_contains_nikud(text):
    return len(set(text) & Nikud.all_nikud_chr) > 0


class NikudDataset(Dataset):

    def __init__(self, folder='../data/hebrew_diacritized', split=None, val_size=0.1):
        self.data = self.read_data_folder(folder)

    def read_data_folder(self, folder_path: str):
        all_files = glob2.glob(f'{folder_path}/**/*.txt', recursive=True)
        all_data = []
        if DEBUG_MODE:
            all_files = all_files[:2]
        for file in all_files:
            all_data.extend(self.read_data(file))
        return all_data

    def read_data(self, filepath: str) -> List[Tuple[str, list]]:
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            file_data = file.read()
        data_list = file_data.split("\n")
        for sen in tqdm(data_list):
            if sen == "" or not text_contains_nikud(sen):
                continue
            # split_sentences = sen.split('\n')
            labels = []
            text = ""
            index = 0
            sentance_length = len(sen)
            while index < sentance_length:
                label = []
                text += sen[index]
                if sen[index] in Letters.hebrew:
                    index += 1
                    while index < sentance_length and sen[index] in Nikud.any_niqud:
                        label.append(sen[index])
                        index += 1
                else:
                    index += 1
                print([Nikud.sign_2_name[s] for s in label])
                labels.append(label)

            data.append((text, labels))

        return data

    def show_data_labels(self):
        vowels = [vowel for _, label_list in self.data for vowels in label_list for vowel in vowels]
        unique_vowels, label_counts = np.unique(vowels, return_counts=True)
        unique_vowels_names = [Nikud.sign_2_name[vowel] for vowel in unique_vowels]
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

    def calc_max_length(self):
        max_length = 0
        for s, _ in self.data:
            if len(s) > max_length:
                max_length = len(s)
        return max_length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]


class NikudCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate(self, batch):
        sentances = []
        # male_texts = [male for haser, male in batch]
        # raw_targets = [haser_male2target(haser, male) for haser, male in batch]
        X = self.tokenizer(sentances, padding='longest', truncation=True, return_tensors='pt')
        LEN = len(X.input_ids[0])  # includes [CLS] & [SEP]

        def pad_target(tgt):
            N_PADDING = max(0, 1 + LEN - tgt.shape[0] - 2)
            return np.pad(tgt, ((1, N_PADDING), (0, 0)))[:LEN]

        # y = torch.tensor(np.stack([pad_target(t) for t in raw_targets])).float()
        #
        # return {**X, 'labels': y}


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


def main():
    # folder_path = r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized"  # Replace with the root path of the folder containing sub-folders with .txt files
    # all_data = read_data_folder(folder_path)
    dataset = NikudDataset()
    dataset.show_data_labels()
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    NikudCollator(tokenizer)
    prepare_data(dataset, tokenizer, Nikud.label2id, 0, batch_size=8)


if __name__ == '__main__':
    main()
