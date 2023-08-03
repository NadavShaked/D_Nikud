import os.path
from typing import List, Tuple

import glob2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.running_params import DEBUG_MODE

matplotlib.use('Tkagg')

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
    sin = [nikud_dict["RAFE"], nikud_dict["SHIN_YEMANIT"], nikud_dict["SHIN_SMALIT"]]
    dagesh = [nikud_dict["RAFE"], nikud_dict['DAGESH OR SHURUK']]  # note that DAGESH and SHURUK are one and the same
    nikud = []
    for v in nikud_dict.values():
        if v not in sin:
            nikud.append(v)
    all_nikud_ord = {v for v in nikud_dict.values()}
    all_nikud_chr = {chr(v) for v in nikud_dict.values()}

    label_2_id = {"nikud": {label: i for i, label in enumerate(nikud + ["WITHOUT"])},
                  "dagesh": {label: i for i, label in enumerate(dagesh + ["WITHOUT"])},
                  "sin": {label: i for i, label in enumerate(sin + ["WITHOUT"])}}
    id_2_label = {"nikud": {i: label for i, label in enumerate(nikud + ["WITHOUT"])},
                  "dagesh": {i: label for i, label in enumerate(dagesh + ["WITHOUT"])},
                  "sin": {i: label for i, label in enumerate(sin + ["WITHOUT"])}}

    DAGESH_LETTER = nikud_dict['DAGESH OR SHURUK']
    RAFE = nikud_dict['RAFE']
    PAD = -1
    IRRELEVANT = PAD

    LEN_NIKUD = len(label_2_id["nikud"])
    LEN_DAGESH = len(label_2_id["dagesh"])
    LEN_SIN = len(label_2_id["sin"])


class Letters:
    hebrew = [chr(c) for c in range(0x05d0, 0x05ea + 1)]
    VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + hebrew
    SPECIAL_TOKENS = ['H', 'O', '5']
    ENDINGS_TO_REGULAR = dict(zip('ךםןףץ', 'כמנפצ'))


class Letter:
    def __init__(self, letter):
        self.letter = letter
        self.normalized = None
        self.dagesh = None
        self.sin = None
        self.nikud = None

    def normalize(self, letter):
        if letter in Letters.VALID_LETTERS: return letter
        if letter in Letters.ENDINGS_TO_REGULAR: return Letters.ENDINGS_TO_REGULAR[letter]
        if letter in ['\n', '\t']: return ' '
        if letter in ['־', '‒', '–', '—', '―', '−']: return '-'
        if letter == '[': return '('
        if letter == ']': return ')'
        if letter in ['´', '‘', '’']: return "'"
        if letter in ['“', '”', '״']: return '"'
        if letter.isdigit() and int(letter) != 1: return '5'
        if letter == '…': return ','
        if letter in ['ײ', 'װ', 'ױ']: return 'H'
        return 'O'

    def can_dagesh(self, letter):
        return letter in ('בגדהוזטיכלמנספצקשת' + 'ךף')

    def can_sin(self, letter):
        return letter == 'ש'

    def can_nikud(self, letter):
        return letter in ('אבגדהוזחטיכלמנסעפצקרשת' + 'ךן')

    def get_label_letter(self, labels):
        # todo - consider reorgenize func
        dagesh_sin_nikud = [True if self.can_dagesh(self.letter) else False,
                            True if self.can_sin(self.letter) else False,
                            True if self.can_nikud(self.letter) else False]

        labels_ids = {"nikud": Nikud.IRRELEVANT,
                  "dagesh": Nikud.IRRELEVANT,
                  "sin": Nikud.IRRELEVANT}

        normalized = self.normalize(self.letter)
        i = 0
        for index, (name_class, group) in enumerate(
                zip(["dagesh", "sin", "nikud"], [[Nikud.DAGESH_LETTER], Nikud.sin, Nikud.nikud])):
            # notice - order is important : dagesh then sin and then nikud
            if dagesh_sin_nikud[index]:
                if i < len(labels) and labels[i] in group:
                    labels_ids[name_class] = Nikud.label_2_id[name_class][labels[i]]
                    i += 1
                else:
                    labels_ids[name_class] = Nikud.label_2_id[name_class]["WITHOUT"]

        if self.letter == 'ו' and labels_ids["dagesh"] == Nikud.DAGESH_LETTER and labels_ids["nikud"] == \
                Nikud.label_2_id["nikud"]["WITHOUT"]:
            labels_ids["dagesh"] = Nikud.RAFE
            labels_ids["nikud"] = Nikud.DAGESH_LETTER

        self.normalized = normalized
        self.dagesh = labels_ids["dagesh"]
        self.sin = labels_ids["sin"]
        self.nikud = labels_ids["nikud"]

    def name_of(self, letter):
        if 'א' <= letter <= 'ת':
            return letter
        if letter == Nikud.DAGESH_LETTER: return 'דגש\שורוק'
        if letter == Nikud.KAMATZ: return 'קמץ'
        if letter == Nikud.PATAKH: return 'פתח'
        if letter == Nikud.TZEIRE: return 'צירה'
        if letter == Nikud.SEGOL: return 'סגול'
        if letter == Nikud.SHVA: return 'שוא'
        if letter == Nikud.HOLAM: return 'חולם'
        if letter == Nikud.KUBUTZ: return 'קובוץ'
        if letter == Nikud.HIRIK: return 'חיריק'
        if letter == Nikud.REDUCED_KAMATZ: return 'חטף-קמץ'
        if letter == Nikud.REDUCED_PATAKH: return 'חטף-פתח'
        if letter == Nikud.REDUCED_SEGOL: return 'חטף-סגול'
        if letter == Nikud.SHIN_SMALIT: return 'שין-שמאלית'
        if letter == Nikud.SHIN_YEMANIT: return 'שין-ימנית'
        if letter.isprintable():
            return letter
        return "לא ידוע ({})".format(hex(ord(letter)))


def text_contains_nikud(text):
    return len(set(text) & Nikud.all_nikud_chr) > 0


class NikudDataset(Dataset):

    def __init__(self, folder='../data/hebrew_diacritized', logger=None):
        self.max_length = 0
        self.data = self.read_data_folder(folder, logger)

    def read_data_folder(self, folder_path: str, logger=None):
        all_files = glob2.glob(f'{folder_path}/**/*.txt', recursive=True)
        msg = f"number of files: " + str(len(all_files))
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        all_data = []
        if DEBUG_MODE:
            all_files = all_files[2:4]
            # all_files = [os.path.join(folder_path, "WikipediaHebrewWithVocalization-WithMetegToMarkMatresLectionis.txt")]
        for file in all_files:
            if "not_use" not in file:
                all_data.extend(self.read_data(file))
        return all_data

    def read_data(self, filepath: str) -> List[Tuple[str, list]]:
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            file_data = file.read()
        data_list = self.split_text(file_data)

        for sen in tqdm(data_list, desc=f"Source: {os.path.basename(filepath)}"):
            if sen == "" or not text_contains_nikud(sen):  # todo- mabye add check for every word
                continue
            length_max = len(sen)
            if length_max > self.max_length:
                self.max_length = length_max
            # split_sentences = sen.split('\n')
            labels = []
            text = ""
            index = 0
            sentance_length = len(sen)
            while index < sentance_length:
                label = []
                l = Letter(sen[index])
                if sen[index] in Letters.hebrew:
                    index += 1
                    while index < sentance_length and ord(sen[index]) in Nikud.all_nikud_ord:
                        label.append(ord(sen[index]))
                        index += 1
                else:
                    index += 1
                l.get_label_letter(label)
                text += l.normalized
                labels.append(l)

            data.append((text, labels))

        return data

    def split_text(self, file_data):
        file_data = file_data.replace(". ", ".\n")
        file_data = file_data.replace("? ", "?\n")
        file_data = file_data.replace("! ", "!\n")
        file_data = file_data.replace("” ", "”\n")
        file_data = file_data.replace("\" ", "\"\n")
        file_data = file_data.replace("\" ", "\"\n")
        file_data = file_data.replace("\t", "\n")
        data_list = file_data.split("\n")
        return  data_list


    def show_data_labels(self, debug_folder=None):
        vowels = [Nikud.id_2_label["nikud"][label.nikud] for _, label_list in self.data for label in label_list if label.nikud != -1]
        dageshs = [Nikud.id_2_label["dagesh"][label.dagesh] for _, label_list in self.data for label in label_list if label.dagesh != -1]
        sin = [Nikud.id_2_label["sin"][label.sin] for _, label_list in self.data for label in label_list if label.sin != -1]
        vowels = vowels + dageshs + sin
        unique_vowels, label_counts = np.unique(vowels, return_counts=True)
        unique_vowels_names = [Nikud.sign_2_name[int(vowel)] for vowel in unique_vowels if vowel!='WITHOUT'] + ["WITHOUT"]
        fig, ax = plt.subplots(figsize=(16, 6))

        bar_positions = np.arange(len(unique_vowels))
        bar_width = 0.15
        ax.bar(bar_positions, list(label_counts), bar_width)

        ax.set_title("Distribution of Vowels in dataset")
        ax.set_xlabel('Vowels')
        ax.set_ylabel('Count')
        ax.legend(loc='right', bbox_to_anchor=(1, 0.85))
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(unique_vowels_names, rotation=30, ha='right', fontsize=8)

        if debug_folder is None:
            plt.show()
        else:
            plt.savefig(os.path.join(debug_folder, 'show_data_labels.jpg'))

    def calc_max_length(self, maximum=1000):
        if self.max_length > maximum:
            self.max_length = maximum
        return self.max_length
        # max_length = 0
        # for s, _ in self.data:
        #     if len(s) > max_length:
        #         max_length = len(s)
        # self.max_length = max_length +1

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


def prepare_data(data, tokenizer, max_length, batch_size=8, name="train"):
    dataset = []
    for index, (sentence, label) in tqdm(enumerate(data), desc=f"prepare data {name}"):
        encoded_sequence = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        label_lists = [[letter.nikud, letter.dagesh, letter.sin] for letter in label]
        label = torch.tensor(
            [[Nikud.PAD, Nikud.PAD, Nikud.PAD]] + label_lists[:(max_length-1)] + [[Nikud.PAD, Nikud.PAD, Nikud.PAD] for i in
                                                                 range(max_length - len(label) - 1)])

        dataset.append((encoded_sequence['input_ids'][0], encoded_sequence['attention_mask'][0], label))


    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader


def main():
    # folder_path = r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized"  # Replace with the root path of the folder containing sub-folders with .txt files
    # all_data = read_data_folder(folder_path)
    dataset = NikudDataset()


if __name__ == '__main__':
    main()
