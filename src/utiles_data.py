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
                  'KAMATZ_KATAN': 1479,
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

    def id_2_char(self, c, type):
        label = self.id_2_label[type][c]
        if label != "WITHOUT":
            return chr(self.id_2_label[type][c])
        return ""


class Letters:
    hebrew = [chr(c) for c in range(0x05d0, 0x05ea + 1)]
    VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + hebrew
    SPECIAL_TOKENS = ['H', 'O', '5', '1']
    ENDINGS_TO_REGULAR = dict(zip('ךםןףץ', 'כמנפצ'))
    vocab = VALID_LETTERS + SPECIAL_TOKENS
    vocab_size = len(vocab)


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
        if letter in ['־', '‒', '–', '—', '―', '−', '+']: return '-'
        if letter == '[': return '('
        if letter == ']': return ')'
        if letter in ['´', '‘', '’']: return "'"
        if letter in ['“', '”', '״']: return '"'
        if letter.isdigit():
            if int(letter) == 1:
                return '1'
            else:
                return '5'
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
    def __init__(self, tokenizer, folder='../data/hebrew_diacritized', logger=None, max_length=0):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = self.read_data_folder(folder, logger)
        self.prepered_data = None

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
            if "not_use" in file or "NakdanResults" in file:
                continue
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



    def split_data(self, folder_path: str, logger=None):#TODO: DELETE
        all_data = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.lower().endswith('.txt') and os.path.isfile(file_path):
                all_data.extend(self.read_data_split(file_path))
            elif os.path.isdir(file_path) and filename != ".git":
                self.split_data(file_path, logger)

        import random
        random.shuffle(all_data)

        if len(all_data) > 0:
            self.split_data_data(all_data, folder_path)

        return all_data

    def read_data_split(self, filepath: str) -> List[Tuple[str, list]]:#TODO: DELETE
        data_list = []
        with open(filepath, 'r', encoding='utf-8') as file:
            file_data = file.read()
        data_list = self.split_text(file_data)

        return data_list

    def split_data_data(self, data_list, filepath):#TODO: DELETE
        import math
        train_size = (int)(0.9 * len(data_list))
        dev_size = math.ceil(0.05 * len(data_list))
        dev_end_index = train_size + dev_size
        train_data = data_list[: train_size]
        dev_data = data_list[train_size : dev_end_index]
        test_data = data_list[dev_end_index :]

        x = "\\"
        self.write_list_to_text_file(train_data, filepath + f"\\{filepath.split(x)[-1]}.txt", "train")
        self.write_list_to_text_file(dev_data, filepath + f"\\{filepath.split(x)[-1]}.txt", "dev")
        self.write_list_to_text_file(test_data, filepath + f"\\{filepath.split(x)[-1]}.txt", "test")

    def write_list_to_text_file(self, data_list, file_path, type):#TODO: DELETE
        file_path = file_path.replace("hebrew_diacritized", type)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write("%s\n" % item)

    def delete_files(self, folder_path): #TODO: DELETE
        all_files = glob2.glob(f'{folder_path}/**/*.txt', recursive=True)

        for file_path in all_files:
            if "hebrew_diacritized" in file_path:
                continue
            if os.path.exists(file_path):
                os.remove(file_path)





    def split_text(self, file_data):
        file_data = file_data.replace(". ", ".\n")
        file_data = file_data.replace("? ", "?\n")
        file_data = file_data.replace("! ", "!\n")
        file_data = file_data.replace("” ", "”\n")
        file_data = file_data.replace("\" ", "\"\n")
        file_data = file_data.replace("\" ", "\"\n")
        file_data = file_data.replace("\t", "\n")
        data_list = file_data.split("\n")
        return data_list

    def show_data_labels(self, debug_folder=None):
        vowels = [Nikud.id_2_label["nikud"][label.nikud] for _, label_list in self.data for label in label_list if
                  label.nikud != -1]
        dageshs = [Nikud.id_2_label["dagesh"][label.dagesh] for _, label_list in self.data for label in label_list if
                   label.dagesh != -1]
        sin = [Nikud.id_2_label["sin"][label.sin] for _, label_list in self.data for label in label_list if
               label.sin != -1]
        vowels = vowels + dageshs + sin
        unique_vowels, label_counts = np.unique(vowels, return_counts=True)
        unique_vowels_names = [Nikud.sign_2_name[int(vowel)] for vowel in unique_vowels if vowel != 'WITHOUT'] + [
            "WITHOUT"]
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

    def calc_max_length(self, maximum=512):
        if self.max_length > maximum:
            self.max_length = maximum
        return self.max_length
        # max_length = 0
        # for s, _ in self.data:
        #     if len(s) > max_length:
        #         max_length = len(s)
        # self.max_length = max_length +1

    def prepare_data(self, name="train"):#, with_label=False):
        dataset = []
        for index, (sentence, label) in tqdm(enumerate(self.data), desc=f"prepare data {name}"):

            encoded_sequence = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            label_lists = [[letter.nikud, letter.dagesh, letter.sin] for letter in label]
            label = torch.tensor(
                [[Nikud.PAD, Nikud.PAD, Nikud.PAD]] + label_lists[:(self.max_length - 1)] + [
                    [Nikud.PAD, Nikud.PAD, Nikud.PAD] for i in
                    range(self.max_length - len(label) - 1)])

            dataset.append((encoded_sequence['input_ids'][0], encoded_sequence['attention_mask'][0], label))

        self.prepered_data = dataset

    def back_2_text(self, labels):
        nikud = Nikud()
        for indx_sentance, (input_ids, _) in enumerate(self.prepered_data):
            new_line = ""
            for indx_char, c in enumerate(input_ids):
                decode_char = self.tokenizer.decode(c)
                if decode_char not in self.tokenizer.all_special_tokens:
                    new_line += (decode_char + nikud.id_2_char(labels[indx_sentance, indx_char, 0], "nikud") +
                                 nikud.id_2_char(labels[indx_sentance, indx_char, 1], "dagesh") +
                                 nikud.id_2_char(labels[indx_sentance, indx_char, 2], "sin"))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
