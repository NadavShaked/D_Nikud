# general
import os.path
import random
from pathlib import Path
from typing import List, Tuple
from uuid import uuid1
import re
import glob2

# visual
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# ML
import numpy as np
import torch
from torch.utils.data import Dataset

from src.running_params import DEBUG_MODE, MAX_LENGTH_SEN

matplotlib.use('agg')
unique_key = str(uuid1())


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
    PAD_OR_IRRELEVANT = -1

    LEN_NIKUD = len(label_2_id["nikud"])
    LEN_DAGESH = len(label_2_id["dagesh"])
    LEN_SIN = len(label_2_id["sin"])

    def id_2_char(self, c, class_type):
        if c == -1:
            return ""

        label = self.id_2_label[class_type][c]
        if label != "WITHOUT":
            return chr(self.id_2_label[class_type][c])

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
        if letter in ['‒', '–', '—', '―', '−', '+']: return '-'
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
        dagesh_sin_nikud = [True if self.can_dagesh(self.letter) else False,
                            True if self.can_sin(self.letter) else False,
                            True if self.can_nikud(self.letter) else False]

        labels_ids = {"nikud": Nikud.PAD_OR_IRRELEVANT,
                      "dagesh": Nikud.PAD_OR_IRRELEVANT,
                      "sin": Nikud.PAD_OR_IRRELEVANT}

        normalized = self.normalize(self.letter)

        i = 0
        if Nikud.nikud_dict["PUNCTUATION MAQAF"] in labels:
            labels.remove(Nikud.nikud_dict["PUNCTUATION MAQAF"])
        if Nikud.nikud_dict["METEG"] in labels:
            labels.remove(Nikud.nikud_dict["METEG"])
        for index, (class_name, group) in enumerate(
                zip(["dagesh", "sin", "nikud"], [[Nikud.DAGESH_LETTER], Nikud.sin, Nikud.nikud])):
            # notice - order is important: dagesh then sin and then nikud
            if dagesh_sin_nikud[index]:
                if i < len(labels) and labels[i] in group:
                    labels_ids[class_name] = Nikud.label_2_id[class_name][labels[i]]
                    i += 1
                else:
                    labels_ids[class_name] = Nikud.label_2_id[class_name]["WITHOUT"]

        if np.array(dagesh_sin_nikud).all() and len(labels) == 3 and labels[0] in Nikud.sin:
            labels_ids["nikud"] = Nikud.label_2_id["nikud"][labels[2]]
            labels_ids["dagesh"] = Nikud.label_2_id["dagesh"][labels[1]]

        if self.can_sin(self.letter) and len(labels) == 2 and labels[1] == Nikud.DAGESH_LETTER:
            labels_ids["dagesh"] = Nikud.label_2_id["dagesh"][labels[1]]
            labels_ids["nikud"] = Nikud.label_2_id[class_name]["WITHOUT"]

        if self.letter == 'ו' and labels_ids["dagesh"] == Nikud.DAGESH_LETTER and labels_ids["nikud"] == \
                Nikud.label_2_id["nikud"]["WITHOUT"]:
            labels_ids["dagesh"] = Nikud.label_2_id["dagesh"]["WITHOUT"]
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


def combine_sentences(list_sentences, max_length=0, is_train=False):
    all_new_sentences = []
    new_sen = ""
    index = 0
    while index < len(list_sentences):
        sen = list_sentences[index]

        if not text_contains_nikud(sen) and ('------------------' in sen or sen == '\n'):
            if len(new_sen) > 0:
                all_new_sentences.append(new_sen)
                if not is_train:
                    all_new_sentences.append(sen)
                new_sen = ""
                index += 1
                continue

        if not text_contains_nikud(sen) and is_train:
            index += 1
            continue

        if len(sen) > max_length:
            update_sen = sen.replace(". ", f". {unique_key}")
            update_sen = update_sen.replace("? ", f"? {unique_key}")
            update_sen = update_sen.replace("! ", f"! {unique_key}")
            update_sen = update_sen.replace("” ", f"” {unique_key}")
            update_sen = update_sen.replace("\t", f"\t{unique_key}")
            part_sentence = update_sen.split(unique_key)

            good_parts = []
            for p in part_sentence:
                if len(p) < max_length:
                    good_parts.append(p)
                else:
                    prev = 0
                    while prev <= len(p):
                        part = p[prev:(prev + max_length)]
                        last_space = 0
                        if " " in part:
                            last_space = part[::-1].index(" ") + 1
                        next = prev + max_length - last_space
                        part = p[prev:next]
                        good_parts.append(part)
                        prev = next
            list_sentences = list_sentences[:index] + good_parts + list_sentences[index + 1:]
            continue
        if new_sen == "":
            new_sen = sen
        elif len(new_sen) + len(sen) < max_length:
            new_sen += sen
        else:
            all_new_sentences.append(new_sen)
            new_sen = sen

        index += 1
    if len(new_sen) > 0:
        all_new_sentences.append(new_sen)
    return all_new_sentences


class NikudDataset(Dataset):
    def __init__(self, tokenizer, folder=None, file=None, logger=None, max_length=0, is_train=False):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        if folder is not None:
            self.data, self.origin_data = self.read_data_folder(folder, logger)
        elif file is not None:
            self.data, self.origin_data = self.read_data(file, logger)
        self.prepered_data = None

    def read_data_folder(self, folder_path: str, logger=None):
        all_files = glob2.glob(f'{folder_path}/**/*.txt', recursive=True)
        msg = f"number of files: " + str(len(all_files))
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        all_data = []
        all_origin_data = []
        if DEBUG_MODE:
            all_files = all_files[0:2]
        for file in all_files:
            if "not_use" in file or "NakdanResults" in file:
                continue
            data, origin_data = self.read_data(file, logger)
            all_data.extend(data)
            all_origin_data.extend(origin_data)
        return all_data, all_origin_data

    def read_data(self, filepath: str, logger=None) -> List[Tuple[str, list]]:
        msg = f"read file: {filepath}"
        if logger:
            logger.debug(msg)
        else:
            print(msg)
        data = []
        orig_data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            file_data = file.read()
        data_list = self.split_text(file_data)

        for sen in tqdm(data_list, desc=f"Source: {os.path.basename(filepath)}"):
            if sen == "":
                continue

            labels = []
            text = ""
            text_org = ""
            index = 0
            sentence_length = len(sen)
            while index < sentence_length:
                if ord(sen[index]) == Nikud.nikud_dict['PUNCTUATION MAQAF'] or ord(sen[index]) == Nikud.nikud_dict[
                    'PUNCTUATION PASEQ'] or ord(sen[index]) == Nikud.nikud_dict['METEG']:
                    index += 1
                    continue

                label = []
                l = Letter(sen[index])

                assert l.letter not in Nikud.all_nikud_chr
                if sen[index] in Letters.hebrew:
                    index += 1
                    while index < sentence_length and ord(sen[index]) in Nikud.all_nikud_ord:
                        label.append(ord(sen[index]))
                        index += 1
                else:
                    index += 1

                l.get_label_letter(label)
                text += l.normalized
                text_org += l.letter
                labels.append(l)

            data.append((text, labels))
            orig_data.append(text_org)

        return data, orig_data

    def split_data(self, folder_path: str, logger=None, main_folder_name="hebrew_diacritized"):
        msg = f"prepare data in folder - {os.path.basename(folder_path)}"
        logger.debug(msg)
        for type_data in ["train", "dev", "test"]:
            folder_type = folder_path.replace(main_folder_name, type_data)
            create_missing_folders(folder_type)

        all_data = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.lower().endswith('.txt') and os.path.isfile(file_path):
                all_data.extend(self.read_data_split(file_path))
            elif os.path.isdir(file_path) and filename != ".git":
                self.split_data(file_path, logger)

        random.shuffle(all_data)

        if len(all_data) > 0:
            self.split_2_train_dev_test(all_data, folder_path)

        return all_data

    def read_data_split(self, filepath: str) -> List[Tuple[str, list]]:  # TODO: DELETE
        with open(filepath, 'r', encoding='utf-8') as file:
            file_data = file.read()
        data_list = self.split_text(file_data)

        return data_list

    def split_2_train_dev_test(self, data_list, filepath):  # TODO: DELETE
        import math
        train_size = (int)(0.9 * len(data_list))
        dev_size = math.ceil(0.05 * len(data_list))
        dev_end_index = train_size + dev_size
        train_data = data_list[: train_size]
        dev_data = data_list[train_size: dev_end_index]
        test_data = data_list[dev_end_index:]

        name_folder = os.path.basename(filepath)
        self.write_list_to_text_file(train_data, os.path.join(filepath, f"{name_folder}.txt"), "train")
        self.write_list_to_text_file(dev_data, os.path.join(filepath, f"{name_folder}.txt"), "dev")
        self.write_list_to_text_file(test_data, os.path.join(filepath, f"{name_folder}.txt"), "test")

    def write_list_to_text_file(self, data_list, file_path, type,
                                main_folder_name="hebrew_diacritized"):  # TODO: DELETE
        file_path = file_path.replace(main_folder_name, type)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(item + "\n------------------\n")

    def delete_files(self, folder_path):  # TODO: DELETE
        all_files = glob2.glob(f'{folder_path}/**/*.txt', recursive=True)

        for file_path in all_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def split_text(self, file_data):
        file_data = file_data.replace("\n", f"\n{unique_key}")
        data_list = file_data.split(unique_key)
        data_list = combine_sentences(data_list, is_train=self.is_train, max_length=MAX_LENGTH_SEN)
        return data_list

    def show_data_labels(self, plots_folder=None):
        nikud = [Nikud.id_2_label["nikud"][label.nikud] for _, label_list in self.data for label in label_list if
                  label.nikud != -1]
        dagesh = [Nikud.id_2_label["dagesh"][label.dagesh] for _, label_list in self.data for label in label_list if
                   label.dagesh != -1]
        sin = [Nikud.id_2_label["sin"][label.sin] for _, label_list in self.data for label in label_list if
               label.sin != -1]

        vowels = nikud + dagesh + sin
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

        if plots_folder is None:
            plt.show()
        else:
            plt.savefig(os.path.join(plots_folder, 'show_data_labels.jpg'))

    def calc_max_length(self, maximum=MAX_LENGTH_SEN):
        if self.max_length > maximum:
            self.max_length = maximum
        return self.max_length

    def prepare_data(self, name="train"):
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
                [[Nikud.PAD_OR_IRRELEVANT, Nikud.PAD_OR_IRRELEVANT, Nikud.PAD_OR_IRRELEVANT]] + label_lists[:(
                        self.max_length - 1)] + [
                    [Nikud.PAD_OR_IRRELEVANT, Nikud.PAD_OR_IRRELEVANT, Nikud.PAD_OR_IRRELEVANT] for i in
                    range(self.max_length - len(label) - 1)])

            dataset.append((encoded_sequence['input_ids'][0], encoded_sequence['attention_mask'][0], label))

        self.prepered_data = dataset

    def back_2_text(self, labels):
        nikud = Nikud()
        all_text = ""
        for indx_sentance, (input_ids, _, label) in enumerate(self.prepered_data):
            new_line = ""
            for indx_char, c in enumerate(self.origin_data[indx_sentance]):
                new_line += (c + nikud.id_2_char(labels[indx_sentance, indx_char + 1, 1], "dagesh") +
                             nikud.id_2_char(labels[indx_sentance, indx_char + 1, 2], "sin") +
                             nikud.id_2_char(labels[indx_sentance, indx_char + 1, 0], "nikud"))
            all_text += new_line
        return all_text

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]


def get_sub_folders_paths(main_folder):
    list_paths = []
    for filename in os.listdir(main_folder):
        path = os.path.join(main_folder, filename)
        if os.path.isdir(path) and filename != ".git":
            list_paths.append(path)
            list_paths.extend(get_sub_folders_paths(path))
    return list_paths

def create_missing_folders(folder_path):
    # Check if the folder doesn't exist and create it if needed
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def organize_data(main_folder, logger):
    x = NikudDataset(None)
    x.delete_files(os.path.join(Path(main_folder).parent, "train"))
    x.delete_files(os.path.join(Path(main_folder).parent, "dev"))
    x.delete_files(os.path.join(Path(main_folder).parent, "test"))
    x.split_data(main_folder, main_folder_name=os.path.basename(main_folder), logger=logger)

def info_folder(folder, num_files, num_hebrew_letters):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.lower().endswith('.txt') and os.path.isfile(file_path):
            num_files += 1
            dataset = NikudDataset(None, file=file_path)
            for line in dataset.data:
                for c in line[0]:
                    if c in Letters.hebrew:
                        num_hebrew_letters += 1

        elif os.path.isdir(file_path) and filename != ".git":
            sub_folder = file_path
            n1, n2 = info_folder(sub_folder, num_files, num_hebrew_letters)
            num_files += n1
            num_hebrew_letters += n2
    return num_files, num_hebrew_letters

def extract_text_to_compare_nakdimon(text):
    res = text.replace('|', '')
    res = res.replace(chr(Nikud.nikud_dict["KUBUTZ"]) + 'ו' + chr(Nikud.nikud_dict["METEG"]),
                      'ו' + chr(Nikud.nikud_dict['DAGESH OR SHURUK']))
    res = res.replace(chr(Nikud.nikud_dict["HOLAM"]) + 'ו' + chr(Nikud.nikud_dict["METEG"]),
                      'ו')
    res = res.replace("ו" + chr(Nikud.nikud_dict["HOLAM"]) + chr(Nikud.nikud_dict["KAMATZ"]),
                      'ו' + chr(Nikud.nikud_dict["KAMATZ"]))
    res = res.replace(chr(Nikud.nikud_dict["METEG"]), '')
    res = res.replace(chr(Nikud.nikud_dict["KAMATZ"]) + chr(Nikud.nikud_dict["HIRIK"]),
                      chr(Nikud.nikud_dict["KAMATZ"]) + 'י' + chr(Nikud.nikud_dict["HIRIK"]))
    res = res.replace(chr(Nikud.nikud_dict["PATAKH"]) + chr(Nikud.nikud_dict["HIRIK"]),
                      chr(Nikud.nikud_dict["PATAKH"]) + 'י' + chr(Nikud.nikud_dict["HIRIK"]))
    res = res.replace(chr(Nikud.nikud_dict["PUNCTUATION MAQAF"]), '')
    res = res.replace(chr(Nikud.nikud_dict["PUNCTUATION PASEQ"]), '')
    res = res.replace(chr(Nikud.nikud_dict["KAMATZ_KATAN"]), chr(Nikud.nikud_dict["KAMATZ"]))

    res = re.sub(chr(Nikud.nikud_dict["KUBUTZ"]) + 'ו' + '(?=[א-ת])', 'ו',
                 res)
    res = res.replace(chr(Nikud.nikud_dict["REDUCED_KAMATZ"]) + 'ו', 'ו')

    res = res.replace(chr(Nikud.nikud_dict["DAGESH OR SHURUK"]) * 2, chr(Nikud.nikud_dict["DAGESH OR SHURUK"]))
    res = res.replace('\u05be', '-')
    res = res.replace('יְהוָֹה', 'יהוה')

    return res
