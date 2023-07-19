import os
from typing import List, Tuple
import glob2
from tqdm import tqdm

class Niqqud:
    SHVA = '\u05B0'
    REDUCED_SEGOL = '\u05B1'
    REDUCED_PATAKH = '\u05B2'
    REDUCED_KAMATZ = '\u05B3'
    HIRIK = '\u05B4'
    TZEIRE = '\u05B5'
    SEGOL = '\u05B6'
    PATAKH = '\u05B7'
    KAMATZ = '\u05B8'
    HOLAM = '\u05B9'
    KUBUTZ = '\u05BB'
    SHURUK = '\u05BC'
    METEG = '\u05BD'
    DAGESH_LETTER = '\u05bc'
    RAFE = '\u05BF'
    SHIN_YEMANIT = '\u05c1'
    SHIN_SMALIT = '\u05c2'


HEBREW_LETTERS = [chr(c) for c in range(0x05d0, 0x05ea + 1)]

NIQQUD = [Niqqud.RAFE] + [chr(c) for c in range(0x05b0, 0x05bc + 1)] + ['\u05b7']
HEBREW_LETTERS = [chr(c) for c in range(0x05d0, 0x05ea + 1)]

NIQQUD = [Niqqud.RAFE] + [chr(c) for c in range(0x05b0, 0x05bc + 1)] + ['\u05b7']

HOLAM = Niqqud.HOLAM

NIQQUD_SIN = [Niqqud.RAFE, Niqqud.SHIN_YEMANIT, Niqqud.SHIN_SMALIT]  # RAFE is for acronyms
DAGESH = [Niqqud.RAFE, Niqqud.DAGESH_LETTER]  # note that DAGESH and SHURUK are one and the same

ANY_NIQQUD = [Niqqud.RAFE] + NIQQUD[1:] + NIQQUD_SIN[1:] + DAGESH[1:]


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
            if sen[index] in HEBREW_LETTERS:
                index += 1
                while index < sentance_length and sen[index] in ANY_NIQQUD:
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
    for file in all_files:
        all_data.extend(read_data(file))


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
    folder_path = r"C:\Users\adir\Desktop\studies\nlp\nlp-final-project\data\hebrew_diacritized"  # Replace with the root path of the folder containing sub-folders with .txt files
    read_data_folder(folder_path)


if __name__ == '__main__':
    main()
