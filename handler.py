from typing import Dict, List, Any
from transformers import AutoConfig, AutoTokenizer
from src.models import DNikudModel, ModelConfig
from src.running_params import BATCH_SIZE, MAX_LENGTH_SEN
from src.utiles_data import Nikud, NikudDataset
from src.models_utils import predict_single, predict
import torch
import os
from tqdm import tqdm


class EndpointHandler:
    def __init__(self, path=""):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("tau/tavbert-he")
        dir_model_config = os.path.join("models", "config.yml")
        self.config = ModelConfig.load_from_file(dir_model_config)
        self.model = DNikudModel(
            self.config,
            len(Nikud.label_2_id["nikud"]),
            len(Nikud.label_2_id["dagesh"]),
            len(Nikud.label_2_id["sin"]),
            device=self.DEVICE,
        ).to(self.DEVICE)
        state_dict_model = self.model.state_dict()
        state_dict_model.update(torch.load("./models/Dnikud_best_model.pth"))
        self.model.load_state_dict(state_dict_model)
        self.max_length = MAX_LENGTH_SEN

    def back_2_text(self, labels, text):
        nikud = Nikud()
        new_line = ""

        for indx_char, c in enumerate(text):
            new_line += (
                c
                + nikud.id_2_char(labels[indx_char][1][1], "dagesh")
                + nikud.id_2_char(labels[indx_char][1][2], "sin")
                + nikud.id_2_char(labels[indx_char][1][0], "nikud")
            )
            print(indx_char, c)
        print(labels)
        return new_line

    def prepare_data(self, data, name="train"):
        print("Data = ", data)
        dataset = []
        for index, (sentence, label) in tqdm(
            enumerate(data), desc=f"Prepare data {name}"
        ):
            encoded_sequence = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            label_lists = [
                [letter.nikud, letter.dagesh, letter.sin] for letter in label
            ]
            label = torch.tensor(
                [
                    [
                        Nikud.PAD_OR_IRRELEVANT,
                        Nikud.PAD_OR_IRRELEVANT,
                        Nikud.PAD_OR_IRRELEVANT,
                    ]
                ]
                + label_lists[: (self.max_length - 1)]
                + [
                    [
                        Nikud.PAD_OR_IRRELEVANT,
                        Nikud.PAD_OR_IRRELEVANT,
                        Nikud.PAD_OR_IRRELEVANT,
                    ]
                    for i in range(self.max_length - len(label) - 1)
                ]
            )

            dataset.append(
                (
                    encoded_sequence["input_ids"][0],
                    encoded_sequence["attention_mask"][0],
                    label,
                )
            )

        self.prepered_data = dataset

    def predict_single_text(
        self,
        text,
    ):
        dataset = NikudDataset(tokenizer=self.tokenizer, max_length=MAX_LENGTH_SEN)
        data, orig_data = dataset.read_single_text(text)
        print("data", data, len(data))
        dataset.prepare_data(name="inference")
        mtb_prediction_dl = torch.utils.data.DataLoader(
            dataset.prepered_data, batch_size=BATCH_SIZE
        )
        # print("dataset", dataset, len(dataset))
        # data = self.tokenizer(text, return_tensors="pt")
        all_labels = predict(self.model, mtb_prediction_dl, self.DEVICE)
        text_data_with_labels = dataset.back_2_text(labels=all_labels)
        # all_labels = predict_single(self.model, dataset, self.DEVICE)
        return text_data_with_labels

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        data args:
        """

        # get inputs
        inputs = data.pop("text", data)

        # run normal prediction
        prediction = self.predict_single_text(inputs)

        # result = []
        # for pred in prediction:
        #     result.append(self.back_2_text(pred, inputs))
        # result = self.back_2_text(prediction, inputs)
        return prediction
