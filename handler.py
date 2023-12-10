from typing import Dict, List, Any
from transformers import AutoConfig, AutoTokenizer
from src.models import DNikudModel, ModelConfig
from src.running_params import BATCH_SIZE, MAX_LENGTH_SEN
from src.utiles_data import Nikud
from src.models_utils import predict_single
import torch
import os


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

    def back_2_text(self, labels, text):
        nikud = Nikud()
        new_line = ""
        for indx_char, c in enumerate(text):
            new_line += (
                c
                + nikud.id_2_char(labels[0][1][1], "dagesh")
                + nikud.id_2_char(labels[0][1][2], "sin")
                + nikud.id_2_char(labels[0][1][0], "nikud")
            )
            print(indx_char, c)
        print(labels)
        return new_line

    def predict_single_text(
        self,
        text,
    ):
        data = self.tokenizer(text, return_tensors="pt")
        all_labels = predict_single(self.model, data, self.DEVICE)
        return all_labels

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
        result = self.back_2_text(prediction, inputs)
        return result
