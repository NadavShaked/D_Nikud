# general
import subprocess
import yaml

# ML
import torch.nn as nn
from transformers import AutoConfig, RobertaForMaskedLM, PretrainedConfig


class DNikudModel(nn.Module):
    def __init__(self, config, nikud_size, dagesh_size, sin_size, pretrain_model=None, device='cpu'):
        super(DNikudModel, self).__init__()

        if pretrain_model is not None:
            model_base = RobertaForMaskedLM.from_pretrained(pretrain_model).to(device)
        else:
            model_base = RobertaForMaskedLM(config=config).to(device)

        self.model = model_base.roberta
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.lstm1 = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=True, dropout=0.1, batch_first=True)
        self.lstm2 = nn.LSTM(2 * config.hidden_size, config.hidden_size, bidirectional=True, dropout=0.1, batch_first=True)
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.out_n = nn.Linear(config.hidden_size, nikud_size)
        self.out_d = nn.Linear(config.hidden_size, dagesh_size)
        self.out_s = nn.Linear(config.hidden_size, sin_size)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        lstm1, _ = self.lstm1(last_hidden_state)
        lstm2, _ = self.lstm2(lstm1)
        dense = self.dense(lstm2)

        nikud = self.out_n(dense)
        dagesh = self.out_d(dense)
        sin = self.out_s(dense)

        return nikud, dagesh, sin


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit_hash
    except subprocess.CalledProcessError:
        # This will be raised if you're not in a Git repository
        print("Not inside a Git repository!")
        return None


class ModelConfig(PretrainedConfig):
    def __init__(self, max_length=None, dict=None):
        super(ModelConfig, self).__init__()
        if dict is None:
            self.__dict__.update(AutoConfig.from_pretrained("tau/tavbert-he").__dict__)
            self.max_length = max_length
            self._commit_hash = get_git_commit_hash()
        else:
            self.__dict__.update(dict)

    def print(self):
        print(self.__dict__)

    def save_to_file(self, file_path):
        with open(file_path, "w") as yaml_file:
            yaml.dump(self.__dict__, yaml_file, default_flow_style=False)

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "r") as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
        return cls(dict=config_dict)
