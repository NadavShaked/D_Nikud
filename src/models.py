import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaForMaskedLM, RobertaModel

from src.utiles_data import Nikud

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# class RobertaWithoutLMHead(RobertaForMaskedLM):
#     def __init__(self, config):
#         super(RobertaWithoutLMHead, self).__init__(config)
#
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
#         # Call the forward method of the base class (RobertaForMaskedLM)
#         outputs = super(RobertaWithoutLMHead, self).forward(input_ids, attention_mask=attention_mask,
#                                                             token_type_ids=token_type_ids,
#                                                             position_ids=position_ids,
#                                                             head_mask=head_mask, output_hidden_states=True
#                                                             )
#
#         # Exclude the lm_head's output from the outputs
#         last_hidden_states = outputs.hidden_states[-1]
#
#         return last_hidden_states


class DiacritizationModel(nn.Module):
    def __init__(self, base_model_name, nikud_size, dagesh_size, sin_size):
        super(DiacritizationModel, self).__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        tav_bert = RobertaForMaskedLM.from_pretrained(base_model_name).to(
            DEVICE)
        self.model = tav_bert.roberta
        # self.model.load_state_dict(tav_bert.roberta.state_dict())
        # self.model.lm_head = None
        # self.model.roberta = list(tav_bert.children())[:-1][0]
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        # super(DiacritizationModel, self).__init__()
        # config = AutoConfig.from_pretrained(base_model_name)
        # self.model = RobertaWithoutLMHead.from_pretrained(base_model_name).to(
        #     DEVICE)
        # for name, param in self.model.named_parameters():
        #     param.requires_grad = False

        self.lstm1 = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=True, dropout=0.1, batch_first=True)  # num_layers=1,
        self.lstm2 = nn.LSTM(2 * config.hidden_size, config.hidden_size, bidirectional=True, dropout=0.1, batch_first=True)
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.out_n = nn.Linear(config.hidden_size, nikud_size)
        self.out_d = nn.Linear(config.hidden_size, dagesh_size)
        self.out_s = nn.Linear(config.hidden_size, sin_size)


    def forward(self, input_ids, attention_mask, only_nikud=False):
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        lstm1, y1 = self.lstm1(last_hidden_state)
        lstm2, y2 = self.lstm2(lstm1)
        dense = self.dense(lstm2)
        nikud = self.out_n(dense)

        if not only_nikud:
            dagesh = self.out_d(dense)
            sin = self.out_s(dense)
        else:
            dagesh, sin = None, None
        return nikud, dagesh, sin


# import torch
# from torch import nn

# class BaseModel(nn.Module):
#     def __init__(self, units, vocab_size):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, units, padding_idx=0)
#         self.lstm1 = nn.LSTM(units, units, bidirectional=True, dropout=0.1, batch_first=True)
#         self.lstm2 = nn.LSTM(2*units, units, bidirectional=True, dropout=0.1, batch_first=True)
#
#     def forward(self, x):
#         x = self.embed(x)
#         x, _ = self.lstm1(x)
#         x = (x[:, :, :units] + x[:, :, units:])
#         x, _ = self.lstm2(x)
#         x = (x[:, :, :units] + x[:, :, units:])
#         return x
import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, units, vocab_size, nikud_size, dagesh_size, sin_size):
        super().__init__()
        self.units = units
        self.embed = nn.Embedding(vocab_size, units, padding_idx=0)
        self.lstm1 = nn.LSTM(units, units, bidirectional=True, dropout=0.1, batch_first=True)  # num_layers=1,
        self.lstm2 = nn.LSTM(2 * units, units, bidirectional=True, dropout=0.1, batch_first=True)
        self.dense = nn.Linear(2 * units, units)
        self.out_n = nn.Linear(units, nikud_size)
        self.out_d = nn.Linear(units, dagesh_size)
        self.out_s = nn.Linear(units, sin_size)

    def forward(self, x):
        # h0 = torch.zeros(2, x.size(0), self.units).to(DEVICE)
        # c0 = torch.zeros(2, x.size(0), self.units).to(DEVICE)
        embeding = self.embed(x)
        lstm1, y1 = self.lstm1(embeding)
        # lstm1_combine = (lstm1[:, :, :self.units] + lstm1[:, :, self.units:])
        lstm2, y2 = self.lstm2(lstm1)
        # lstm2_combine = (lstm2[:, :, :self.units] + lstm2[:, :, self.units:])
        dense = self.dense(lstm2)
        nikud = self.out_n(dense)
        dagesh = self.out_d(dense)
        sin = self.out_s(dense)
        return nikud, dagesh, sin

class BaseModel(nn.Module):
    def __init__(self, units, vocab_size, nikud_size, dagesh_size, sin_size):
        super().__init__()
        self.units = units
        self.embed = nn.Embedding(vocab_size, units, padding_idx=0)
        self.lstm1 = nn.LSTM(units, units, bidirectional=True, dropout=0.1, batch_first=True)  # num_layers=1,
        self.lstm2 = nn.LSTM(2 * units, units, bidirectional=True, dropout=0.1, batch_first=True)
        self.dense = nn.Linear(2 * units, units)
        self.out_n = nn.Linear(units, nikud_size)
        self.out_d = nn.Linear(units, dagesh_size)
        self.out_s = nn.Linear(units, sin_size)

    def forward(self, x):
        # h0 = torch.zeros(2, x.size(0), self.units).to(DEVICE)
        # c0 = torch.zeros(2, x.size(0), self.units).to(DEVICE)
        embeding = self.embed(x)
        lstm1, y1 = self.lstm1(embeding)
        # lstm1_combine = (lstm1[:, :, :self.units] + lstm1[:, :, self.units:])
        lstm2, y2 = self.lstm2(lstm1)
        # lstm2_combine = (lstm2[:, :, :self.units] + lstm2[:, :, self.units:])
        dense = self.dense(lstm2)
        nikud = self.out_n(dense)
        dagesh = self.out_d(dense)
        sin = self.out_s(dense)
        return nikud, dagesh, sin


class CharClassifierTransformer(nn.Module):
    def __init__(self, vocab_size, nikud_size, dagesh_size, sin_size, d_model=64, nhead=1, num_encoder_layers=2):
        super(CharClassifierTransformer, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Transformer layer
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers)

        # Classification layer
        self.classifier_nikud = nn.Linear(d_model, nikud_size)
        self.classifier_sin = nn.Linear(d_model, dagesh_size)
        self.classifier_dagesh = nn.Linear(d_model, sin_size)

    def forward(self, x):
        embedding = self.embedding(x)
        sequence_transformer_output = self.transformer(embedding, embedding)
        nikud_logits = self.classifier_nikud(sequence_transformer_output)
        sin_logits = self.classifier_sin(sequence_transformer_output)
        dagesh_logits = self.classifier_dagesh(sequence_transformer_output)
        return nikud_logits, sin_logits, dagesh_logits
