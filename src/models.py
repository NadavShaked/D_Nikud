import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaForMaskedLM

from src.utiles_data import Nikud

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class RobertaWithoutLMHead(RobertaForMaskedLM):
    def __init__(self, config):
        super(RobertaWithoutLMHead, self).__init__(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        # Call the forward method of the base class (RobertaForMaskedLM)
        outputs = super(RobertaWithoutLMHead, self).forward(input_ids, attention_mask=attention_mask,
                                                            token_type_ids=token_type_ids,
                                                            position_ids=position_ids,
                                                            head_mask=head_mask, output_hidden_states=True
                                                            )

        # Exclude the lm_head's output from the outputs
        last_hidden_states = outputs.hidden_states[-1]

        return last_hidden_states


class DiacritizationModel(nn.Module):
    def __init__(self, base_model_name):
        super(DiacritizationModel, self).__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        self.model = RobertaWithoutLMHead.from_pretrained(base_model_name).to(
            DEVICE)
        # self.model.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.classifier_nikud = nn.Linear(config.hidden_size, Nikud.LEN_NIKUD)
        self.classifier_sin = nn.Linear(config.hidden_size, Nikud.LEN_SIN)
        self.classifier_dagesh = nn.Linear(config.hidden_size, Nikud.LEN_DAGESH)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
        normalized_hidden_states = self.LayerNorm(last_hidden_state)

        # Classifier for Nikud
        nikud_logits = self.classifier_nikud(normalized_hidden_states)
        nikud_probs = self.softmax(nikud_logits)

        # Classifier for Dagesh
        dagesh_logits = self.classifier_dagesh(normalized_hidden_states)
        dagesh_probs = self.softmax(dagesh_logits)

        # Classifier for Sin
        sin_logits = self.classifier_sin(normalized_hidden_states)
        sin_probs = self.softmax(sin_logits)

        # Return the probabilities for each diacritical mark
        return nikud_probs, dagesh_probs, sin_probs


