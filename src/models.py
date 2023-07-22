from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig

# DL
import torch
import torch.nn as nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda'

def model(model="imvladikon/alephbertgimmel-base-512"):
    model = AutoModelForMaskedLM.from_pretrained("imvladikon/alephbertgimmel-base-512")
DMtokenizer = AutoTokenizer.from_pretrained("imvladikon/alephbertgimmel-base-512")

def get_parameters(params):
    top_layer_params = []
    for name, param in params:
        if name.startswith('classifier') or name.startswith('bert.pooler') or 'layer.11' in name:
            top_layer_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    return top_layer_params


class DiacritizationModel(nn.Module):
    def __init__(self, base_model_name):
        super(DiacritizationModel, self).__init__()
        config = AutoConfig.from_pretrained(base_model_name, num_labels=NIKUD_LEN)
        self.model = AutoModelForMaskedLM.from_pretrained(base_model_name, num_labels=num_labels).bert.to(
            DEVICE)
        self.model.resize_token_embeddings(len(DMtokenizer))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, e1_start, e2_start):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        batch_size = e1_start.size(0)
        e1_output = torch.stack([sequence_output[i][e1_start[i]] for i in range(batch_size)], dim=0)
        e2_output = torch.stack([sequence_output[i][e2_start[i]] for i in range(batch_size)], dim=0)
        e1_and_e2_output = torch.cat((e1_output, e2_output), dim=1)
        e1_and_e2_output = self.LayerNorm(e1_and_e2_output)

        output = self.classifier(e1_and_e2_output)
        return self.softmax(output)


model_DM = DiacritizationModel("imvladikon/alephbertgimmel-base-512").to(DEVICE)
all_model_params_MTB = model_DM.named_parameters()
top_layer_params = get_parameters(all_model_params_MTB)
optimizerDM = torch.optim.Adam(top_layer_params, lr=0.0001)
criterion = nn.CrossEntropyLoss().to(DEVICE)