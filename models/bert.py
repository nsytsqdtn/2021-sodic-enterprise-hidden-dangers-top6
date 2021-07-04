import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertForPreTraining, BertModel
class Bert(nn.Module):
    def __init__(self,config):
        super(Bert, self).__init__()
        self.config = config
        self.embedding_size = self.config.embedding_size
        self.num_classes = self.config.num_classes
        self.bert = BertModel.from_pretrained(self.config.model_path,return_dict=True,output_hidden_states=True)
        self.fc = nn.Linear(self.embedding_size*2, self.num_classes)
        self.dropout = nn.Dropout(self.config.dropout)
    def forward(self, token_ids, seg_ids, label=None):
        attention_mask = (token_ids > 0)
        bert_out = self.bert(input_ids=token_ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        x1 = torch.mean(bert_out[2][-1],dim=1)
        x2 = torch.mean(bert_out[2][-2],dim=1)
#         x3 = torch.mean(bert_out[2][-3],dim=1)
#         x4 = torch.mean(bert_out[2][-4],dim=1)
        out_new = torch.cat((x1,x2),dim=-1)
        out = self.dropout(self.fc(out_new))
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1,1).float(), label.view(-1,1).float())
            return loss
        else:
            return F.sigmoid(out)