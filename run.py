#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
train_data = pd.read_csv('../../raw_data/train.csv')
test_data = pd.read_csv('../../raw_data/test.csv')
train_data['pinjie'] = train_data['level_4'] + ' ' + train_data['content']
test_data['pinjie'] = test_data['level_4'] + ' ' + test_data['content']
# all_sentences = pd.concat([train_data['level_1'],train_data['level_2'],train_data['level_3'],train_data['level_4'],
#                            train_data['content'],train_data['pinjie'],test_data['level_1'],test_data['level_2'],test_data['level_3'],
#                           test_data['level_4'],test_data['content'],test_data['pinjie']]).reset_index(drop=True)
all_sentences = pd.concat([train_data['pinjie'],test_data['pinjie']]).reset_index(drop=True)
all_sentences = all_sentences.drop_duplicates().reset_index(drop=True)
print(all_sentences.shape)
all_sentences.to_csv('sentense.txt', index=False, header=False)


# import tokenizers
# filepath = 'sentense.txt'

# bwpt = tokenizers.BertWordPieceTokenizer()
# bwpt.train(
#     files=[filepath],
#     min_frequency=5,
#     limit_alphabet=1000
# )
# bwpt.save_model("../user_data/zjcmodel")


import torch
torch.cuda.is_available()
from transformers import BertConfig
from transformers import BertTokenizer
if not os.path.exists('../../user_data/model_wwm/'):
    os.makedirs('../../user_data/model_wwm/')
tokenizer = BertTokenizer.from_pretrained('../../user_data/model_wwm/')
from transformers import BertForMaskedLM
# from modeling_nezha import NeZhaForMaskedLM
model = BertForMaskedLM.from_pretrained('../../user_data/pre_model/chinese_wwm_ext_pytorch/')
# model.resize_token_embeddings(len(tokenizer))

from typing import Any, Dict, List, NewType, Optional, Tuple, Union
import random
import numpy as np
import os
seed = 777
def seed_everything(SEED):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE=='cuda':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed_everything(seed)
def tolist(x: Union[List[Any], torch.Tensor]):
    return x.tolist() if isinstance(x, torch.Tensor) else x

def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None,  mode='input'):
    if mode == 'input':
        examples = [e['input_ids'] for e in examples]
    else:
        examples = [torch.tensor(e) for e in examples]
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

class DataCollatorForLanguageModeling():
    def __init__(self,tokenizer,mlm=True,mlm_probability=0.15,pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = _collate_batch(examples, self.tokenizer)
        inputs, labels = self.mask_tokens(batch)
        return {"input_ids": inputs, "labels": labels}

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch_input = _collate_batch(examples, self.tokenizer)
        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _collate_batch(mask_labels, self.tokenizer, mode='mask')
        inputs, labels = self.mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

class DataCollatorForNgramsMask():
    def __init__(self,tokenizer,mlm=True,mlm_probability=0.15,ngrams=3,max_mask_num=5):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.ngrams = ngrams
        self.max_mask_num = max_mask_num
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch_input = _collate_batch(examples, self.tokenizer)
        mask_labels = []
        for b in batch_input:
            mask_labels.append(self.ngrams_mask(b))
        batch_mask = _collate_batch(mask_labels, self.tokenizer, mode='mask')
        inputs, labels = self.mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def ngrams_mask(self, inputs):
        ngrams = np.arange(1, self.ngrams + 1, dtype=np.int64) # [1 2 3]
        pvals = 1. / np.arange(1, self.ngrams + 1)
        pvals /= pvals.sum(keepdims=True)  # array([0.54545455, 0.27272727, 0.18181818])
        cand_indices = []
        for (i, token) in enumerate(inputs):
            if token == 2 or token == 3:
                continue
            elif token == 0:
                break
            cand_indices.append(i)
        num_to_mask = min(self.max_mask_num, max(1, int(round(len(inputs) * self.mlm_probability))))
        random.shuffle(cand_indices)
        masked_token_labels = []
        covered_indices = set()
        for index in cand_indices:
            n = np.random.choice(self.ngrams, p=pvals)
            if len(masked_token_labels) >= num_to_mask:
                break
            if index in covered_indices:
                continue
            if index < len(cand_indices) - (n - 1):
                for i in range(n):
                    ind = index + i
                    if ind in covered_indices:
                        continue
                    covered_indices.add(ind)
                    masked_token_labels.append({'index':ind, 'label':inputs[ind]})
        masked_token_labels = sorted(masked_token_labels, key=lambda x: x['index'])
        mask_indices = torch.tensor([p['index'] for p in masked_token_labels])
        mask_labels = torch.zeros(len(inputs)).scatter_(0, mask_indices,1).long()
        return mask_labels.tolist()

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = mask_labels
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
#
#
from transformers import LineByLineTextDataset
from transformers import TrainingArguments
from transformers import Trainer
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="sentense.txt",
    block_size=400,
)


# data_collator = DataCollatorForNgramsMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,drop_last=False,collate_fn=data_collator)

data_collator = DataCollatorForNgramsMask(tokenizer=tokenizer,mlm=True,mlm_probability=0.2,ngrams=5,max_mask_num=30)
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=32,
    save_steps=20_00,
    save_total_limit=2,
    prediction_loss_only=True,
    seed=seed,
    learning_rate=5e-5,
    weight_decay=1e-2,
    warmup_ratio=0.1
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


trainer.train()
trainer.save_model("../../user_data/model_wwm")


# In[2]:


import pandas as pd
import numpy as np
import collections
import re
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from transformers import BertForPreTraining, BertModel, BertTokenizer, AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
import warnings
import torch.nn as nn
import tqdm.auto as tqdm
import random
import gensim
import os
from torch import nn
import torch.nn.functional as F
from torch.optim import *
torch.set_printoptions(edgeitems=768)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)


# In[3]:


from utils.premodule import Seed_Everything
from training.train_logger import TrainLogger
from config import Config
from preprocessing.prepare_data import Prepare_Data
from preprocessing.dataloader_generater import Dataloader_Generater
from models.bert import Bert
from trainer import Trainer

class MyConfig(Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        # path
        self.name = 'wwm'
#         self.model_path = '../../pre_model/bert_base_chinese'
        self.model_path = '../../user_data/model_wwm'
        self.train_path = '../../raw_data/train.csv'
        self.test_path = '../../raw_data/test.csv'
        self.init_path()
        # param
        self.seed = 8912
        self.fold = 5
        self.batch_size = 32
        self.lr = 5e-5
        self.early_stop = 5
        self.save_model_epoch = 1
        self.weight_decay = 1e-2
        self.max_len = 300
        self.model_num = 1
        # model
        self.embedding_size = 768
        self.num_classes = 1
        self.dropout = 0.1
        self.epochs = 9
        
        # tirck param
        self.T_0 = 3
        self.T_mult = 2
        self.eta_min = 1e-7
#         self.warmup_radio = 0.1
        self.adversarial = 'fgm'
        self.fgm_epsilon = 0.1
        
config = MyConfig()
device = Seed_Everything(config.seed)
config.device = device
logger = TrainLogger(config.log_path)
train_data,test_data,train_dataset,test_dataset = Prepare_Data(config).data_to_json()


# In[4]:


def train_model(mode='debug'):
    if mode == 'debug':
#         dataloader_generater = Dataloader_Generater(config)
#         test_dataloader, test_torchdata = dataloader_generater.generate_loader(test_dataset, mode='test')
        trainer = Trainer(config,logger,train_data[:100],test_data[:100],train_dataset[:100],test_dataset[:100])
        trainer.train_method('debug')
        test_preds_merge = trainer.predict()
        return test_preds_merge
    elif mode == 'train':
        trainer = Trainer(config,logger,train_data,test_data,train_dataset,test_dataset)
        trainer.train_method()
    elif mode == 'predict':
        trainer = Trainer(config,logger,train_data,test_data,train_dataset,test_dataset)
        test_preds_merge = trainer.predict()
        return test_preds_merge
    elif mode == 'all':
        trainer = Trainer(config,logger,train_data,test_data,train_dataset,test_dataset)
        trainer.train_method()
        results,test_preds_merge = trainer.predict()
        return results,test_preds_merge


# In[5]:


results,test_preds_merge = train_model('all')


# In[6]:


test_data['label'] = results
test_data[['id','label']].to_csv('results.csv',index=False)
test_data['label'].value_counts()


# - 第1折中，best score: 0.9404761904761906
# - 第2折中，best score: 0.918032786885246
# - 第3折中，best score: 0.956
# - 第4折中，best score: 0.919626168224299
# - 第5折中，best score: 0.948339483394834
