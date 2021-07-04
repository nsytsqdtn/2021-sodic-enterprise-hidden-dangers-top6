import torch
from transformers import BertTokenizer, AutoTokenizer
import tqdm.auto as tqdm
SEP_TOKEN_ID = 102
class Dataloader_Generater():
    def __init__(self,config):
        self.config = config
    
    class DataSet():
        def __init__(self, config, data, mode='train'):
            self.config = config
            self.mode = mode
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_path, cache_dir=None)
            self.dataset = self.get_data(data,self.mode)

        def get_data(self, data, mode):
            dataset = []
            s = []
            for data_li in tqdm.tqdm(data):
                level_1 = data_li['level_1']
                level_2 = data_li['level_2']
                level_3 = data_li['level_3']
                level_4 = data_li['level_4']
                content = data_li['content']
                token1 = self.tokenizer.tokenize(level_4)
                token2 = self.tokenizer.tokenize(content)
                if len(token1)<len(token2):
                    if len(token1) > self.config.max_len//2:
                        tmp_len = len(token1) - self.config.max_len//2
                    else:
                        tmp_len = 0
                    token1 = token1[tmp_len:]
                    token2 = token2[:self.config.max_len-len(token1)-3]
                else:
                    token2 = token2[:self.config.max_len//2]
                    token1 = token1[-(self.config.max_len-len(token2)-3):]
                token_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']+token1+['[SEP]']+token2+['[SEP]'])
                if len(token_ids) < self.config.max_len:
                    token_ids += [0] * (self.config.max_len - len(token_ids))
                else:
                    token_ids = token_ids[:self.config.max_len]
                s.append(max(token_ids))
                label = data_li['label']
                dataset_dict = {'token_ids':token_ids, 'label':label}
                dataset.append(dataset_dict)
            return dataset

        def __len__(self):
            return len(self.dataset)
        
        def get_seg_ids(self, ids):
            seg_ids = torch.zeros_like(ids)
            seg_idx = 0
            for i, e in enumerate(ids):
                seg_ids[i] = seg_idx
                if e == SEP_TOKEN_ID:
                    seg_idx += 1
            max_idx = torch.nonzero(seg_ids == seg_idx)
            seg_ids[max_idx] = 0
            return seg_ids
        def __getitem__(self, idx):
            data = self.dataset[idx]
            token_ids = torch.tensor(data['token_ids'])
            seg_ids = self.get_seg_ids(token_ids)
            if self.mode=='test':
                return token_ids, seg_ids
            label = torch.tensor(data['label'])
            return token_ids, seg_ids, label
    
    def get_dataloader(self, data, mode):
        torchdata = self.DataSet(self.config,data,mode=mode)
        if mode == 'train':
            dataloader = torch.utils.data.DataLoader(torchdata, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, drop_last=True)
        elif mode == 'test':
            dataloader = torch.utils.data.DataLoader(torchdata, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, drop_last=False)
        elif mode == 'eval':
            dataloader = torch.utils.data.DataLoader(torchdata, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, drop_last=False)
        return dataloader, torchdata
    
    def generate_loader(self,data,mode):
        dataloader, torchdata = self.get_dataloader(data,mode=mode)
        return dataloader, torchdata