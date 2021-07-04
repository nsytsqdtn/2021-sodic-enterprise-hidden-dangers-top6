import pandas as pd
import tqdm.auto as tqdm
class Prepare_Data():
    def __init__(self,config):
        self.config = config
    def read_data(self,path):
        data = pd.read_csv(path)
        return data
    
    def deal_data(self,path,mode='train'):
        data = self.read_data(path)
        data['level_1'] = data['level_1'].astype(str)
        data['level_2'] = data['level_2'].astype(str)
        data['level_3'] = data['level_3'].astype(str)
        data['level_4'] = data['level_4'].astype(str)
        data['content'] = data['content'].astype(str)
        return data

    def data_to_json(self):
        train_data = self.deal_data(self.config.train_path,mode='train')
        test_data = self.deal_data(self.config.test_path,mode='test')
        train_dataset = []
        for i in tqdm.tqdm(range(len(train_data))):
            train_dict = {}
            train_dict['id'] = train_data.loc[i, 'id']
            train_dict['level_1'] = train_data.loc[i, 'level_1']
            train_dict['level_2'] = train_data.loc[i, 'level_2']
            train_dict['level_3'] = train_data.loc[i, 'level_3']
            train_dict['level_4'] = train_data.loc[i, 'level_4']
            train_dict['content'] = train_data.loc[i, 'content']
            train_dict['label'] = train_data.loc[i, 'label']
            train_dataset.append(train_dict)
        test_dataset = []
        for i in tqdm.tqdm(range(len(test_data))):
            test_dict = {}
            test_dict['id'] = test_data.loc[i, 'id']
            test_dict['level_1'] = test_data.loc[i, 'level_1']
            test_dict['level_2'] = test_data.loc[i, 'level_2']
            test_dict['level_3'] = test_data.loc[i, 'level_3']
            test_dict['level_4'] = test_data.loc[i, 'level_4']
            test_dict['content'] = test_data.loc[i, 'content']
            test_dict['label'] = ''
            test_dataset.append(test_dict)
        return train_data,test_data,train_dataset,test_dataset