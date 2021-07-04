import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, roc_auc_score

class SearchF1():
    def __init__(self):
        self.maxf1=0
        self.yuzhi=0
        
    def get_result(self, oof, mode='eval'):
        result = []
        for pred in oof:
            res = 1 if pred > self.yuzhi else 0
            result.append(res)
        if mode == 'eval':
            print('acc：',precision_score(self.label, result))
            print('recall：',recall_score(self.label, result))
            print('f1：',f1_score(self.label, result))
        return result

    def get_yuzhi(self,oof,label):
        self.oof = oof
        self.label = label
        for i in range(350,650):
            t=i/1000
            tem=[0 if i<t else 1 for i in self.oof]
            if f1_score(self.label,tem)>self.maxf1:
                self.maxf1=f1_score(self.label,tem)
                self.yuzhi=t
        print('best threshold：',self.yuzhi)

class MetricsFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.search = SearchF1()
        
    def forward(self,logits,labels=None,mode='eval'):
        if mode == 'eval':
            self.search.get_yuzhi(logits,labels)
            auc = roc_auc_score(labels,logits)
            print('auc:',auc)
            results = self.search.get_result(logits,mode)
            f1 = f1_score(labels, results)
            return f1
        else:
            results = self.search.get_result(logits,mode)
            return results

