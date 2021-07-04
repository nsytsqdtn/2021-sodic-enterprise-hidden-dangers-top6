from torch import nn
import torch.nn.functional as F
from torch.optim import *
import torch
from sklearn.model_selection import KFold
import numpy as np
from metrics.metric import MetricsFunc
from training.adversarial import FGM
import tqdm.auto as tqdm
from sklearn.metrics import roc_auc_score
from models.bert import Bert
from preprocessing.dataloader_generater import Dataloader_Generater
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
class Trainer():
    def __init__(self,config,logger,train_data,test_data,train_dataset,test_dataset):
        # trainer conifg
        self.config = config
        self.use_sheduler = False
        self.logger = logger
        # training init
        self.dataloader_generater = Dataloader_Generater(self.config)
        self.train_data = train_data
        self.test_data = test_data
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_dataloader, self.test_torchdata = self.dataloader_generater.generate_loader(self.test_dataset, mode='test')
        self.train_dataset = train_dataset
        self.eval = self.config.eval
        self.use_scheduler = None
        
        self.score_name = 'f1_score'
        self.init_param()
        
    def init_param(self):
        if self.config.ema:
            self.ema = EMA(self.model, self.ema_rate)
            self.ema.register()
        self.eval_metric = MetricsFunc()
        self.no_improve = 0
        self.best_score = -np.inf
        self.score_list,self.total_train_loss = [], []
        self.all_score = {}
        self.all_score[self.score_name] = self.score_list
    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.lr, amsgrad=True)
        return optimizer
    
    def get_scheduler(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                         T_0=self.config.T_0,
                                                                         T_mult=self.config.T_mult, 
                                                                         eta_min=self.config.eta_min)
        self.use_scheduler = 'restarts'
#         total_steps = self.config.epochs*len(self.train_dataloader)
#         scheduler = get_cosine_schedule_with_warmup(self.optimizer,num_warmup_steps=total_steps*self.config.warmup_radio,num_training_steps=total_steps)
#         self.use_scheduler = 'warmup'
        return scheduler

    def train(self,fold):
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        for epoch in range(self.config.epochs):
            self.model.train()
            self.train_loss = []
            if self.config.adversarial == 'fgm':
                self.fgm = FGM(self.model,self.config.emb_name,self.config.fgm_epsilon)
            bar = tqdm.tqdm(self.train_dataloader,desc='Training')
            for i, (input_data) in enumerate(bar):
                self.train_step(i,epoch,input_data)
                bar.set_postfix(fold=fold,epoch=epoch,tloss=np.array(self.train_loss).mean())
            
            if self.eval:
                if self.config.ema:
                    self.ema.apply_shadow()
                    self.eval_results(self.eval_dataloader,mode='eval')
                    self.ema.restore()
                else:
                    self.eval_results(self.eval_dataloader,mode='eval')
                print('Epoch:[{}]\tloss: {:.5f}\tscore: {:.5f}\n'.format(epoch+1,np.mean(self.train_loss),self.score))
                self.logger.info('Epoch:[{}]\tloss: {:.5f}\tscore: {:.5f}\n'.format(epoch+1,np.mean(self.train_loss),self.score))
            self.score_list.append(self.score)
            self.total_train_loss.extend(self.train_loss)
            
            is_continue = self.save_model(epoch)
            if not is_continue or epoch==self.config.epochs-1:
                self.config.model_num += 1
                self.current_epoch = epoch
                break
        self.all_score[self.score_name] = self.score_list
    
    def train_step(self,i,epoch,input_data):
        self.optimizer.zero_grad()
        output = self.model(input_data[0].to(self.config.device), input_data[1].to(self.config.device),input_data[2].to(self.config.device))
        loss = output
        loss.backward()
        if self.config.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=self.config.norm_type)
        self.train_loss.append(loss.item())
        if self.config.adversarial == 'fgm':
            self.fgm.attack()
            loss_adv = self.model(input_data[0].to(self.config.device), input_data[1].to(self.config.device),input_data[2].to(self.config.device))
            loss_ad = loss_adv
            loss_ad.backward()
            self.fgm.restore()
        if self.use_scheduler == 'restarts':
            self.scheduler.step(epoch + i / len(self.train_dataloader))
        self.optimizer.step()
        if self.use_scheduler != None and self.use_scheduler != 'restarts':
            self.scheduler.step()
        if self.config.ema:
            self.ema.update()
            
    def eval_results(self,dataloader,mode='eval'):
        self.model.eval()
        pred_list = []
        labels_list = []
        if mode == 'eval':
            for i, input_data in enumerate(tqdm.tqdm(dataloader,desc='Evaluate')):
                output = self.model(input_data[0].to(self.config.device), input_data[1].to(self.config.device))
                pred_list += list(output.detach().cpu()) 
#                 pred_list += output.sigmoid().detach().cpu().numpy().tolist()
                labels_list += list(input_data[2])
            self.score = self.eval_metric(pred_list,labels_list,)
        elif mode == 'predict':
            for i, input_data in enumerate(tqdm.tqdm(dataloader,desc='Predict')):
                output = self.model(input_data[0].to(self.config.device), input_data[1].to(self.config.device))
                pred_list += list(output.detach().cpu()) 
            return pred_list
    
    def predict(self):
        self.train_dataloader, self.train_torchdata = self.dataloader_generater.generate_loader(self.train_dataset, mode='eval')
        self.init_param()
        self.model = Bert(self.config)
        self.model.to(self.config.device)
        test_preds_total = []
        train_preds_total = []
        for i in range(1,self.config.model_num):
            self.model.load_state_dict(torch.load('{}/{}_model_{}.bin'.format(self.config.save_model_path,self.config.name, i)))
            test_pred_results = self.eval_results(self.test_dataloader,'predict')
            test_preds_total.append(test_pred_results)
            train_pred_results = self.eval_results(self.train_dataloader,'predict')
            train_preds_total.append(train_pred_results)
        test_preds_merge = np.sum(test_preds_total, axis=0) / (self.config.model_num - 1)
        train_preds_merge = np.sum(train_preds_total, axis=0) / (self.config.model_num - 1)
        score = self.eval_metric(train_preds_merge,self.train_data['label'],'eval')
        results = self.eval_metric(test_preds_merge,mode='predict')
        return results,test_preds_merge
    
    def save_model(self,epoch):
        if not os.path.exists(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if self.eval:
            if self.config.early_stop:
                # 验证函数
                if self.score > self.best_score:
                    self.best_score = self.score
                    if self.config.ema:
                        self.ema.apply_shadow()
                        torch.save(self.model.state_dict(), '{}/{}_model_{}.bin'.format(self.config.save_model_path,self.config.name, self.config.model_num))
                        self.ema.restore()
                    else:
                        torch.save(self.model.state_dict(), '{}/{}_model_{}.bin'.format(self.config.save_model_path,self.config.name, self.config.model_num))
                else:
                    self.no_improve += 1
                if self.no_improve == self.config.early_stop:
                    return False
            else:
                if self.score > self.best_score:
                    self.best_score = self.score
                if (epoch+1) % self.config.save_model_epoch == 0:
                    print('-----save model-----')
                    torch.save(self.model.state_dict(), '{}/{}_model_{}.bin'.format(self.config.save_model_path,self.config.name, self.config.model_num))
        else:
            torch.save(self.model.state_dict(), '{}/{}_model_{}.bin'.format(self.config.save_model_path,self.config.name, self.config.model_num))
        return True
    
    def train_method(self,mode='train'):
        if mode == 'debug':
            self.config.fold = 2
            self.config.epochs = 2
            self.config.early_stop = False
            self.config.save_model_epoch = 1
        if self.eval:
            self.k_fold_train(mode)
        else:
            self.single_trian(mode)
    
    def single_trian(self,mode='train'):
        use_dataset = self.train_dataset
        self.train_dataloader, train_torchdata = self.dataloader_generater.generate_loader(use_dataset, mode='train')
        self.train()
        print('- best score: {}'.format(self.score))
        
    def k_fold_train(self,mode='train'):
        best_scores = []
        kf = KFold(n_splits=self.config.fold, shuffle=True, random_state=self.config.seed)
        use_dataset = self.train_dataset
        for i, (train_index, test_index) in enumerate(kf.split(use_dataset)):
            print(str(i+1), '-'*50)
            tra = [self.train_dataset[index] for index in train_index]
            val = [self.train_dataset[index] for index in test_index]
            print(len(tra))
            print(len(val))
            self.train_dataloader, train_torchdata = self.dataloader_generater.generate_loader(tra, mode='train')
            self.eval_dataloader, eval_torchdata = self.dataloader_generater.generate_loader(val, mode='eval')
            self.model = Bert(self.config)
            self.model.to(self.config.device)
            self.train(i+1)
            best_scores.append(self.best_score)
            torch.cuda.empty_cache()
            if self.eval:
                self.plot(i+1)
            self.init_param()
        for i in range(self.config.fold):
            print('- 第{}折中，best score: {}'.format(i+1, best_scores[i]))
            
    def plot(self,fold):
        plt.figure(figsize=(15,7), dpi=80)
        num = len(self.all_score)
        plt.figure(1)
        ax = plt.subplot(211)
        ax.set_title('train loss')
        plt.plot(range(len(self.total_train_loss)),self.total_train_loss, color="r")
        pos = num+1
        for name,value in self.all_score.items():
            ax = plt.subplot(200+num*10+pos)
            ax.set_title('evaluate {}'.format(name))
            plt.plot(range(self.current_epoch+1),value, color="r")
            pos += 1
            y_max=np.argmax(value)
            y_min=np.argmin(value)
            plt.plot(y_max,value[y_max],'ko')
            plt.plot(y_min,value[y_min],'ko')
            show_max = str(round(value[y_max], 4))
            show_min = str(round(value[y_min], 4))
            plt.annotate(show_max,xy=(y_max,value[y_max]),xytext=(y_max,value[y_max]))
            plt.annotate(show_min,xy=(y_min,value[y_min]),xytext=(y_min,value[y_min]))
        if not os.path.exists(self.config.save_png_path):
            os.makedirs(self.config.save_png_path)
        plt.savefig('{}/training_fold_{}.png'.format(self.config.save_png_path,fold))
        plt.show()