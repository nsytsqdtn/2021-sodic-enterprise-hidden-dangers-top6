import os
class Config():
    def __init__(self):
        self.name = 'baseline'
        self.init_global()
        self.init_trick()
        self.init_path()
        
    def init_path(self):
        self.user_data_path = '../user_data/{}'.format(self.name)
        if not os.path.exists(self.user_data_path):
            os.makedirs(self.user_data_path)
        self.save_model_path = '{}/models'.format(self.user_data_path)
        self.save_png_path = '{}/png'.format(self.user_data_path)
        self.log_path = '{}/logging.log'.format(self.user_data_path)
        
    def init_global(self):
        self.model_num = 1
        self.num_workers = 4
        self.eval = True
        
    def init_trick(self):
        self.adversarial = False
        self.emb_name = 'bert.embeddings.word_embeddings.weight'
        self.ema = False
        self.ema_rate = 0.999
        self.clip = False
        self.max_norm = 20
        self.norm_type = 2
        