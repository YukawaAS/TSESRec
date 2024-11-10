from dataset import *
from model import TransformerBlocks, Model
from loss import *
from torchility import Trainer
from torchility.utils import set_metric_attr
import numpy as np
import random
from metrics import metrics
import warnings

warnings.filterwarnings('ignore')

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

train_path = './datasets/diginetica/slices/train.csv'
val_path = './datasets/diginetica/slices/test.csv'
item_map_path = './datasets/diginetica/itemmap.csv'
item_map = pd.read_csv(item_map_path)

# item总数
item_num = padd_idx = len(item_map)

batch_size = 64             # 批量大小  
model_dim =  16             # 模型特征维度 
lr = 0.0001                  # 学习率 
layer_num = 2               # 模型层数  
head_num = 2                # 注意力头数    
at_k = 20                   # hit@k中的k    
epochs = 10                 # 最大迭代次数  
val_freq = 1                # 验证频率  


# 损失函数
loss = TOP1Loss(padd_idx)

# dataloaders
train_dl = SessionLoader(SessionDataset(train_path, item_map), padd_idx, batch_size, True)
val_dl = SessionLoader(SessionDataset(val_path, item_map), padd_idx, batch_size, False)

# 模型定义
blocks = TransformerBlocks(layer_num, model_dim, head_num, model_dim*4)
model = Model(blocks, model_dim, item_num+1, padd_idx)
opt = torch.optim.Adam(model.parameters(), lr=lr)

# 训练
m = set_metric_attr(metrics, '', ['train', 'val'], padd_idx=padd_idx, k=at_k)  # pylint: disable=E1123
trainer = Trainer(model, loss, opt, epochs, long_output=True, metrics=[m], val_freq=val_freq)
trainer.fit(train_dl, val_dl)
