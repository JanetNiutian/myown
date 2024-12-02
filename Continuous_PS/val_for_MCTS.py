# 需要先打开/data/niutian权限：sudo chmod -R 777 "/data/niutian"
# 需要在引用这个函数前用迭代的方式确定好input
# input : [x,y,vx,vy,theta]*50， 最后一个是最新的state
# output : qcnet的预测轨迹

from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder

import numpy as np
import pickle
import torch

# 装饰器，用于查看函数被调用多少次
class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

model = {'QCNet': QCNet,}['QCNet'].load_from_checkpoint(checkpoint_path='/home/niutian/myown/Continuous_PS/checkpoints_for_QCNet/QCNet_AV2.ckpt')
val_dataset = {'argoverse_v2': ArgoverseV2Dataset,}[model.dataset](root='/home/niutian/原data/Argoverse 2 Motion Forecasting Dataset single', split='val',transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8,pin_memory=False, persistent_workers=True)
trainer = pl.Trainer(accelerator='gpu', devices=1)#, strategy='ddp')

#_transition_other
@CallingCounter
def qc_for_state (input):

    data_new=input
    f_save = open('/home/niutian/myown/intermediate_results/con_ps/data_new_01.pkl', 'wb')  #new
    pickle.dump(data_new, f_save)

    #with torch.no_grad(): 主要是为了防止出现cuda out of memory但是不确定是否有用
    trainer.validate(model, dataloader)
    f_read = open('/home/niutian/myown/intermediate_results/con_ps/pred_for_MCTS_01.pkl', 'rb')
    pred = pickle.load(f_read)
    #print(pred[0]) #返回{}
    return pred