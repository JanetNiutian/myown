# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import faulthandler
faulthandler.enable()
import torch
torch.set_float32_matmul_precision('medium')

from argparse import ArgumentParser

import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule # 问题

from predictors import QCNet


if __name__ == '__main__':
    
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str,default='/home/niutian/原data/Argoverse 2 Motion Forecasting Dataset' )#required=True
    parser.add_argument('--train_batch_size', type=int, default=32)#required=True,default=32
    parser.add_argument('--val_batch_size', type=int, default=1)#required=True
    parser.add_argument('--test_batch_size', type=int, default=1)#required=True
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=160)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu') #auto
    parser.add_argument('--devices', type=int, default=1)#required=True  8
    parser.add_argument('--max_epochs', type=int, default=64) #64
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    QCNet.add_model_specific_args(parser) #添加可接受的参数
    args = parser.parse_args()
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()
    
    model = QCNet(**vars(args)).cuda()# **vars(args)表示告诉函数按dict来解析，还多了一个vars()格式转换，args需要是dict对象，将args传递的参数从namespace 转换为dict,这样就不用将args包含的参数一一列举出来再传入相应函数中,化简代码,增加可读性.相当于args.d1,args.d2
    print('111')
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args)) #字典的创建、key对应的value
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min') #根据monitor监控的量，保存5个min的模型
    lr_monitor = LearningRateMonitor(logging_interval='epoch') #pl提供的回调函数，用于记录学习率的变化
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         #resume_from_checkpoint='/data/niutian/code/QCNet-main/lightning_logs/version_3/checkpoints/epoch=6-step=43736.ckpt',
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=64)  #选择将要训练的硬件（auto），硬件数量，Strategy for multi-process single-device training on one or multiple nodes
    trainer.fit(model, datamodule)
# 先更改resume_from_checkpoint=
# nohup python train_qcnet.py --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --devices 8 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 >> train_qcnet_02.txt &