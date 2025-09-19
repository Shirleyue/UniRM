'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-21
'''
import torch.optim as optim

import numpy as np
import time
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
import os
import datetime
import logging
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from utils import *
from CreateDataLoader import LoadData

import copy
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import fvcore.common.config
from train import TrainLoop


from cond_unet import PromptUnet_model


#  正常情况下修改4个参数， mode， file_load_path, baseline, few_ratio，h_loss，patch_size, stage， enable_prompt, log_interval, num_epochs
def create_argparser():
    defaults = dict(
        # dataset settings
        few_ratio = 1, # []
        img_size = 128,
        sparse_rate = 0.01,  # [0.003,0.008,0.1]
        tx_num = 15,   # 假定发射器的个数
        dataset = '',
        mode='training', # ['training','prompting','testing']

        # file_load_path = 'results/prompUnet/pretrain/20250430_225538/Few_shot1_/model_best_stage_0',  #整体实验预训练

        file_load_path = '',
        # file_load_path = 'results/prompUnet/prompt/20250508_094148/Few_shot1_/model_best_stage_1',  # 整体实验prompt，具有所有的提示信息

        # experimental settings
        batch_size = 8, # [32,64,128]
        device = "cuda" if torch.cuda.is_available() else "cpu",
        log_interval = 1,
        num_epochs = 200,
        lr_start = 5e-5,  
        lr_end = 1e-6,  

        # lr_start = 1e-4,  
        # lr_end = 1e-6,  

        weight_decay=1e-6,
        early_stop = 5,
        clip_grad = 0.05,
        lr_anneal_steps = 100,
       
        # model settings 
        size = '4',
        patch_size = 16,
        png_save = False,  # 是否保存测试生成的图片
        baseline = False,

        # prompt learning settings
        stage = 0,
        enable_prompt = 0,
        prompt_content = 'f_h_b',

        # prompt_content = 'f_h',
        # prompt_content = 'h_b',
        # prompt_content = 'f_b',

        h_loss = True,  # 如果只嵌入频率或者建筑物，那么h_loss为False

        # 建筑物参数
        num_memory_spatial = 256,
        in_channels = 3,
        conv_num = 2,
        dilations = [1, 4, 8],
        num_memory_freq = 256,
        num_memory_height = 256,

    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == '__main__':
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    args = create_argparser().parse_args()


    num_samples = math.ceil( args.img_size ** 2 * args.sparse_rate )
    print('num_samples:', num_samples)

    # train_name = 'freq_train'
    # data_name = 'gen_freq_4'
    # train_name = 'scene_train'
    # data_name = 'gen_scenario_9'

    train_name = 'train'
    data_name = 'test'

    valid_data = LoadData("./dataset/valid.txt", num_samples=num_samples, few_shot_ratio=args.few_ratio, tx_num=args.tx_num, h_loss=args.h_loss)
    train_data = LoadData("./dataset/"+ train_name + '.txt', num_samples=num_samples, few_shot_ratio=args.few_ratio, tx_num=args.tx_num, h_loss=args.h_loss)

    # data_name = 'scenario_10'
    # data_name = 'freq_4'
    # data_name = 'freq_0'

    test_data = LoadData("./dataset/"+ data_name + '.txt', num_samples=num_samples, few_shot_ratio=args.few_ratio, tx_num=args.tx_num, h_loss=args.h_loss)


    train_dataloader = DataLoader(dataset=train_data, num_workers=0, pin_memory=True, batch_size=args.batch_size,
                                  shuffle=True)
    print("the number of training set", len(train_data))
    valid_dataloader = DataLoader(dataset=valid_data, num_workers=0, pin_memory=True, batch_size=args.batch_size)
    print("the number of valid set", len(valid_data))
    test_dataloader = DataLoader(dataset=test_data, num_workers=0, pin_memory=True, batch_size=args.batch_size)
    print("the number of testing set", len(test_data))


    model = PromptUnet_model(args=args).to(args.device)
    modelName = 'prompUnet'

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M") 

    # 假设参数类型是 32 位浮点数（float32），每个参数占用 4 字节
    bytes_per_param = 4
    total_bytes = total_params * bytes_per_param

    # 将字节数转换为 MB
    total_mb = total_bytes / (1024 * 1024)
    print(f"Total memory usage: {total_mb:.2f} MB")
    
    current_time = datetime.datetime.now()
    time_name = current_time.strftime("%Y%m%d_%H%M%S/")

    if args.few_ratio < 1.0:
        if args.few_ratio == 0.0:
            args.mode = 'testing' # just evaluation on the test set
        else:
            args.mode = 'prompting' # with prompt-tuning
            
    if args.mode in ['training','prompting']:
        if args.enable_prompt != 0:
            args.folder_path = 'results/' + modelName + '/' + 'prompt'+ '/' + time_name 
            # args.folder_path = 'results/' + modelName + '/' + 'prompt'+ '/' + 'Few_shot{}_'.format(args.few_ratio) + args.prompt_content + '_/'

        else:
            args.folder_path = 'results/' + modelName+ '/' + 'pretrain' + '/' + time_name 
            # args.folder_path = 'results/' + modelName+ '/' + 'pretrain' + '/' + 'Few_shot{}_'.format(args.few_ratio) + args.prompt_content + '_/'

    else:
        # args.folder_path = 'results/' + modelName+ '/' + 'test' + '/' + time_name 
        args.folder_path = 'results/' + modelName+ '/' + 'test' + '/' + 'Fshot{}_'.format(args.few_ratio) + args.prompt_content + '_' + data_name + '_/' 


    args.folder_path = f"{args.folder_path}/Few_shot{args.few_ratio}_{args.prompt_content}_{train_name}/"



    if not os.path.exists(args.folder_path):
        os.makedirs(args.folder_path)
        print('make folder!')

    
    if args.file_load_path != '':
        model.load_state_dict(torch.load('{}.pth'.format(args.file_load_path),map_location=args.device), strict=False)
        print('pretrained model loaded') 

    logdir = "{}/logs".format(args.folder_path)
    writer = SummaryWriter(log_dir = logdir,flush_secs=5)
    

    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        data=train_dataloader,
        test_data=test_dataloader, 
        val_data=valid_dataloader,
        device=args.device,
        early_stop = args.early_stop,
    ).run_loop()

    print('Done!')


    