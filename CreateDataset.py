'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-21
'''

import os
import random
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from readPng import parse_png_filename
from readNpz import parse_npz_filename

import re

from tqdm import tqdm
import numpy as np

def get_file_names(folder_path):
    file_pths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_pths


def build_npz_dict(npz_file_lists):
    npz_dict = defaultdict(list)
    for npz_file_pth in npz_file_lists:
        npz_file_name = os.path.basename(npz_file_pth)
        npz_params = parse_npz_filename(npz_file_name)
        key = (npz_params["terrain_type"], npz_params["climate_type"], npz_params["map_id"], npz_params["sample_id"])
        npz_dict[key].append(npz_file_pth)
    return npz_dict

def find_npz_from_png(png_file_name, npz_dict):
    png_params = parse_png_filename(png_file_name)
    key = (png_params["terrain_type"], png_params["climate_type"], png_params["map_id"], png_params["sample_id"])
    return npz_dict.get(key, [])

def get_file_to_Lst(png_pth, npz_pth):
    pathLst = []
    npz_file_lists = get_file_names(npz_pth)
    npz_dict = build_npz_dict(npz_file_lists)  

    for item in os.listdir(png_pth):  # '01.Grassland'
        item_path = os.path.join(png_pth, item).replace("\\", "/")  # ./data/01.Grassland/
        for file in os.listdir(item_path):
            npz_pth_Lst = find_npz_from_png(file, npz_dict) 
            if npz_pth_Lst:
                filePth = os.path.join(item_path, file).replace("\\", "/")
                dlst = f"{filePth}\t{npz_pth_Lst[0]}\n"  
                pathLst.append(dlst)
    return pathLst

def find_png_from_npz(npz_file_lists, png_file_lists):

 
    png_dict = defaultdict(list)
    for png_path in png_file_lists:
        base_name = os.path.basename(png_path)
        
        key = '_'.join(base_name.split('_')[:2])   # T01C0D0000_n00
        png_dict[key].append(png_path)

    match_list = []  
    pattern = re.compile(r"(T\d{2}C\dD\d{4}_n\d{2})_bdtr\.npz") 
    for npz_pth in tqdm(npz_file_lists, desc="Processing NPZ files", unit="file"):
        npz_name = os.path.basename(npz_pth)
        match = re.match(pattern, npz_name)
        common_part = match.group(1)  # T01C0D0000_n00
        matched_pngs = png_dict.get(common_part, [])
        if len(matched_pngs) == 15:
            match_list.extend(f"{png}\t{npz_pth}\n" for png in matched_pngs)

    return match_list

    


def file_filter(dat_lst, dic):
    filtered_dat_lst = []
    for lst in dat_lst:
        png_lst = lst.strip().split('\t')
        png_file_name = png_lst[0].split("/")[-1]
        params = parse_png_filename(png_file_name)  
        
        match = True
        for key, value in dic.items():
            if params.get(key) != value:  
                match = False
                break

        if match:
            filtered_dat_lst.append(lst)

    return filtered_dat_lst

    

# Write the labeled path list to txt
def write_data_path(dataList, fileName, isShuf=False):
    # dataList: a list of data paths and tags,
    # fileName: the txt file name to write path to
    # isShuf: Indicates whether the data set is shuffled
    if isShuf:
        random.shuffle(dataList)  
    with open(fileName, 'w', encoding='UTF-8') as f:
        for dat in dataList:
            f.write(str(dat))
            
def read_data_path(fileName):

    dataList = []
    with open(fileName, 'r', encoding='UTF-8') as f:
        for line in f:
            dataList.append(line)
    return dataList





def div_train_test(dataList, train_ratio, valid_ratio, test_ratio, isShuf=True, group_size=15, random_seed=42):
    """
        
    Returns:
        train_list, valid_list, test_list
    """
    group_indices = np.arange(total_groups)  # 组索引 [0,1,2,...]
    
    if isShuf:
        np.random.seed(random_seed)
        np.random.shuffle(group_indices)
    

    
    train_end = 3001
    valid_end = 3401
    

    train_groups = group_indices[:train_end]
    valid_groups = group_indices[train_end:valid_end]
    test_groups = group_indices[valid_end:]
    
 
    def expand_group_indices(groups):
        indices = []
        for group_id in groups:
            start = group_id * group_size
            indices.extend(range(start, start + group_size))
        return indices
    
    train_indices = expand_group_indices(train_groups)
    valid_indices = expand_group_indices(valid_groups)
    test_indices = expand_group_indices(test_groups)
    

    train_list = [dataList[i] for i in train_indices]
    valid_list = [dataList[i] for i in valid_indices]
    test_list = [dataList[i] for i in test_indices]
    

    print(f"Total samples: {total_size}")
    print(f"Train set: {len(train_list)} ({(len(train_list)/total_size)*100:.1f}%)")
    print(f"Valid set: {len(valid_list)} ({(len(valid_list)/total_size)*100:.1f}%)")
    print(f"Test set: {len(test_list)} ({(len(test_list)/total_size)*100:.1f}%)")
    

    write_data_path(train_list, './dataset/train.txt', isShuf=False)
    write_data_path(valid_list, './dataset/valid.txt', isShuf=False)
    write_data_path(test_list, './dataset/test.txt', isShuf=False)
    
    return train_list, valid_list, test_list
    





if __name__ == '__main__':
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15

    png_pth = './data/png/'
    npz_pth = './data/npz/'

    # dat_lst = get_file_to_Lst(png_pth, npz_pth)
    # print('dat_lst len:', len(dat_lst))
    # write_data_path(dat_lst, './dataset/data.txt', isShuf=False)  # 94458
    # div_train_test(dat_lst, train_ratio, isShuf=True)  


   
    # dat_lst = read_data_path('./dataset/data.txt')
    # print('data len:', len(dat_lst))
    # npz_file_lists = get_file_names(npz_pth)
    # png_file_lists = [item.split('\t')[0] for item in dat_lst]
    # match_lst = find_png_from_npz(npz_file_lists, png_file_lists)
    # write_data_path(match_lst, './dataset/matched_dat.txt', isShuf=False)  # 94458

     
    # dat_lst = read_data_path('./dataset/matched_dat.txt')
    # print('matched_dat len:', len(dat_lst))
    # div_train_test(dat_lst, train_ratio, valid_ratio, test_ratio, isShuf=True)  


    
    # dat_lst = read_data_path('./dataset/test.txt')
    # print('data len:', len(dat_lst))
    # dic_lst = [{"frequency_id": 0}, {"frequency_id":1}, {"frequency_id":2}, {"frequency_id":3}, {"frequency_id":4}]
    # scen_lst = []
    # for idx, terrain_list in enumerate(dic_lst, start=0):      
    #     filtered_dat_lst = file_filter(dat_lst, terrain_list)
    #     # scen_lst.extend(filtered_dat_lst)
    #     print(f'freq{idx} len: {len(filtered_dat_lst)}')
    #     fre_file = f'./dataset/freq_{idx}.txt'  #
    #     write_data_path(filtered_dat_lst, fre_file, isShuf=False)  # 94458
    # print('Done!')


    print('Done!')

    