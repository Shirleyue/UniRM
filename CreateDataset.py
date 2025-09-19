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

# # 获取某一文件夹下的所有文件路径列表
def get_file_names(folder_path):
    # 获取文件夹下的所有文件和子文件夹
    file_pths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_pths



# 构建 NPZ 文件的参数字典
def build_npz_dict(npz_file_lists):
    npz_dict = defaultdict(list)
    for npz_file_pth in npz_file_lists:
        npz_file_name = os.path.basename(npz_file_pth)
        npz_params = parse_npz_filename(npz_file_name)
        key = (npz_params["terrain_type"], npz_params["climate_type"], npz_params["map_id"], npz_params["sample_id"])
        npz_dict[key].append(npz_file_pth)
    return npz_dict

# 根据 PNG 文件名，找到与之匹配的 NPZ 文件
def find_npz_from_png(png_file_name, npz_dict):
    png_params = parse_png_filename(png_file_name)
    key = (png_params["terrain_type"], png_params["climate_type"], png_params["map_id"], png_params["sample_id"])
    return npz_dict.get(key, [])

# 获取 PNG 和 NPZ 文件的匹配列表
def get_file_to_Lst(png_pth, npz_pth):
    pathLst = []
    npz_file_lists = get_file_names(npz_pth)
    npz_dict = build_npz_dict(npz_file_lists)  # 构建 NPZ 参数字典

    for item in os.listdir(png_pth):  # '01.Grassland'
        item_path = os.path.join(png_pth, item).replace("\\", "/")  # ./data/01.Grassland/
        for file in os.listdir(item_path):
            npz_pth_Lst = find_npz_from_png(file, npz_dict)  # 通过字典快速查找匹配的 NPZ
            if npz_pth_Lst:
                filePth = os.path.join(item_path, file).replace("\\", "/")
                dlst = f"{filePth}\t{npz_pth_Lst[0]}\n"  # 拼接 PNG 和 NPZ，npz_pth_Lst是一个列表，实际只能匹配1条
                pathLst.append(dlst)
    return pathLst

def find_png_from_npz(npz_file_lists, png_file_lists):

    # 预处理：构建PNG文件名快速查找字典（O(1)查找）
    png_dict = defaultdict(list)
    for png_path in png_file_lists:
        base_name = os.path.basename(png_path)
        # 提取公共部分（假设PNG文件名格式为 T01C0D0000_n00_fX_ss_zY.png）
        key = '_'.join(base_name.split('_')[:2])   # 获取T01C0D0000_n00部分
        png_dict[key].append(png_path)

    match_list = []  # 用于存储所有匹配结果
    pattern = re.compile(r"(T\d{2}C\dD\d{4}_n\d{2})_bdtr\.npz")  # 预编译正则
    # 使用 tqdm 包装 npz_file_lists 的循环，显示进度条
    for npz_pth in tqdm(npz_file_lists, desc="Processing NPZ files", unit="file"):
        npz_name = os.path.basename(npz_pth)
        match = re.match(pattern, npz_name)
        common_part = match.group(1)  # T01C0D0000_n00
        matched_pngs = png_dict.get(common_part, [])
        if len(matched_pngs) == 15:
            # 批量生成结果（减少列表append次数）
            match_list.extend(f"{png}\t{npz_pth}\n" for png in matched_pngs)

    return match_list

    


def file_filter(dat_lst, dic):
    """
    根据字典 dic 中的条件筛选 dat_lst 中的数据
    :param dat_lst: 数据列表，每个元素是一个元组 (png_path, npz_path)
    :param dic: 筛选条件字典，例如 {"height_id": 1, "frequency_id": 2, "terrain_type": 1}
    :return: 筛选后的数据列表
    """
    filtered_dat_lst = []
    for lst in dat_lst:
        png_lst = lst.strip().split('\t')
        png_file_name = png_lst[0].split("/")[-1]
        params = parse_png_filename(png_file_name)  # 解析 png 文件名
        # 检查是否满足所有筛选条件
        match = True
        for key, value in dic.items():
            if params.get(key) != value:  # 如果某个条件不满足
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
        random.shuffle(dataList)  # 打乱数据集
    with open(fileName, 'w', encoding='UTF-8') as f:
        for dat in dataList:
            f.write(str(dat))
            
def read_data_path(fileName):
    """
    从 txt 文件中读取存储的文件路径列表
    :param fileName: txt 文件路径
    :return: 文件路径列表
    """
    dataList = []
    with open(fileName, 'r', encoding='UTF-8') as f:
        for line in f:
            dataList.append(line)
    return dataList


# div data into train and test
# def div_train_test(dataList, train_ratio, valid_ratio, test_ratio, isShuf=True):
#     """
#     按比例划分数据集为训练集、验证集和测试集
#     Args:
#         dataList: 原始数据集（列表）
#         train_ratio: 训练集比例（0~1）
#         valid_ratio: 验证集比例（0~1）
#         test_ratio: 测试集比例（0~1）
#         isShuf: 是否打乱数据
#     Returns:
#         train_list, valid_list, test_list
#     """
#     # 校验比例总和是否为1
#     assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
#     if isShuf:
#         random.shuffle(dataList)  # 打乱数据

#     total_size = len(dataList)
#     # 计算各分区的索引边界
#     train_end = int(total_size * train_ratio)
#     valid_end = train_end + int(total_size * valid_ratio)
    
#     # 划分数据集
#     train_list = dataList[:train_end]
#     valid_list = dataList[train_end:valid_end]
#     test_list = dataList[valid_end:]
    
#     # 打印信息
#     print(f"Total samples: {total_size}")
#     print(f"Train set: {len(train_list)} ({(len(train_list)/total_size)*100:.1f}%)")
#     print(f"Valid set: {len(valid_list)} ({(len(valid_list)/total_size)*100:.1f}%)")
#     print(f"Test set: {len(test_list)} ({(len(test_list)/total_size)*100:.1f}%)")
    
#     # 保存到文件（可选）
#     write_data_path(train_list, './dataset/train.txt', isShuf=False)
#     write_data_path(valid_list, './dataset/valid.txt', isShuf=False)
#     write_data_path(test_list, './dataset/test.txt', isShuf=False)
    
#     return train_list, valid_list, test_list


def div_train_test(dataList, train_ratio, valid_ratio, test_ratio, isShuf=True, group_size=15, random_seed=42):
    """
    按分组比例划分数据集为训练集、验证集和测试集（保证每组元素完整）
    
    Args:
        dataList: 原始数据集（列表或数组）
        train_ratio: 训练集比例（0~1）
        valid_ratio: 验证集比例（0~1）
        test_ratio: 测试集比例（0~1）
        isShuf: 是否打乱组顺序（默认True）
        group_size: 每组元素数量（默认15）
        random_seed: 随机种子（默认42）
        
    Returns:
        train_list, valid_list, test_list
    """
    # ========== 参数校验 ==========
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    assert len(dataList) % group_size == 0, f"数据总长度必须能被{group_size}整除"
    
    # ========== 核心逻辑 ==========
    total_size = len(dataList)
    total_groups = len(dataList) // group_size
    group_indices = np.arange(total_groups)  # 组索引 [0,1,2,...]
    
    # 打乱组顺序（保持可复现性）
    if isShuf:
        np.random.seed(random_seed)
        np.random.shuffle(group_indices)
    
    # 计算各集合的组边界
    # train_end = int(total_groups * train_ratio)
    # valid_end = train_end + int(total_groups * valid_ratio)
    
    train_end = 3001
    valid_end = 3401
    
    # 获取各组对应的原始索引
    train_groups = group_indices[:train_end]
    valid_groups = group_indices[train_end:valid_end]
    test_groups = group_indices[valid_end:]
    
    # 展开组索引到元素级别
    def expand_group_indices(groups):
        indices = []
        for group_id in groups:
            start = group_id * group_size
            indices.extend(range(start, start + group_size))
        return indices
    
    train_indices = expand_group_indices(train_groups)
    valid_indices = expand_group_indices(valid_groups)
    test_indices = expand_group_indices(test_groups)
    
    # 划分数据集
    train_list = [dataList[i] for i in train_indices]
    valid_list = [dataList[i] for i in valid_indices]
    test_list = [dataList[i] for i in test_indices]
    
    # 打印信息
    print(f"Total samples: {total_size}")
    print(f"Train set: {len(train_list)} ({(len(train_list)/total_size)*100:.1f}%)")
    print(f"Valid set: {len(valid_list)} ({(len(valid_list)/total_size)*100:.1f}%)")
    print(f"Test set: {len(test_list)} ({(len(test_list)/total_size)*100:.1f}%)")
    
    # 保存到文件（可选）
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
    #   ----- 读取所有数据，写入txt文件  ----- #
    # dat_lst = get_file_to_Lst(png_pth, npz_pth)
    # print('dat_lst len:', len(dat_lst))
    # write_data_path(dat_lst, './dataset/data.txt', isShuf=False)  # 94458
    # div_train_test(dat_lst, train_ratio, isShuf=True)  


    # ---- 对数据进行匹配筛选 ------- # 
    # dat_lst = read_data_path('./dataset/data.txt')
    # print('data len:', len(dat_lst))
    # npz_file_lists = get_file_names(npz_pth)
    # png_file_lists = [item.split('\t')[0] for item in dat_lst]
    # match_lst = find_png_from_npz(npz_file_lists, png_file_lists)
    # write_data_path(match_lst, './dataset/matched_dat.txt', isShuf=False)  # 94458

    # ---- 选取数据进行训练-测试划分 ------- # 
    # dat_lst = read_data_path('./dataset/matched_dat.txt')
    # print('matched_dat len:', len(dat_lst))
    # div_train_test(dat_lst, train_ratio, valid_ratio, test_ratio, isShuf=True)  


    #  --- 对测试数据进行筛选  按频率划分--------- #
    # dat_lst = read_data_path('./dataset/test.txt')
    # print('data len:', len(dat_lst))
    # dic_lst = [{"frequency_id": 0}, {"frequency_id":1}, {"frequency_id":2}, {"frequency_id":3}, {"frequency_id":4}]
    # scen_lst = []
    # for idx, terrain_list in enumerate(dic_lst, start=0):      
    #     filtered_dat_lst = file_filter(dat_lst, terrain_list)
    #     # scen_lst.extend(filtered_dat_lst)
    #     print(f'freq{idx} len: {len(filtered_dat_lst)}')
    #     fre_file = f'./dataset/freq_{idx}.txt'  # 更规范的命名
    #     write_data_path(filtered_dat_lst, fre_file, isShuf=False)  # 94458
    # print('Done!')


    print('Done!')

    