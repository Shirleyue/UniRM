'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-21
'''

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
from readPng import load_png_data, parse_png_filename
from readNpz import load_npz_data

def interpolation(tx_points, tx_dBm, f_MHz, img_size):
    dis_ratio_km = 0.01  # distance ratio

    y, x = np.mgrid[0:img_size[0], 0:img_size[1]]
    yx = np.column_stack((y.ravel(), x.ravel()))
    d_km = np.sqrt((yx[:, None, 0] - tx_points[:, 0])**2 + 
           (yx[:, None, 1] - tx_points[:, 1])**2) * dis_ratio_km
    zero_mask = (d_km == 0)
    fspl = np.zeros_like(d_km) 
    non_zero = ~zero_mask
    d_scale = 20
    f_scale = 20
    fspl[non_zero] = d_scale * np.log10(d_km[non_zero]) + f_scale * np.log10(f_MHz) + 32.44

    ap_dBm = tx_dBm - fspl
    # ap_dBm[zero_mask] = tx_dBm[np.where(zero_mask)[1]] 

    return ap_dBm.mean(axis=1).reshape(img_size)
    # return ap_dBm.reshape(img_size[0], img_size[1], -1)


def freq_interpolation(sparse_matrix, tx_num, f_MHz, h_km=None, building_mask=None):

    ap_dBm = np.zeros_like(sparse_matrix)

    y, x = np.where(sparse_matrix != 0)
    if len(y) == 0: 
        return ap_dBm
    
    nonzero_coords = np.column_stack((y, x))
    nonzero_values = sparse_matrix[y, x]
    
    n = min(tx_num, len(nonzero_values)) 
    top_indices = np.argpartition(-nonzero_values, n-1)[:n]  
    tx_points = nonzero_coords[top_indices]
    tx_dBm = nonzero_values[top_indices]

    # sorted_indices = np.argsort(-nonzero_values)[:n]
    # tx_points = nonzero_coords[sorted_indices]
    # tx_dBm = nonzero_values[sorted_indices]
    

    ap_dBm = interpolation(tx_points, tx_dBm, f_MHz, sparse_matrix.shape)
    
    if h_km is not None:
        ap_dBm -= 20 * np.log10(h_km)
    
    if building_mask is not None:
        # ap_dBm_min = ap_dBm.min()
        ap_dBm[building_mask == 1] = ap_dBm[building_mask == 1] - 100
    
    return ap_dBm


def _get_height_code (height):

    height_map = {
        1.5: 0,
        30.0: 1,
        200.0: 2
    }
    return height_map.get(height, -1.0)



def matrix_sampling (matrix, num_samples):

    seed = 42
    np.random.seed(seed)  
    h, w = matrix.shape
    sparse_matrix = np.zeros_like(matrix)
    x = np.random.randint(0,  h, size=num_samples)
    y = np.random.randint(0,  w, size=num_samples)
    sparse_matrix[y, x] = matrix[y, x]
    return sparse_matrix


def stack_input_data (png_path, npz_path, num_samples, tx_num=15, h_loss=False):

    png_data = load_png_data(png_path)
    npz_data = load_npz_data(npz_path)
    h_m = png_data['height']
    # frequency = png_data['frequency']/1e3  # frequency, GHz    
    f_MHz = png_data['frequency']

    heigtht_code = _get_height_code(png_data['height'])
    terrain_yx = npz_data["arrays"]["terrain_yx"]  
    building_yx = npz_data["arrays"]["inBldg_zyx"][heigtht_code] 
    building_3d = npz_data["arrays"]["inBldg_zyx"]
    rss = png_data['data']
    rss = png_data['data'].astype(np.float32) /255.0
    rss_sparse = matrix_sampling(rss, num_samples)  
    if h_loss:   
        ap = freq_interpolation(rss_sparse, tx_num, f_MHz, h_m*0.001, building_mask=None)
    else:
        ap = freq_interpolation(rss_sparse, tx_num, f_MHz, h_km=None, building_mask=None)

    building_yx = building_yx[np.newaxis, :, :]  
    rss_sparse = rss_sparse[np.newaxis, :, :]  
    terrain_yx = terrain_yx[np.newaxis, :, :]  # 形(1, 128, 128)
    ap = ap[np.newaxis, :, :]  #  (1, 128, 128)

    jpg_data = np.vstack([building_yx, rss_sparse, building_3d, ap, terrain_yx])

    return jpg_data, np.expand_dims(rss, axis=0), h_m, f_MHz 


class LoadData(Dataset):
    def __init__(self, txt_path, num_samples=50, few_shot_ratio=1.0, tx_num=15, h_loss=False):
        self.imgs_info = self.get_images(txt_path)
        self.num_samples = num_samples
        self.few_shot_ratio = few_shot_ratio
        self.tx_num = tx_num
        self.h_loss = h_loss

        if 0 < few_shot_ratio < 1.0:
            total_len = len(self.imgs_info)
            self.few_shot_len = int(total_len * few_shot_ratio)
            self.imgs_info = self.imgs_info[:self.few_shot_len]  

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def __getitem__(self, index):
        img_path = self.imgs_info[index]
        jpg_data, rss_gt, height, frequency = stack_input_data(img_path[0], img_path[1], self.num_samples, self.tx_num, self.h_loss)
        return jpg_data, rss_gt, img_path[0], height, frequency

    def __len__(self):
        return len(self.imgs_info)


def save_rss_to_png(pthLst, rcv_rss, prefix):

    assert len(pthLst) == rcv_rss.size(0), "pthLst 的长度必须与 rcv_rss 的批次大小一致"
    for idx, lst in enumerate(pthLst):

        file_name = os.path.join(prefix, lst[2:])  
        os.makedirs(os.path.dirname(file_name), exist_ok=True) 

        rss_data = rcv_rss[idx, 0].cpu().numpy()  
        rss_data = (rss_data * 255).astype(np.uint8)  
        image = Image.fromarray(rss_data)  
        image.save(file_name)  

