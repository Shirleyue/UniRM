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
    """向量化插值计算"""
    dis_ratio_km = 0.01  # 距离差与真实距离比例

    y, x = np.mgrid[0:img_size[0], 0:img_size[1]]
    yx = np.column_stack((y.ravel(), x.ravel()))
    d_km = np.sqrt((yx[:, None, 0] - tx_points[:, 0])**2 + 
           (yx[:, None, 1] - tx_points[:, 1])**2) * dis_ratio_km
    zero_mask = (d_km == 0)
    fspl = np.zeros_like(d_km)  # 初始化FSPL
    non_zero = ~zero_mask
    d_scale = 20
    f_scale = 20
    fspl[non_zero] = d_scale * np.log10(d_km[non_zero]) + f_scale * np.log10(f_MHz) + 32.44

    ap_dBm = tx_dBm - fspl
    # ap_dBm[zero_mask] = tx_dBm[np.where(zero_mask)[1]]  # 对齐tx_dBm索引

    return ap_dBm.mean(axis=1).reshape(img_size)
    # return ap_dBm.reshape(img_size[0], img_size[1], -1)


def freq_interpolation(sparse_matrix, tx_num, f_MHz, h_km=None, building_mask=None):
    """
    从稀疏矩阵中选取前tx_num个最大值作为发射点，进行插值计算radio map
    
    参数:
        sparse_matrix: 稀疏采样矩阵
        tx_num: 要选择的发射点数量
        f_MHz: 频率(MHz)
        h_km: 高度(km)，可选
        building_mask: 建筑物掩码，可选
        
    返回:
        插值后的radio map
    """
    # 初始化结果矩阵
    ap_dBm = np.zeros_like(sparse_matrix)
    
    # 获取非零点坐标和值
    y, x = np.where(sparse_matrix != 0)
    if len(y) == 0:  # 没有非零点直接返回
        return ap_dBm
    
    nonzero_coords = np.column_stack((y, x))
    nonzero_values = sparse_matrix[y, x]
    
    # 选择前tx_num个最大值
    n = min(tx_num, len(nonzero_values))  # 确保不超过实际非零点数
    top_indices = np.argpartition(-nonzero_values, n-1)[:n]  # 更高效的top n选择
    tx_points = nonzero_coords[top_indices]
    tx_dBm = nonzero_values[top_indices]

    # sorted_indices = np.argsort(-nonzero_values)[:n]
    # tx_points = nonzero_coords[sorted_indices]
    # tx_dBm = nonzero_values[sorted_indices]
    
    # 插值计算
    ap_dBm = interpolation(tx_points, tx_dBm, f_MHz, sparse_matrix.shape)
    
    # 高度修正
    if h_km is not None:
        ap_dBm -= 20 * np.log10(h_km)
    
    # 建筑物掩码处理
    if building_mask is not None:
        # ap_dBm_min = ap_dBm.min()
        ap_dBm[building_mask == 1] = ap_dBm[building_mask == 1] - 100
    
    return ap_dBm


def _get_height_code (height):
    """实际值 (米)转高度层编号"""
    height_map = {
        1.5: 0,
        30.0: 1,
        200.0: 2
    }
    return height_map.get(height, -1.0)

# def matrix_sampling (matrix, num_samples):
#     """对二维矩阵进行稀疏采样
#     """
#     h, w = matrix.shape
#     # sparse_ratio = h * w / num_samples
#     # print('稀疏率：', sparse_ratio)
#     all_coords = np.array(np.meshgrid(np.arange(h), np.arange(w))).T.reshape(-1, 2)
#     sampled_indices = np.random.choice(len(all_coords), num_samples, replace=False) # 生成图像中每个像素的坐标对
#     sampled_coords = all_coords[sampled_indices]
#     sparse_matrix = np.zeros_like(matrix)
#     for (y, x) in sampled_coords:
#         sparse_matrix[y, x] = matrix[y, x]
#     return sparse_matrix

def matrix_sampling (matrix, num_samples):
    """对二维矩阵进行稀疏采样
    """
    seed = 42
    np.random.seed(seed)  # 设置随机种子
    h, w = matrix.shape
    # sparse_ratio = h * w / num_samples
    # print('稀疏率：', sparse_ratio)
    sparse_matrix = np.zeros_like(matrix)
    x = np.random.randint(0,  h, size=num_samples)
    y = np.random.randint(0,  w, size=num_samples)
    sparse_matrix[y, x] = matrix[y, x]
    return sparse_matrix


def stack_input_data (png_path, npz_path, num_samples, tx_num=15, h_loss=False):
    """根据输入的无线电地图的文件png和npz，并合成带稀疏采样的输入数据
     - num_samples: 采样点数，50对应0.3%，82对应0.5%，131对应0.8%，164对应1%
    """
    png_data = load_png_data(png_path)
    npz_data = load_npz_data(npz_path)
    h_m = png_data['height']
    # frequency = png_data['frequency']/1e3  # frequency, GHz    
    f_MHz = png_data['frequency']

    
    # 对应png和npz的数据，拼成一个三通道数据
    heigtht_code = _get_height_code(png_data['height'])
    terrain_yx = npz_data["arrays"]["terrain_yx"]  # 地形数据
    building_yx = npz_data["arrays"]["inBldg_zyx"][heigtht_code] # 获取对应高度编码的建筑数据
    building_3d = npz_data["arrays"]["inBldg_zyx"]
    rss = png_data['data']
    rss = png_data['data'].astype(np.float32) /255.0
    rss_sparse = matrix_sampling(rss, num_samples)  #对rss进行稀疏采样
    if h_loss:   # 考虑高度loss
        ap = freq_interpolation(rss_sparse, tx_num, f_MHz, h_m*0.001, building_mask=None)
    else:
        ap = freq_interpolation(rss_sparse, tx_num, f_MHz, h_km=None, building_mask=None)

    building_yx = building_yx[np.newaxis, :, :]  # 形状变为 (1, 128, 128)
    rss_sparse = rss_sparse[np.newaxis, :, :]  # 形状变为 (1, 128, 128)
    terrain_yx = terrain_yx[np.newaxis, :, :]  # 形状变为 (1, 128, 128)
    ap = ap[np.newaxis, :, :]  # 形状变为 (1, 128, 128)

    jpg_data = np.vstack([building_yx, rss_sparse, building_3d, ap, terrain_yx])

    return jpg_data, np.expand_dims(rss, axis=0), h_m, f_MHz  # 返回jpd和真实的rss



# # 读取路径文件TXT，该方法可获取单个或多个图像的输入，例如两个图像对应一个标签的情况
class LoadData(Dataset):
    def __init__(self, txt_path, num_samples=50, few_shot_ratio=1.0, tx_num=15, h_loss=False):
        """
        Args:
            txt_path (str): 数据列表文件路径
            num_samples (int): 每个样本的采样点数
            few_shot_ratio (float): 数据缩减比例（0.0 ~ 1.0）
        """
        self.imgs_info = self.get_images(txt_path)
        self.num_samples = num_samples
        self.few_shot_ratio = few_shot_ratio
        self.tx_num = tx_num
        self.h_loss = h_loss

        # 根据 few_shot_ratio 截断数据
        if 0 < few_shot_ratio < 1.0:
            total_len = len(self.imgs_info)
            self.few_shot_len = int(total_len * few_shot_ratio)
            self.imgs_info = self.imgs_info[:self.few_shot_len]  # 取前 N 条数据

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
    """
    将 rcv_rss 保存到 pthLst 中指定的路径，并添加前缀 prefix。
    
    参数:
        pthLst (list): 包含原始文件路径的列表，每个路径是一个字符串。
        rcv_rss (torch.Tensor): 大小为 (b, 1, 128, 128) 的张量，表示恢复的 RSS 数据。
        prefix (str): 新文件路径的前缀。
    """
    # 确保 rcv_rss 的大小与 pthLst 的长度一致
    assert len(pthLst) == rcv_rss.size(0), "pthLst 的长度必须与 rcv_rss 的批次大小一致"
    for idx, lst in enumerate(pthLst):
        # 提取原始文件的路径
        # 生成新的文件路径
        file_name = os.path.join(prefix, lst[2:])  # 去掉 './' 并添加前缀
        os.makedirs(os.path.dirname(file_name), exist_ok=True)  # 创建目录（如果不存在）
        
        # 将 rcv_rss 转换为图像并保存
        rss_data = rcv_rss[idx, 0].cpu().numpy()  # 提取第 idx 个样本，并转换为 NumPy 数组
        rss_data = (rss_data * 255).astype(np.uint8)  # 将数据缩放到 [0, 255] 范围
        image = Image.fromarray(rss_data)  # 将 NumPy 数组转换为 PIL 图像
        image.save(file_name)  # 保存图像



if __name__ == "__main__":
    batsz = 1
    train_dataset = LoadData("./dataset/test.txt")
    train = train_dataset.imgs_info
    print("the number of data:", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batsz,
                                               shuffle=False)

    batch_count = 0

    for batch_idx, (jpg_data, rss_gt, rss_paths , height, frequency) in enumerate(train_loader):
        # jpg_data, rss_gt, _ = batch  # jpg_data 的形状为 (batchsize, 3, H, W)
        # print(png_path)

        if batch_idx >=30: 
            batchsize = jpg_data.shape[0]  # 获取当前 batch 的大小
            batchsize = 1


            # 创建一个包含 (batchsize, 3) 个子图的图像
            fig, axes = plt.subplots(nrows=batchsize, ncols=8, figsize=(12, 4* batchsize))
            # 如果 batchsize=1，axes 是一个 1D 数组，需要调整为 2D 数组
            if batchsize == 1:
                axes = axes.reshape(1, -1)  # 将 axes 从 (3,) 调整为 (1, 3)

            # 遍历 batch 中的每个样本
            for i in range(batchsize):
                # 显示第一个通道，当前高度的building
                axes[i, 0].imshow(jpg_data[i, 0], cmap='gray')  
                axes[i, 0].set_title(rss_paths[0].split("/")[-1])
                axes[i, 0].axis('off')

                # 显示第二个通道 (稀疏RSS 数据)
                axes[i, 1].imshow(jpg_data[i, 1])  # RSS 数据使用热图色图
                axes[i, 1].set_title('Sparse RSS')
                axes[i, 1].axis('off')

                # 显示第三个通道 (1.5m建筑物数据)
                axes[i, 2].imshow(jpg_data[i, 2], cmap='gray')  
                axes[i, 2].set_title('building 0')
                axes[i, 2].axis('off')

                axes[i, 3].imshow(jpg_data[i, 3], cmap='gray')  
                axes[i, 3].set_title('building 1')
                axes[i, 3].axis('off')

                axes[i, 4].imshow(jpg_data[i, 4], cmap='gray')  
                axes[i, 4].set_title('building 2')
                axes[i, 4].axis('off')

                axes[i, 5].imshow(jpg_data[i, 5])  
                axes[i, 5].set_title('Interpolated RSS')
                axes[i, 5].axis('off')

                axes[i, 6].imshow(rss_gt[i, 0])  # RSS 数据使用热图色图
                axes[i, 6].set_title('GroundTruth RSS ' )
                axes[i, 6].axis('off')

                axes[i, 7].imshow(jpg_data[i, 6], cmap='terrain')  # RSS 数据使用热图色图
                axes[i, 6].set_title('Terrain ' )
                axes[i, 6].axis('off')
                


            # 设置整个图的标题
            titleStr = 'freq_' + str(frequency.item())  + '_h_' + str(height.item())
            plt.suptitle(titleStr)
            save_name = str(batch_idx) + '_' + titleStr + '.png'
            plt.savefig(save_name, format='png')  # 保存图像

            # plt.suptitle(rss_paths[i].split("/")[-1])

            # 自动调整子图间距
            plt.tight_layout()

            # 显示图像
            plt.close()

            # 更新 batch 计数
            batch_count += 1
            if batch_count == 16:
                break

    print('Done!')