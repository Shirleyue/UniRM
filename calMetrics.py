import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# from torchmetrics.functional import structural_similarity_index_measure as ssim_tor
# from torchmetrics.functional import peak_signal_noise_ratio as psnr_tor

from torchmetrics.functional.image import structural_similarity_index_measure as ssim_tor
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_tor
import warnings
import torch
import torch.nn as nn



warnings.filterwarnings("ignore")


def calculate_mse(image1, image2):
    """
    计算均方误差 (MSE)
    :param image1: 原始图像 (numpy array)
    :param image2: 重构图像 (numpy array)
    :return: MSE值
    """
    return np.mean((image1 - image2) ** 2)


def calculate_nmse(image1, image2):
    """
    计算归一化均方误差 (NMSE)
    :param image1: 原始图像 (numpy array)
    :param image2: 重构图像 (numpy array)
    :return: NMSE值
    """
    mse = np.mean((image1 - image2) ** 2)
    norm = np.mean(image1 ** 2)
    return mse / norm


def calculate_rmse(image1, image2):
    """
    计算均方根误差 (RMSE)
    :param image1: 原始图像 (numpy array)
    :param image2: 重构图像 (numpy array)
    :return: RMSE值
    """
    return np.sqrt(np.mean((image1 - image2) ** 2))

def calculate_ssim(image1, image2):
    """
    计算结构相似性指数 (SSIM)
    :param image1: 原始图像 (numpy array)
    :param image2: 重构图像 (numpy array)
    :return: SSIM值
    """
        # 根据数据类型自动确定data_range
    if image1.dtype == np.uint8:
        data_range = 255  # uint8类型范围为0-255
    else:
        data_range = 1.0  # 假设非uint8类型为归一化到[0,1]的浮点数

    # return ssim(image1, image2, data_range=image1.max() - image1.min())
    return ssim(image1, image2, data_range=data_range)
    # return ssim(image1, image2)



def calculate_psnr(image1, image2):
    """
    计算峰值信噪比 (PSNR)
    :param image1: 原始图像 (numpy array)
    :param image2: 重构图像 (numpy array)
    :return: PSNR值
    """
    # 根据数据类型自动确定data_range
    if image1.dtype == np.uint8:
        data_range = 255  # uint8类型范围为0-255
    else:
        data_range = 1.0  # 假设非uint8类型为归一化到[0,1]的浮点数
    
    return psnr(image1, image2, data_range=data_range)
    # return psnr(image1, image2)


# def calculate_psnr(image1, image2):
#     """
#     计算峰值信噪比 (PSNR)
#     :param image1: 原始图像 (numpy array)
#     :param image2: 重构图像 (numpy array)
#     :return: PSNR值
#     """
#     # 根据数据类型自动确定data_range
#     if image1.dtype == np.uint8:
#         data_range = 255  # uint8类型范围为0-255
#     else:
#         data_range = 1.0  # 假设非uint8类型为归一化到[0,1]的浮点数
        
#     # return psnr(image1, image2, data_range=data_range)
    
#     err = np.mean((image1 - image2) ** 2)

#     return 10 * np.log10((data_range ** 2) / (err + 1e-10))
    


def evaluate_metrics(target, pred):
    criterion = nn.MSELoss()
    ssim_values = ssim_tor(pred, target) 
    psnr_values = psnr_tor(pred, target, data_range=1.0) 

    mse_values = criterion(pred, target) 
    rmse_values = torch.sqrt(criterion(pred, target))
    nmse_values = criterion(pred, target) / criterion(target, 0 * target) 

    rmse_values = rmse_values.cpu().numpy()* target.shape[0]
    ssim_values = ssim_values.cpu().numpy()* target.shape[0]
    psnr_values = psnr_values.cpu().numpy()* target.shape[0]
    mse_values =   mse_values.cpu().numpy()* target.shape[0]
    nmse_values = nmse_values.cpu().numpy()* target.shape[0]

    # rmse_values = rmse_values.cpu().numpy()
    # ssim_values = ssim_values.cpu().numpy()
    # psnr_values = psnr_values.cpu().numpy()
    # mse_values =   mse_values.cpu().numpy()
    # nmse_values = nmse_values.cpu().numpy()
    
    # print(f"Total RMSE_1: {rmse_values}")
    # print(f"Total SSIM_1: {ssim_values}")
    # print(f"Total PSNR_1: {psnr_values}")
    # print(f"Total MSE_1: {mse_values}")
    # print(f"Total NMSE_1: {nmse_values}")
    
    # print("\n")

    return rmse_values, ssim_values, psnr_values, mse_values, nmse_values





# 示例使用
if __name__ == "__main__":
    import torch

    # 假设我们有两个批量图像，rcv_rss 是生成的图像，rss_gt 是真实图像
    rcv_rss = torch.rand(5, 1, 128, 128)  # 生成的图像 (b=5, 1, 128, 128)
    rss_gt = torch.rand(5, 1, 128, 128)  # 真实图像 (b=5, 1, 128, 128)

    # 计算指标
    avg_rmse, avg_ssim, avg_psnr, mse, nmse = evaluate_metrics(rcv_rss, rss_gt)
    ssim_1 = ssim_tor(rss_gt)

    # 输出结果
    print(f"Average RMSE: {avg_rmse}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average MSE: {mse}")
    print(f"Average NMSE: {nmse}")