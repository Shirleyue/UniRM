'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-10
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re

def parse_png_filename(filename):
    """
    解析文件名参数（示例文件名：T01C0D0000_n00_f01_ss_z00.png）
    - Txx: 地形类型（11种，如T01=密集城区）
    - Cx: 气候类型（3种，C0=热带，C1=亚热带，C2=温带）
    - Dxxxx: 地图编号（0000-9999）
    - nxx: 采样编号（00-99）
    - fxx: 频率编号（00-04对应5个频段，如f00=150MHz）
    - zxx: 高度层（00=1.5m，01=30m，02=200m）
    - ss: 固定标识（可能表示信号强度图）
    """
    pattern = r"T(\d{2})C(\d)D(\d{4})_n(\d{2})_f(\d{2})_ss_z(\d{2})\.png"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"文件名格式错误: {filename}")
    
    params = {
        "terrain_type": int(match.group(1)),  # 地形类型编码
        "climate_type": int(match.group(2)),  # 气候类型编码
        "map_id": match.group(3),             # 地图ID
        "sample_id": int(match.group(4)),     # 采样编号
        "frequency_id": int(match.group(5)),  # 频率编号
        "height_id": int(match.group(6))      # 高度层编号
    }
    return params

def load_png_data(filepath):
    """加载JPG文件并解析元数据"""
    filename = filepath.split("/")[-1]
    
    # 1. 解析文件名参数
    try:
        params = parse_png_filename(filename)
    except ValueError as e:
        print(f"[错误] 文件名解析失败: {e}")
        return None
    
    # 2. 加载图像数据
    try:
        img = Image.open(filepath)
        img_array = np.array(img)  # 转换为numpy数组 (H, W, C)
    except Exception as e:
        print(f"[错误] 图像加载失败: {e}")
        return None
    
    # 3. 添加元数据注释
    metadata = {
        "terrain_type": _get_terrain_name(params["terrain_type"]),
        "climate": _get_climate_name(params["climate_type"]),
        "frequency": _get_frequency(params["frequency_id"]),
        "height": _get_height(params["height_id"]),
        "data": img_array  # 图像数据 (128x128x3)
    }
    return metadata

# --------------- 辅助函数：编码转可读文本 ---------------
def _get_terrain_name(code):
    """地形类型编码转名称"""
    terrain_map = {
        1: "Dense Urban",
        2: "Ordinary Urban",
        3: "Rural",
        4: "Suburban",
        5: "Mountainous",
        6: "Forest",
        7: "Desert",
        8: "Grassland",
        9: "Island",
        10: "Ocean",
        11: "Lake"
    }
    return terrain_map.get(code, "Unknown")

def _get_climate_name(code):
    """气候类型编码转名称"""
    climate_map = {
        0: "Tropical",
        1: "Subtropical",
        2: "Temperate"
    }
    return climate_map.get(code, "Unknown")

def _get_frequency(code):
    """频率编号转实际值 (MHz)"""
    freq_map = {
        0: 150,    # 150 MHz
        1: 1500,   # 1.5 GHz
        2: 1700,   # 1.7 GHz
        3: 3500,   # 3.5 GHz
        4: 22000   # 22 GHz
    }
    return freq_map.get(code, -1)

def _get_height(code):
    """高度层编号转实际值 (米)"""
    height_map = {
        0: 1.5,
        1: 30.0,
        2: 200.0
    }
    return height_map.get(code, -1.0)

# --------------- 示例用法 ---------------
if __name__ == "__main__":
    # 示例文件路径
    filepath = "Fshot1_f_h_b_paper_show_/data/png/04.Lake/T04C0D0036_n02_f02_ss_z00.png"
    
    # 加载数据
    data = load_png_data(filepath)
    
    if data is not None:
        # 打印元数据
        print("元数据解析结果:")
        print(f"- 地形类型: {data['terrain_type']}")
        print(f"- 气候: {data['climate']}")
        print(f"- 频率: {data['frequency']} MHz")
        print(f"- 高度: {data['height']} 米")
        print(f"- 图像形状: {data['data'].shape}")
        
        # 可视化图像
        plt.figure(figsize=(6, 6))
        plt.imshow(data['data'])
        plt.title(f"{data['terrain_type']} - {data['height']}m - {data['frequency']}MHz")
        plt.axis('off')
        plt.show()
        plt.close()



