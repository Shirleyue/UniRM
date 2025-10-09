'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-10
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re

def parse_png_filename(filename):
    pattern = r"T(\d{2})C(\d)D(\d{4})_n(\d{2})_f(\d{2})_ss_z(\d{2})\.png"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"文件名格式错误: {filename}")
    
    params = {
        "terrain_type": int(match.group(1)),  
        "climate_type": int(match.group(2)),  
        "map_id": match.group(3),             
        "sample_id": int(match.group(4)),     
        "frequency_id": int(match.group(5)),  
        "height_id": int(match.group(6))      
    }
    return params

def load_png_data(filepath):
    filename = filepath.split("/")[-1]
    

    try:
        params = parse_png_filename(filename)
    except ValueError as e:
        print(f"[错误] 文件名解析失败: {e}")
        return None
    

    try:
        img = Image.open(filepath)
        img_array = np.array(img)  
    except Exception as e:
        print(f"[错误] 图像加载失败: {e}")
        return None
    
    metadata = {
        "terrain_type": _get_terrain_name(params["terrain_type"]),
        "climate": _get_climate_name(params["climate_type"]),
        "frequency": _get_frequency(params["frequency_id"]),
        "height": _get_height(params["height_id"]),
        "data": img_array  
    }
    return metadata

def _get_terrain_name(code):
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
    climate_map = {
        0: "Tropical",
        1: "Subtropical",
        2: "Temperate"
    }
    return climate_map.get(code, "Unknown")

def _get_frequency(code):
    freq_map = {
        0: 150,    # 150 MHz
        1: 1500,   # 1.5 GHz
        2: 1700,   # 1.7 GHz
        3: 3500,   # 3.5 GHz
        4: 22000   # 22 GHz
    }
    return freq_map.get(code, -1)

def _get_height(code):
    height_map = {
        0: 1.5,
        1: 30.0,
        2: 200.0
    }
    return height_map.get(code, -1.0)

if __name__ == "__main__":
    filepath = "Fshot1_f_h_b_paper_show_/data/png/04.Lake/T04C0D0036_n02_f02_ss_z00.png"
    
    data = load_png_data(filepath)
    
    if data is not None:
        print("元数据解析结果:")
        print(f"- 地形类型: {data['terrain_type']}")
        print(f"- 气候: {data['climate']}")
        print(f"- 频率: {data['frequency']} MHz")
        print(f"- 高度: {data['height']} 米")
        print(f"- 图像形状: {data['data'].shape}")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(data['data'])
        plt.title(f"{data['terrain_type']} - {data['height']}m - {data['frequency']}MHz")
        plt.axis('off')
        plt.show()
        plt.close()



