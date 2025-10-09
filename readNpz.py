'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-10
'''


import numpy as np
import matplotlib.pyplot as plt
import re

def parse_npz_filename(filename):
    pattern = r"T(\d{2})C(\d)D(\d{4})_n(\d{2})_bdtr\.npz"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"文件名格式错误: {filename}")
    
    params = {
        "terrain_type": int(match.group(1)),  
        "climate_type": int(match.group(2)),  
        "map_id": match.group(3),             
        "sample_id": int(match.group(4))      
    }
    return params

def load_npz_data(filepath):
    filename = filepath.split("/")[-1]
    

    try:
        params = parse_npz_filename(filename)
    except ValueError as e:
        print(f"[错误] 文件名解析失败: {e}")
        return None
    

    try:
        data = np.load(filepath)
        array_names = data.files

        arrays = {name: data[name] for name in array_names}
        data.close()
    except Exception as e:
        print(f"[错误] .npz文件加载失败: {e}")
        return None
    
    metadata = {
        "terrain_type": _get_terrain_name(params["terrain_type"]),
        "climate": _get_climate_name(params["climate_type"]),
        "map_id": params["map_id"],
        "sample_id": params["sample_id"],
        "arrays": arrays  
    }
    return metadata


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
    climate_map = {
        0: "Tropical",
        1: "Subtropical",
        2: "Temperate"
    }
    return climate_map.get(code, "Unknown")


if __name__ == "__main__":
 
    filepath = "T06C2D0048_n02_bdtr.npz"
    
    data = load_npz_data(filepath)
    
    if data is not None:
        print("元数据解析结果:")
        print(f"- 地形类型: {data['terrain_type']}")
        print(f"- 气候: {data['climate']}")
        print(f"- 地图ID: {data['map_id']}")
        print(f"- 采样编号: {data['sample_id']}")
        
        print("\n数组信息:")
        for name, array in data["arrays"].items():
            print(f"- {name}: 形状={array.shape}, 数据类型={array.dtype}")
        if "terrain_yx" in data["arrays"]:
            plt.figure(figsize=(6, 6))
            plt.imshow(data["arrays"]["terrain_yx"], cmap="terrain")
            plt.title(f"Terrain Map - {data['terrain_type']}")
            plt.colorbar(label="Elevation")
            plt.show()
        
        if "inBldg_zyx" in data["arrays"]:
            for z in range(data["arrays"]["inBldg_zyx"].shape[0]):
                plt.figure(figsize=(6, 6))
                plt.imshow(data["arrays"]["inBldg_zyx"][z], cmap="binary")
                plt.title(f"Building Map (Layer {z}) - {data['terrain_type']}")
                plt.colorbar(label="Building Presence (0/1)")
                plt.show()