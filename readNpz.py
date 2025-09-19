'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-10
'''


import numpy as np
import matplotlib.pyplot as plt
import re

def parse_npz_filename(filename):
    """
    解析文件名参数（示例文件名：T06C2D0004_n02_bdtr.npz）
    - Txx: 地形类型（11种，如T06=森林）
    - Cx: 气候类型（3种，C0=热带，C1=亚热带，C2=温带）
    - Dxxxx: 地图编号（0000-9999）
    - nxx: 采样编号（00-99）
    - bdtr: 固定标识（可能表示建筑和地形数据）
    """
    pattern = r"T(\d{2})C(\d)D(\d{4})_n(\d{2})_bdtr\.npz"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"文件名格式错误: {filename}")
    
    params = {
        "terrain_type": int(match.group(1)),  # 地形类型编码
        "climate_type": int(match.group(2)),  # 气候类型编码
        "map_id": match.group(3),             # 地图ID
        "sample_id": int(match.group(4))      # 采样编号
    }
    return params

def load_npz_data(filepath):
    """加载.npz文件并解析元数据"""
    filename = filepath.split("/")[-1]
    
    # 1. 解析文件名参数
    try:
        params = parse_npz_filename(filename)
    except ValueError as e:
        print(f"[错误] 文件名解析失败: {e}")
        return None
    
    # 2. 加载.npz文件
    try:
        data = np.load(filepath)
        # 提取数组名称
        array_names = data.files
        # 提取数组数据
        arrays = {name: data[name] for name in array_names}
        data.close()
    except Exception as e:
        print(f"[错误] .npz文件加载失败: {e}")
        return None
    
    # 3. 添加元数据注释
    metadata = {
        "terrain_type": _get_terrain_name(params["terrain_type"]),
        "climate": _get_climate_name(params["climate_type"]),
        "map_id": params["map_id"],
        "sample_id": params["sample_id"],
        "arrays": arrays  # 包含所有数组的字典
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

# --------------- 示例用法 ---------------
if __name__ == "__main__":
    # 示例文件路径
    filepath = "T06C2D0048_n02_bdtr.npz"
    
    # 加载数据
    data = load_npz_data(filepath)
    
    if data is not None:
        # 打印元数据
        print("元数据解析结果:")
        print(f"- 地形类型: {data['terrain_type']}")
        print(f"- 气候: {data['climate']}")
        print(f"- 地图ID: {data['map_id']}")
        print(f"- 采样编号: {data['sample_id']}")
        
        # 打印数组信息
        print("\n数组信息:")
        for name, array in data["arrays"].items():
            print(f"- {name}: 形状={array.shape}, 数据类型={array.dtype}")
        
        # 可视化示例数组（假设包含地形和建筑数据）
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