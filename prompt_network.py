import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

import copy
from CreateDataLoader import _get_height_code


class Memory(nn.Module):
    """ Memory prompt
    """
    def __init__(self, num_memory, memory_dim, args=None):
        super().__init__()

        self.args = args

        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C

        self.x_proj = nn.Linear(memory_dim, memory_dim)
        
        self.initialize_weights()

        print("model initialized memory")


    def initialize_weights(self):
        # torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        # torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x,Type='',shape=None):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product
        assert x.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "dimension mismatch"

        x_query = torch.tanh(self.x_proj(x)) # 

        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]
        loss_top = 0.0

        return dict(out=out, att_weight=att_weight, loss=loss_top)


class ASPP_Terrain(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, dilations=[1, 6, 12]):
        super().__init__()
        middle_channels = out_channels // 2
        # 空洞卷积分支
        self.aspp_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, 
                     padding=dilations[0], dilation=dilations[0]),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, 
                     padding=dilations[1], dilation=dilations[1]),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, 
                     padding=dilations[2], dilation=dilations[2]),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化分支
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, middle_channels, kernel_size=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            # nn.Conv2d(4 * out_channels, out_channels, kernel_size=1),
            nn.Conv2d(middle_channels*4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 初始化所有子模块
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用kaiming_normal_初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
                # 偏置初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm的gamma初始化为1，beta为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入x为地形数据（B×1×H×W）
        x1 = self.aspp_conv1(x)  # 空洞率1
        x2 = self.aspp_conv2(x)  # 空洞率6
        x3 = self.aspp_conv3(x)  # 空洞率12
        
        # 全局平均池化分支
        x4 = self.global_avg(x)
        x4 = F.interpolate(x4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 通道维度拼接
        out = torch.cat([x1, x2, x3, x4], dim=1)
        # out = x1 + x2 +x3 + x4
        out = self.fusion(out)  # 融合后输出（B×32×H×W）
        return out

class Terrain_prompt(nn.Module):
    """ miltiscale convolutional kernels
    """
    def __init__(self, num_memory, memory_dim, in_channels, dilations, args=None):
        super().__init__()

        self.aspp = ASPP_Terrain(in_channels=in_channels, out_channels=memory_dim, dilations=dilations) 

        self.spatial_memory = Memory(num_memory, memory_dim, args=args)

        self.args = args

    def forward(self,x):
        """
        :param x: query features with size [N, C, H, W]
        :return: multiscale spatial_prompt
        """
        out = []
        loss = 0.0
        t = self.aspp(x).permute(0,2,3,1)  # (b,h,w,dim)
        shape = x.shape  #(b,in_chans, h, w)
        t = t.reshape(t.shape[0],-1,t.shape[-1])  # (b, h*w, dim)  2,16384,512
        output = self.spatial_memory(t)
        output_out = output['out']  # (b, h*w, dim)
        out.append(output_out.reshape(x.shape[0],x.shape[2],x.shape[3],t.shape[-1]).permute(0,3,1,2))
        loss += output['loss']
        # out = torch.mean(torch.cat(out, dim=0), dim=0)
        return out

class Building_prompt(nn.Module):
    """ miltiscale convolutional kernels
    """
    def __init__(self, num_memory, memory_dim, in_channels, conv_num, args=None):
        super().__init__()

        self.conv = [nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=1, padding='same')]
        for i in range(conv_num-1):
            self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=2**(i+1)+1, padding='same'))

        self.conv = nn.Sequential(*self.conv)

        self.spatial_memory = Memory(num_memory, memory_dim, args=args)

        for conv in self.conv:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.args = args

    def forward(self,x):
        """
        :param x: query features with size [N, C, H, W]
        :return: multiscale spatial_prompt
        """
        out = []
        loss = 0.0
        for i in range(len(self.conv)):
            t = self.conv[i](x).permute(0,2,3,1)  # (b,h,w,dim)
            shape = x.shape  #(b,in_chans, h, w)
            t = t.reshape(t.shape[0],-1,t.shape[-1])  # (b, h*w, dim)  2,16384,512
            output = self.spatial_memory(t)
            output_out = output['out']  # (b, h*w, dim)
            out.append(output_out.reshape(x.shape[0],x.shape[2],x.shape[3],t.shape[-1]).permute(0,3,1,2))
            loss += output['loss']
        # out = torch.mean(torch.cat(out, dim=0))
        return out


class Height_prompt(nn.Module):
    """ miltiscale convolutional kernels
    """
    def __init__(self, num_memory, memory_dim, in_channels, conv_num, args=None):
        super().__init__()

        self.conv = [nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=1, padding='same')]
        for i in range(conv_num-1):
            self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=2**(i+1)+1, padding='same'))

        self.conv = nn.Sequential(*self.conv)

        self.spatial_memory = Memory(num_memory, memory_dim, args=args)

        for conv in self.conv:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.args = args

    def forward(self,x):
        """
        :param x: query features with size [N, C, H, W]
        :return: multiscale spatial_prompt
        """
        out = []
        loss = 0.0
        for i in range(len(self.conv)):
            t = self.conv[i](x).permute(0,2,3,1)  # (b,h,w,dim)
            shape = x.shape  #(b,in_chans, h, w)
            t = t.reshape(t.shape[0],-1,t.shape[-1])  # (b, h*w, dim)  2,16384,512
            output = self.spatial_memory(t)
            output_out = output['out']  # (b, h*w, dim)
            out.append(output_out.reshape(x.shape[0],x.shape[2],x.shape[3],t.shape[-1]).permute(0,3,1,2))
            loss += output['loss']
        # out = torch.mean(torch.cat(out, dim=0))
        return out

class Frequency_prompt(nn.Module):
    """ miltiscale convolutional kernels
    """
    def __init__(self, num_memory, memory_dim, in_channels, conv_num, args=None):
        super().__init__()

        self.conv = [nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=1, padding='same')]

        for i in range(conv_num-1):
            self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=2**(i+1)+1, padding='same'))

        self.conv = nn.Sequential(*self.conv)

        self.spatial_memory = Memory(num_memory, memory_dim, args=args)

        for conv in self.conv:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.args = args

    def forward(self,x):
        """
        :param x: query features with size [N, C, H, W]
        :return: multiscale spatial_prompt
        """
        out = []
        loss = 0.0
        for i in range(len(self.conv)):
            t = self.conv[i](x).permute(0,2,3,1)  # (b,h,w,dim)
            shape = x.shape  #(b,in_chans, h, w)
            t = t.reshape(t.shape[0],-1,t.shape[-1])  # (b, h*w, dim)  2,16384,512
            output = self.spatial_memory(t)
            output_out = output['out']  # (b, h*w, dim)
            out.append(output_out.reshape(x.shape[0],x.shape[2],x.shape[3],t.shape[-1]).permute(0,3,1,2))
            loss += output['loss']
        # out = torch.mean(torch.cat(out, dim=0))
        return out
 

class Prompt(nn.Module):
    """
    Prompt ST with spatial prompt,freq and height prompt
    spatial prompt: multiscale convolutional kernels
    freq prompt: 
    height prompt: 
    """
    def __init__(self, num_memory_spatial, memory_dim, in_channels, conv_num, dilations, num_memory_freq, 
                 num_memory_height, args=None):
        super().__init__()


        self.building_prompt = Building_prompt(num_memory_spatial, memory_dim, in_channels, conv_num, args=args)
        # self.terrain_prompt = Terrain_prompt(num_memory_spatial, memory_dim, in_channels=1, dilations=dilations, args=args)
        # self.terrain_prompt = Building_prompt(num_memory_spatial, memory_dim, in_channels=1, conv_num=1, args=args)

        self.height_prompt = Height_prompt(num_memory_height, memory_dim, in_channels=1, conv_num=conv_num, args=args) # 高度只要当前高度的building
        self.freq_prompt = Frequency_prompt(num_memory_freq, memory_dim, in_channels=1, conv_num=1, args=args) # 频率只卷积一层，不需要多尺度


        
    def forward(self, x, f_value, h_value):
        x_build_3d = x[:, 2:5, :, :]   # 3d building information
        # x_build_height = x[:, 0, :, :]  # building in given height
        # height_code = _get_height_code(h_value) for 
        # x_build_height_2 = x_build_3d[:, height_code, :, :]  # building in given height
        x_build_height = x[:, 0:1, :, :]  # building in given height
        x_freq_ap = x[:,5:6,:,:]  # frequency
        x_terraian = x[:,6:7,:,:]  

        bd_out = self.building_prompt(x_build_3d)
        stacked_tensor = torch.stack(bd_out, dim=0)
        bd_out = torch.mean(stacked_tensor, dim=0)

        h_out = self.height_prompt(x_build_height)
        stacked_tensor = torch.stack(h_out, dim=0)
        h_out = torch.mean(stacked_tensor, dim=0)


        freq_out = self.freq_prompt(x_freq_ap)
        stacked_tensor = torch.stack(freq_out, dim=0)
        freq_out = torch.mean(stacked_tensor, dim=0)

        td_out = self.terrain_prompt(x_terraian)
        stacked_tensor = torch.stack(td_out, dim=0)
        td_out = torch.mean(stacked_tensor, dim=0)

        # return dict(f=freq_out, h=h_out, t=td_out, b=bd_out)
        return dict(f=freq_out, h=h_out, b=bd_out, t=td_out)
    


# 测试用例
if __name__ == "__main__":
    # 空间参数
    num_memory_spatial = 512
    num_memory_freq = 512
    num_memory_height = 512
    memory_dim = 256
    in_channels = 3  # the input channels of 3d buildings
    conv_num = 3
    dilations = [1,4,8]
    
    
    x = torch.randn(2, 8, 128, 128)
    f_value = torch.tensor([150, 1500], dtype=torch.float32)  
    h_value = torch.tensor([1.5, 30], dtype=torch.float32)  
    model = Prompt (num_memory_spatial, memory_dim, in_channels, conv_num, dilations,
            num_memory_freq, num_memory_height, args=None)
    y = model(x, f_value, h_value)
    # print(y['t'])
    # print(y['b'])
    # print(y['f'])
    # print(y['h'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M") 
    print('Done!')
