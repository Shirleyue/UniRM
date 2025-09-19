import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from CreateDataLoader import LoadData, save_rss_to_png
from torch.utils.data import DataLoader
from calMetrics import calculate_psnr, calculate_rmse, calculate_ssim, evaluate_metrics
import argparse
from utils import *
import math
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from readPng import parse_png_filename
# from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


# {
#     "Crowd": {
#         "random": {0.5: 1.21, 0.7: 1.39},
#         "temporal": {0.5: 1.08}
#     },
#     "Traffic": {
#         "block": {0.3: 0.87}
#     }
# }
def save_metrics_to_file(file_path, metrics):
    """
    将指标保存到文件中
    :param file_path: 文件路径
    :param metrics: 指标值（可以是标量或列表）
    """
    with open(file_path, 'a+') as f:
        if isinstance(metrics, (list, np.ndarray)):
            np.savetxt(f, metrics, fmt='%.6f')  # 保存数组或列表
        else:
            f.write(f"{metrics:.6f}\n")  # 保存标量值

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

class HybridLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    
    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        ssim_loss = 1 - ssim(y_pred, y_true, data_range=1.0)
        return 0.5*mse_loss + 0.5*ssim_loss 


class TrainLoop:
    def __init__(self, args, writer, model, data, test_data, val_data, device, early_stop = 5):
        self.args = args
        self.writer = writer
        self.model = model
        self.data = data
        self.test_data = test_data
        self.val_data = val_data
        self.device = device

        self.loss_fn = nn.MSELoss().to(self.device)
        # self.loss_fn = GradientLoss().to(self.device)
        # self.loss_fn = self.hybrid_loss

        # self.loss_fn = RMSELoss().to(self.device)
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.early_stop = early_stop
        self.best_rmse = 1e9
        self.warmup_steps=5   # 预热步数
        self.folder_path = args.folder_path
          # 分层参数分组
        backbone_params = []
        prompt_params = []
        relation_params = []
            # 根据模块名称分类参数
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 1. 提示模块参数（包含"prompt"关键字）
            if "prompt" in name:
                prompt_params.append(param)
            # 2. 关系网络参数（包含"relation_layers"或"attentions"等关键字）
            elif any(key in name for key in ["relation_layers", "attentions"]):
                relation_params.append(param)
            # 3. 主干网络参数（其他核心组件）
            elif any(key in name for key in ["init_conv", "downs", "mid_block", "ups", "final_conv"]):
                backbone_params.append(param)
            # 4. 默认归入主干网络
            else:
                backbone_params.append(param)
            # 验证参数完整性
        total_params = len(backbone_params) + len(prompt_params) + len(relation_params)
        trainable_params = len([p for p in model.parameters() if p.requires_grad])
        assert total_params == trainable_params, f"参数分组错误！总分组参数:{total_params} != 可训练参数:{trainable_params}"

        self.opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_start * 0.1},  # 主干网络学习率降低10倍
            {"params": prompt_params, "lr": args.lr_start},          # 提示模块保持基准学习率
            {"params": relation_params, "lr": args.lr_start * 2}     # 关系网络使用更高学习率
            ], weight_decay=args.weight_decay
        )

        # 打印各层参数数量（调试用）
        # print(f"优化器分组：主干({len(backbone_params)}) | 提示({len(prompt_params)}) | 关系({len(relation_params)})")

        # self.opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad==True], lr=args.lr_start, weight_decay=args.weight_decay)

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs, eta_min=args.lr_end)
        self.lr_start = args.lr_start
        self.lr_min = args.lr_end
        self.lr_anneal_steps = args.lr_anneal_steps  # 

    def run_loop(self):
        step = 0
        if self.args.mode == 'testing':
            avg_test_loss, avg_rmse, avg_ssim, avg_psnr, avg_mse, avg_nmse, rmse_key_result = self.valid(self.test_data, self.model, self.loss_fn, 0, self.folder_path, type='test', baseline=self.args.baseline, png_save=self.args.png_save)
            print(f'Test Loss:{avg_test_loss:>7f} MSE:{avg_mse:>5f} NMSE:{avg_nmse:>5f} RMSE:{avg_rmse:>5f} SSIM:{avg_ssim:>5f} PSNR:{avg_psnr:>5f} Stage:{self.args.stage} ')
            with open(self.folder_path + 'test.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, test_loss: {}, mse:{}, nmse:{}, rmse:{}, ssim:{}, psnr:{} \n'.format(self.args.stage, 0, avg_test_loss, avg_mse, avg_nmse, avg_rmse, avg_ssim, avg_psnr))
            print('Test End!')
            exit()

        print('------training start-----')
        for epoch in range(self.num_epochs):
            
            loss = self.train(self.data, self.model, self.loss_fn, self.opt, epoch, self.folder_path, baseline=self.args.baseline, args=self.args)
            lst_lr = self.anneal_lr(epoch, self.warmup_steps, self.args.lr_start, self.args.lr_end, self.args.lr_anneal_steps, self.opt, self.writer)

            
            # self.scheduler.step()
            # lst_lr = self.scheduler.get_last_lr()[0]
            
            print(f'Epoch [{epoch + 1}/{self.num_epochs}] Train Loss:{loss:>7f} LR:{lst_lr:.2e} Stage:{self.args.stage}')
            with open(self.folder_path + 'train.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, train_loss: {:>7f}, lr:{:.2e}\n'.format(self.args.stage, epoch+1, loss, lst_lr))
            # 添加梯度监控
            # total_grad_norm = 0.0
            # for p in self.model.parameters():
            #     if p.grad is not None:
            #         total_grad_norm += p.grad.norm().item()
            # print(f"Epoch {epoch} Gradient Norm: {total_grad_norm}")
            
            if epoch % self.log_interval == 0 and epoch > 0 or epoch == 10 or epoch == self.num_epochs-1:
                avg_eval_loss, avg_rmse, avg_ssim, avg_psnr, avg_mse, avg_nmse, rmse_key_result = self.valid(self.val_data, self.model, self.loss_fn, epoch, self.folder_path, type='valid', baseline=self.args.baseline, png_save=self.args.png_save)

                print(f'Epoch[{epoch + 1}/{self.num_epochs}] Evaluation Loss:{avg_eval_loss:>7f} MSE:{avg_mse:>5f} NMSE:{avg_nmse:>5f} RMSE:{avg_rmse:>5f} SSIM:{avg_ssim:>5f} PSNR:{avg_psnr:>5f} Stage:{self.args.stage} ')
                with open(self.folder_path + 'valid.txt', 'a') as f:
                    f.write('stage:{}, epoch:{}, valid_loss: {}, rmse:{}, ssim:{}, psnr:{} \n'.format(self.args.stage, epoch+1, avg_eval_loss, avg_rmse, avg_ssim, avg_psnr))

                best_result = self.best_model_save(epoch, avg_eval_loss, rmse_key_result, self.folder_path)
                
                if best_result == 'save':
                    avg_test_loss, avg_rmse, avg_ssim, avg_psnr, avg_mse, avg_nmse, rmse_key_result = self.valid(self.test_data, self.model, self.loss_fn, epoch, self.folder_path, type='test', baseline=self.args.baseline, png_save=self.args.png_save)
                    print(f' Epoch[{epoch + 1}/{self.num_epochs}] Test Loss:{avg_test_loss:>7f} MSE:{avg_mse:>5f} NMSE:{avg_nmse:>5f} RMSE:{avg_rmse:>5f} SSIM:{avg_ssim:>5f} PSNR:{avg_psnr:>5f} Stage:{self.args.stage} \n')
                    

                    with open(self.folder_path + 'test.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test_loss: {}, mse:{}, nmse:{}, rmse:{}, ssim:{}, psnr:{} \n'.format(self.args.stage, epoch+1, avg_test_loss, avg_mse, avg_nmse, avg_rmse, avg_ssim, avg_psnr))

    def train(self, dataloader, model, loss_fn, optimizer, epoch, folder_path, type='train', baseline=False, args=None):
        size = len(dataloader.dataset)
        error = 0.0
        total_samples = 0
        model.cuda()
        for batch_idx, (jpg_data, rss_gt, _,  height, frequency) in enumerate(dataloader):
            # jpg_data = jpg_data[:,:2,:,:]
            jpg_data = jpg_data.float().cuda()   # Move data to cuda and train with gpu
            rss_gt = rss_gt.float().cuda() 
            height = height.cuda()
            frequency = frequency.cuda()

            # 1） 前向传播+计算损失
            if baseline:
                rcv_rss = model(jpg_data)
            else:
                # rcv_rss = model(jpg_data, height, frequency, mode='backward')
                rcv_rss = model(jpg_data, height, frequency)

            loss = loss_fn(rcv_rss, rss_gt)  
            # 2) 梯度清零
            optimizer.zero_grad()  # 清空之前梯度
            # 3） 反向传播
            loss.backward()  # 反向传播算法计算梯度
            # 4） 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad)
            # 5） 参数更新
            optimizer.step()  # 更新参数
            # loss_change.append(loss.item())

            batch_size = jpg_data.size(0)
            error += loss.data.item() * batch_size
            total_samples += batch_size  
        #     # 每训练100次，输出一次当前信息
            if batch_idx % 400 == 0:
                loss, current = loss.item(), batch_idx * len(rss_gt)
                # print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # self.fig_show(epoch, jpg_data, rss_gt, rcv_rss, folder_path, current, type)                 

        # print('train size', size)
        # print('train total_samples', total_samples)
        avg_error = error/total_samples
        
        return avg_error
    
    def valid(self, dataloader, model, loss_fn, epoch, folder_path, type='valid', baseline=False, png_save=False):  # 验证
        rmse_key_result = {}
        test_loss = 0.0
        total_samples = 0
        ssim = 0.0
        rmse = 0.0
        psnr = 0.0
        mse = 0.0
        nmse = 0.0

        # ssim = []
        # rmse = []
        # psnr = []
        # mse = []
        # nmse = []
        model.eval()  # Turn the model into validation mode
        model.cuda()  # Place the model on cuda
        size = len(dataloader.dataset)
        print('size:', size)
        with torch.no_grad(): # Model parameters do not need to be updated during testing, so no_gard()
            for batch_idx, (jpg_data, rss_gt, rss_paths, height, frequency ) in enumerate(dataloader):
                # jpg_data = jpg_data[:,:2,:,:]
                jpg_data = jpg_data.float().cuda()   # Move data to cuda and train with gpu
                rss_gt = rss_gt.float().cuda() 
                height = height.cuda()
                frequency = frequency.cuda()
                if baseline:
                    rcv_rss = model(jpg_data)
                else:
                    # rcv_rss = model(jpg_data, height, frequency, mode='forward')
                    rcv_rss = model(jpg_data, height, frequency)

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                batch_size = jpg_data.size(0)
                batch_loss = loss_fn(rcv_rss, rss_gt).item()
                test_loss += batch_loss * batch_size
                total_samples += batch_size  
                if png_save == True:  #  如果是纯测试，生成图像保存到路径
                    save_rss_to_png(rss_paths, rcv_rss, prefix=folder_path)

                # 初始化 batch 变量
                rmse_batch = None
                ssim_batch = None
                psnr_batch = None
                mse_batch = None
                nmse_batch = None
                # rmse_batch, ssim_batch, psnr_batch, mse_batch, nmse_batch = evaluate_metrics(rss_gt.to(torch.float64), rcv_rss.to(torch.float64))   
                rmse_batch, ssim_batch, psnr_batch, mse_batch, nmse_batch = evaluate_metrics(rss_gt, rcv_rss)   

                # print('batch_loss:', batch_loss * batch_size)   
                # print('rmse_batch:', rmse_batch)   
         
                # 仅当批次指标非空时拼接
                if rmse_batch:  # 检查列表是否非空
                    rmse += rmse_batch
                if ssim_batch:
                    ssim += ssim_batch
                if psnr_batch:
                    psnr += psnr_batch
                if mse_batch:
                    mse += mse_batch
                if nmse_batch:
                    nmse += nmse_batch

                # if rmse_batch is not None:
                #     rmse.append(rmse_batch)

                # # 判断 ssim_batch 是否为 None，只有在 ssim_batch 有确定值时才追加
                # if ssim_batch is not None:
                #     ssim.append(ssim_batch)

                # # 判断 psnr_batch 是否为 None，只有在 psnr_batch 有确定值时才追加
                # if psnr_batch is not None:
                #     psnr.append(psnr_batch)

                # # 判断 mse_batch 是否为 None，只有在 mse_batch 有确定值时才追加
                # if mse_batch is not None:
                #     mse.append(mse_batch)

                # # 判断 nmse_batch 是否为 None，只有在 nmse_batch 有确定值时才追加
                # if nmse_batch is not None:
                #     nmse.append(nmse_batch)

                if batch_idx % 400 == 0:
                    loss, current = batch_loss, batch_idx * len(rss_gt)
                    # print(f"test loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    self.fig_show_save(epoch, jpg_data, rss_gt, rcv_rss, folder_path, current, type)                 
                   
        print('test total_samples', total_samples)
        print('test size', size)
        avg_test_loss = test_loss / total_samples
        avg_rmse = rmse / total_samples
        avg_ssim = ssim / total_samples
        avg_psnr = psnr / total_samples
        avg_mse = mse / total_samples
        avg_nmse = nmse / total_samples

        # avg_rmse = np.mean(rmse)
        # avg_ssim = np.mean(ssim)
        # avg_psnr = np.mean(psnr) 
        # avg_mse = np.mean(mse) 
        # avg_nmse = np.mean(nmse) 
        # print('Number of non-zero elements!', len(rmse))

        # print(f"Average Test Loss: {avg_test_loss}")
        # print(f"Average RMSE: {avg_rmse}")
        # print(f"Average SSIM: {avg_ssim}")
        # print(f"Average PSNR: {avg_psnr}")
        
        # torch.cuda.empty_cache()  # 清理 GPU 缓存
        return avg_test_loss, avg_rmse, avg_ssim, avg_psnr, avg_mse, avg_nmse, rmse_key_result
    

    def best_model_save(self, step, rmse, rmse_key_result, folder_path):
        if rmse < self.best_rmse:
            self.early_stop = 0
            torch.save(self.model.state_dict(), folder_path + 'model_best_stage_{}.pth'.format(self.args.stage))
            torch.save(self.model.state_dict(), folder_path + 'model_best.pth')
            self.best_rmse = rmse
            self.writer.add_scalar('Evaluation/RMSE_best', self.best_rmse, step)
            print('Epoch [{}/{}]  RMSE_best:{:>5f}'.format(step+1, self.num_epochs, self.best_rmse))

            # print(str(rmse_key_result)+'\n')

            with open(folder_path +'result.txt', 'w') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step+1, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            with open(folder_path + 'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step+1, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            return 'save'
        else:
            self.early_stop += 1
            print('stage:{}, epoch:{}, RMSE:{:>5f}, RMSE_best:{:>5f}, early_stop:{}\n'.format(self.args.stage, step+1, rmse, self.best_rmse, self.early_stop))
            with open(self.args.folder_path+'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, RMSE:{:>7f}, not optimized, early_stop:{}\n'.format(self.args.stage, step+1, rmse, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                with open(self.args.folder_path+'result.txt', 'a') as f:
                    f.write('Early stop!\n')
                with open(self.args.folder_path+'result_all.txt', 'a') as f:
                    f.write('Early stop!\n')
                    exit()
    
    def fig_show(self, epoch, jpg_data, rss_gt, rcv_rss, folder_path, current, type='valid'):  # 验证
        batchsize = min(jpg_data.shape[0], 10)  # 取当前批次大小和10中的最小值，最多画10张
        # 创建一个包含 (batchsize, 3) 个子图的图像
        fig, axes = plt.subplots(nrows=batchsize, ncols=5, figsize=(16, 4* batchsize))
        # 如果 batchsize=1，axes 是一个 1D 数组，需要调整为 2D 数组
        if batchsize == 1:
            axes = axes.reshape(1, -1)  # 将 axes 从 (3,) 调整为 (1, 3)
                # 遍历 batch 中的每个样本
        for i in range(batchsize):
            # 显示第一个通道 (地形数据)
            # axes[i, 0].imshow(jpg_data[i, 0].cpu(), cmap='terrain')  # 地形通道使用 terrain 色图
            # axes[i, 0].set_title('Terrain')
            # axes[i, 0].axis('off')

            # 显示第二个通道 (建筑物数据)
            axes[i, 1].imshow(jpg_data[i, 0].cpu(), cmap='gray')  # 建筑物数据可以使用灰度色图
            axes[i, 1].set_title('Buildings')
            axes[i, 1].axis('off')
            
            # 显示采样 (RSS 数据)
            axes[i, 2].imshow(jpg_data[i, 1].detach().cpu())  # RSS 数据使用热图色图
            axes[i, 2].set_title('sparse RSS')
            axes[i, 2].axis('off')

            # 显示groudtruth (RSS 数据)
            axes[i, 3].imshow(rss_gt[i, 0].detach().cpu())  # RSS 数据使用热图色图
            axes[i, 3].set_title('GroundTruth RSS')
            axes[i, 3].axis('off')

            # 显示恢复的 (RSS 数据)
            axes[i, 4].imshow(rcv_rss[i, 0].detach().cpu())  # RSS 数据使用热图色图
            axes[i, 4].set_title('recover RSS')
            axes[i, 4].axis('off')
        # 设置整个图的标题
        titleStr = 'epoch_' + str(epoch+1) +'_bat_'+str(current)
        plt.suptitle(titleStr)
        save_pth = folder_path + type +'/'
        os.makedirs(save_pth, exist_ok=True)
        save_name = 'epoch_'+str(epoch+1) + '_bat_'+str(current) + '.png'
        plt.savefig(save_pth + save_name, format='png')  # 保存图像
        plt.tight_layout()  # 自动调整子图间距
        plt.close()


    def fig_show_save(self, epoch, jpg_data, rss_gt, rcv_rss, folder_path, current, type='valid'):  # 验证
        batchsize = min(jpg_data.shape[0], 10)  # 取当前批次大小和10中的最小值，最多画10张

        plt.figure(figsize=(6, 6))
        img = plt.imshow(rcv_rss[0, 0].detach().cpu())  # 存储 imshow 返回的对象

        # 添加 colorbar
        cbar = plt.colorbar(img, fraction=0.046, pad=0.04)  # 调整 colorbar 大小和间距
        cbar.ax.tick_params(labelsize=14)  # 设置 colorbar 刻度字体大小

        save_pth = os.path.join(folder_path, type)  # 更安全的路径拼接方式
        os.makedirs(save_pth, exist_ok=True)
        save_name = f'epoch_{epoch+1}_recover_{current}.png'  # 使用 f-string 格式化
        plt.savefig(
            os.path.join(save_pth, save_name),
            format='png',
            # bbox_inches='tight',
            pad_inches=0,
            dpi=800,  # 高分辨率
            transparent=False  # 背景透明（如需透明可设为 True）
        )

        save_name = f'epoch_{epoch+1}_recover_{current}.pdf'  # 使用 f-string 格式化
        plt.savefig(
            os.path.join(save_pth, save_name),
            format='pdf',
            # bbox_inches='tight',
            pad_inches=0,
            dpi=800,  # 高分辨率
            transparent=False  # 背景透明（如需透明可设为 True）
        )
        plt.close()

        plt.figure(figsize=(6, 6))
        img = plt.imshow(rss_gt[0, 0].detach().cpu())  # 存储 imshow 返回的对象

        # 添加 colorbar
        cbar = plt.colorbar(img, fraction=0.046, pad=0.04)  # 调整 colorbar 大小和间距
        cbar.ax.tick_params(labelsize=14)  # 设置 colorbar 刻度字体大小

        save_pth = os.path.join(folder_path, type)  # 更安全的路径拼接方式
        os.makedirs(save_pth, exist_ok=True)
        save_name = f'epoch_{epoch+1}_groundtruth_{current}.png'  # 使用 f-string 格式化
        plt.savefig(
            os.path.join(save_pth, save_name),
            format='png',
            # bbox_inches='tight',
            pad_inches=0,
            dpi=800,  # 高分辨率
            transparent=False  # 背景透明（如需透明可设为 True）
        )

        save_name = f'epoch_{epoch+1}_groundtruth_{current}.pdf'  # 使用 f-string 格式化
        plt.savefig(
            os.path.join(save_pth, save_name),
            format='pdf',
            # bbox_inches='tight',
            pad_inches=0,
            dpi=800,  # 高分辨率
            transparent=False  # 背景透明（如需透明可设为 True）
        )
        plt.close()


    def anneal_lr(self, step, warmup_steps, lr, min_lr, lr_anneal_steps, opt, writer):
        '''
        step: 当前的训练步骤或迭代次数
        warmup_steps: 预热步数，用于定义学习率从初始值线性增加到目标学习率的步数。在这段时间内，学习率会逐渐增大，以帮助模型稳定。
        lr: 初始学习率，即训练开始时的学习率。这个值在训练开始时设置，并将在预热阶段和退火阶段中使用。
        min_lr: 最小学习率。在训练的后期阶段，学习率将降低到这个值，以防过拟合或在训练后期调整学习过程的敏感性。
        lr_anneal_steps: 学习率退火步骤数，用于定义在预热之后，学习率逐渐减小的阶段。这一阶段可以让模型在训练后期更细致地学习。
        opt: 优化器对象，通常是一个特定的优化算法（如 SGD、Adam 等）。优化器负责更新模型参数，并且通过访问其 param_groups 来更新学习率。
        writer: 用于记录训练过程的工具，通常是 TensorBoard 的记录对象，可以用来可视化训练指标，比如学习率。通过 writer.add_scalar 方法，可以记录学习率，以便在训练过程中进行监控和分析
        '''
        if step < warmup_steps:
            current_lr = lr * (step + 1) / warmup_steps
        elif step < lr_anneal_steps:
            current_lr = min_lr + (lr - min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi * (step - warmup_steps) / (lr_anneal_steps - warmup_steps)
                )
            )
        else:
            current_lr = min_lr

        for param_group in opt.param_groups:
            param_group["lr"] = current_lr
        
        writer.add_scalar('Training/LR', current_lr, step)
        return current_lr
    
    
    def hybrid_loss(self, pred, target):
        # pred = pred.squeeze(1)
        # target = target.squeeze(1)

        mse_loss = F.mse_loss(pred, target)
        # 计算水平和垂直梯度差异
        grad_x_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_x_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        grad_y_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        grad_y_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        grad_loss = F.l1_loss(grad_x_pred, grad_x_target) + F.l1_loss(grad_y_pred, grad_y_target)
        return mse_loss + 0.5 * grad_loss

# # 在训练循环中联合MSE和梯度损失
# total_loss = mse_loss(pred, target) + 0.5 * gradient_loss(pred, target)

# 新增梯度损失
class GradientLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01, data_range=1.0):
        """
        Args:
            alpha (float): 梯度损失的权重
            beta (float): SSIM损失的权重
            data_range (float): 图像动态范围（如归一化图像为1.0）
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.data_range = data_range
        
        # 定义Sobel梯度算子
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def forward(self, pred, target):
        # --- 梯度损失计算 ---
        # 将Sobel算子移动到与输入相同的设备
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)
        
        # 计算预测图像和真实图像的梯度
        grad_x_pred = F.conv2d(pred, self.sobel_x, padding=1)
        grad_y_pred = F.conv2d(pred, self.sobel_y, padding=1)
        grad_x_target = F.conv2d(target, self.sobel_x, padding=1)
        grad_y_target = F.conv2d(target, self.sobel_y, padding=1)

        mse_loss =  F.mse_loss(pred, target)
        
        # 计算梯度差异的L1损失
        grad_loss = ( F.l1_loss(grad_x_pred, grad_x_target) + F.l1_loss(grad_y_pred, grad_y_target) ) / 2
        
        # --- SSIM损失计算 ---


        pred1 = (pred - pred.min()) / (pred.max() - pred.min()).to(pred.device)
        target1 = (target - target.min()) / (target.max() - target.min()).to(target.device)
        ssim_values = ssim(pred1, target1, data_range=self.data_range)
        # 检查是否有NaN值，并进行修正
        epsilon = 1e-6
        if torch.isnan(ssim_values):
            ssim_values = torch.ones_like(ssim_values) * (1 - epsilon)
        ssim_loss = 1.0 - ssim_values
        # 总损失 = 梯度损失 + SSIM损失
        # total_loss = self.alpha * mse_loss + self.beta * grad_loss + self.gamma * ssim_loss
        
        # total_loss = self.alpha * mse_loss + self.beta * grad_loss
        
        total_loss = self.alpha * mse_loss + self.gamma * ssim_loss
        
        # total_loss = self.alpha * mse_loss
        
        return total_loss




if __name__ == '__main__':
    torch.manual_seed(42)
    
