import torch
import torch.optim
import os
import argparse
import time
import dataloader
from loss import Charbonnier_Loss, SSIM_Loss, torchPSNR

import numpy as np
from Net_RSM_CAT import Net

def train(config):
    i = 0 #now epoch
    enhan_net = Net().cuda()

    train_dataset = dataloader.dehazing_loader(config.enhan_images_path,config.ori_images_path)
    val_dataset = dataloader.dehazing_loader(config.enhan_images_path,config.ori_images_path, mode="val")
    # 训练集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    # 验证集
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)

    criterion_char = Charbonnier_Loss()
    criterion_ssim = SSIM_Loss()

    # Adam优化器能够自适应地调整学习率
    optimizer = torch.optim.Adam(enhan_net.parameters(), lr=config.lr)
    enhan_net.train()

    # 记录最佳
    best_psnr = 0
    best_epoch = 0

    for epoch in range(i,config.num_epochs):
        # 每个epoch完成后测试，放入samples
        _train_loss = []
        _val_psnr = []
        # optimizer.zero_grad()
        print("*" * 80 + "第%i轮" % epoch + "*" * 80)

        for iteration, (img_clean, img_ori) in enumerate(train_loader):

            img_clean = img_clean.cuda()
            img_ori = img_ori.cuda()

            try:
                enhanced_image = enhan_net(img_ori)
                char_loss = criterion_char(img_clean, enhanced_image)
                ssim_loss = criterion_ssim(img_clean, enhanced_image)
                ssim_loss = 1 - ssim_loss  # -SSIM损失
                sum_loss = char_loss + 0.5*ssim_loss
                # print("char_loss:" + str(char_loss))
                # print("ssim_loss:" + str(0.5*ssim_loss))
                # print("all_loss:" + str(char_loss + 0.5*ssim_loss))
                _train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm(enhan_net.parameters(), config.grad_clip_norm)
                optimizer.step()

                print("Loss at iteration", iteration + 1, ":", sum_loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e
        with open(config.checkpoint_folder + "loss.log", "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % ((epoch + 1), np.mean(_train_loss)) + "\n"
            f.write(s)

        #Validation Stage
        with torch.no_grad():
            for iteration, (img_clean, img_ori) in enumerate(val_loader):
                img_clean = img_clean.cuda()
                img_ori = img_ori.cuda()

                enhanced_image = enhan_net(img_ori)

                _psnr = torchPSNR(img_clean, enhanced_image)  # 计算ssim值
                _val_psnr.append(_psnr.item())

        _val_psnr = np.mean(np.array(_val_psnr))

        if _val_psnr > best_psnr:
            best_psnr = _val_psnr
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': enhan_net.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(config.checkpoint_folder, "model_best.pth"))

        print(
            "[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, _val_psnr, best_epoch, best_psnr))

        with open(config.checkpoint_folder + "val_PSNR.log", "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, _val_psnr, best_epoch, best_psnr) + "\n")

        torch.save({'epoch': epoch,
                    'state_dict': enhan_net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(config.checkpoint_folder, "model_latest.pth"))

if __name__ == "__main__":
    """
    	输入为原始水下图像和增强后的水下图像，都转为灰度图进入网络
    	:param enhan_images_path:增强后的水下图像
    	:param orig_images_path:原始水下图像
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--enhan_images_path', type=str, default="./dataset/train/target_E/")
    parser.add_argument('--ori_images_path', type=str, default="./dataset/train/input_E/")

    parser.add_argument('--resume', type=str, default='results')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_folder', type=str, default="trained_model/Net_RSM_CAT_E/")
    parser.add_argument('--cudaid', type=str, default="5",help="choose cuda device id 0-7).")

    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cudaid

    if not os.path.exists(config.checkpoint_folder):
        os.mkdir(config.checkpoint_folder)


    torch.cuda.empty_cache()
    s = time.time()
    train(config)
    e = time.time()
    print(str(e-s))
