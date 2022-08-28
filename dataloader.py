import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import random

def populate_train_list(enhan_images_path, ori_images_path):
    image_list_haze_index = os.listdir(ori_images_path)  # 文件名
    image_list_haze_index= random.sample(image_list_haze_index, 790)
    all_length = len(image_list_haze_index)

    image_dataset = []
    for i in image_list_haze_index:  # 添加路径，并组合为元组
        image_dataset.append((enhan_images_path + i, ori_images_path + i))

    train_list = image_dataset[:int(all_length*0.9)]
    val_list = image_dataset[int(all_length*0.9):]

    return train_list, val_list

class dehazing_loader(data.Dataset):

    def __init__(self, enhan_images_path, ori_images_path, mode='train'):

        self.train_list, self.val_list = populate_train_list(enhan_images_path, ori_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_clean_path, data_ori_path = self.data_list[index]

        data_clean = Image.open(data_clean_path)
        data_ori = Image.open(data_ori_path)

        # data_clean = data_clean.resize((512, 512), Image.CUBIC)
        # data_ori = data_ori.resize((512, 512), Image.BICUBIC)

        data_clean = (np.asarray(data_clean) / 255.0)
        data_ori = (np.asarray(data_ori) / 255.0)

        data_clean = torch.from_numpy(data_clean).float()
        data_ori = torch.from_numpy(data_ori).float()

        return data_clean.permute(2, 0, 1), data_ori.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


class test_loader(data.Dataset):
    def __init__(self, inp_dir, img_options):
        super(test_loader, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename