"""This fucntion is used to get the train and val dataset
    imageformat:mat file, read the list that saved these patches
    1.getting the image and label at the same time
    2.image: limited range, then to [0,1]
    3.augument at the same time
    """

import numpy as np
import os
from config import opt
from dataset.tools import *
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader
from utils.show_results import *
from utils.tools import make_dirs
from dataset.joint_transform import *
import tifffile

"""This is with sparse, middle and dense datasets"""

## generate random patches from the
class GetSemiDataset():
    def __init__(self,prefix,phase='label_train',is_label=True,augument=None):
        # generate the list name of the file
        module_path = os.path.dirname(__file__)
        data_path=module_path+'/'+prefix+'/'

        image_list_name=data_path+phase+'_image_list.txt'
        label_list_name=data_path+phase+'_label_list.txt'

        self.image_list=ReadTxt2List(image_list_name)
        self.label_list=ReadTxt2List(label_list_name)
        self.is_label=is_label

        # get the patch index of the dataset
        index_list_name=data_path+phase+'_random_patch_ind.npy'
        self.index_list=np.load(index_list_name)

        self.augument=augument

        self.mean1=np.array([150],dtype=np.float32)
        self.std1=np.array([350],dtype=np.float32)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, ind):
        # load the image and labels
        image_num = self.index_list[ind, 3]

        image=tifffile.imread(self.image_list[image_num])
        label=tifffile.imread(self.label_list[image_num])

        # get the patches of the image and label
        patch_position=self.index_list[ind,:3]
        image_patch = GetPatches(image,opt.patch_size,patch_position)
        label_patch = GetPatches(label,opt.patch_size,patch_position)

        # get the numpy data of the images: whether the type is ok??
        # image_patch_ori = np.array(image_patch, dtype=np.int32)

        image_patch=np.array(image_patch,dtype=np.float32)
        label_patch=np.array(label_patch,dtype=np.int32)

        # # limit the range of the image_patch
        # image_patch=np.clip(image_patch,opt.limited_range[0],opt.limited_range[1])

        # decide whether to augument or not
        # if self.augument:
        #     if self.is_label:
        #         image_patch,label_patch=self.augument(image_patch,label_patch)
        #     else:
        #         image_patch=self.augument(image_patch)

        ## normalizae
        image_patch=(image_patch-self.mean1)/self.std1

        if np.max(label_patch)>1:
            label_patch=label_patch/np.max(label_patch)

        if self.augument:
            image_patch,label_patch=self.augument(image_patch,label_patch)

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)

        image_patch=torch.from_numpy(image_patch).float()
        label_patch=torch.from_numpy(label_patch).long()

        # consider what to return with or without label
        return image_patch,label_patch




if __name__=="__main__":
    # generate the mips of the images
    test_dataset = GetSemiDataset(prefix='sparse20',phase='label_train',is_label=True,augument=None)
    test_loader=DataLoader(test_dataset)

    save_path='./train_img'
    make_dirs(save_path)


    for ids,(img,label) in enumerate(test_loader):
        img= img.numpy()
        label=label.numpy()
        img=np.squeeze(img)
        label=np.squeeze(label)

        img=Normalization(img)*255
        label=label*255
        img=img.astype(np.uint8)
        label=label.astype(np.uint8)

        im0 = Get_MIP_Image(img, 0)
        im1 = Get_MIP_Image(img, 1)
        im2 = Get_MIP_Image(img, 2)

        lb0 = Get_MIP_Image(label, 0)
        lb1 = Get_MIP_Image(label, 1)
        lb2 = Get_MIP_Image(label, 2)


        imh = Hor_Cat(im0, im1, im2)
        lbh = Hor_Cat(lb0, lb1, lb2)

        out = np.vstack([imh,lbh])

        save_name = os.path.join(save_path, 'mip_B{}_I0.png'.format(ids))
        skio.imsave(save_name, out)

    print('train_img_finished')




