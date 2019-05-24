import sys
sys.path.append('./result')
import numpy as np
import os
from dataset.tools import ReadMatFile,GetPatches,GetImageLabelList_Path,ReadTxt2List
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader
import tifffile
from skimage import morphology
from utils.afterprocessing import *



def compute_patch_indices(image_shape, patch_size, overlap, start=None):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    if start is None:
        n_patches = np.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow / 2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    return get_set_of_patch_indices(start, stop, step)


def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                      start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)


def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0] + patch_shape[0], patch_index[1]:patch_index[1] + patch_shape[1],
           patch_index[2]:patch_index[2] + patch_shape[2]]


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index



def reconstruct_from_patches(patches, patch_indices, data_shape):
    """
    Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
    patches are averaged.
    :param patches: List of numpy array patches.
    :param patch_indices: List of indices that corresponds to the list of patches.
    :param data_shape: Shape of the array from which the patches were extracted.
    :param default_value: The default value of the resulting data. if the patch coverage is complete, this value will
    be overwritten.
    :return: numpy array containing the data reconstructed by the patches.
    """

    data = np.zeros(data_shape)

    image_shape = data_shape[-3:]
    # count = np.zeros(data_shape)

    count = np.zeros(data_shape, dtype=np.int)


    for patch, index in zip(patches, patch_indices):
        image_patch_shape = patch.shape[-3:]

        # consider the situation that patch index small than 0 or larger than image shape
        if np.any(index < 0):
            fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
            patch = patch[..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
            index[index < 0] = 0


        if np.any((index + image_patch_shape) >= image_shape):
            fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
                                                        * ((index + image_patch_shape) - image_shape)), dtype=np.int)
            patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]

        x0, y0, z0 = index[0], index[1], index[2]
        x1, y1, z1 = index[0] + patch.shape[-3], index[1] + patch.shape[-2], index[2] + patch.shape[-1]

        # calculate the new area
        data[x0:x1, y0:y1, z0:z1] += patch
        count[x0:x1, y0:y1, z0:z1] += 1

    # get the average of the data
    data = data / count

    return data


def denoise_by_connection(image, min_size):
    image = image > 0.8
    image = image.astype(np.bool)
    image_new = morphology.remove_small_objects(image, min_size, connectivity=2)
    image_new = image_new.astype(np.int)
    image_new = image_new > 0.8
    image_new = image_new.astype(np.int)
    return image_new


def joinPatches(name, savepath, index, Patches):
    patchindices = compute_patch_indices(image_shape=opt.image_size, patch_size=opt.patch_size, overlap=opt.overlap,
                                         start=None)
    image_recon = reconstruct_from_patches(Patches, patchindices, opt.image_size)

    if str(name) == 'pred':
        image_recon = denoise_by_connection( image_recon, min_size=opt.thred )
        tifffile.imsave( os.path.join(savepath, str(index) + name + 'denoise.tif'), np.uint8(image_recon*255) )

    # tifffile.imsave(os.path.join(savepath, str(index) + name + '.tif'), np.uint8(image_recon * 255))

    return image_recon


def SaveImageLabel(name, savepath, index, img,phase='image'):
    """save the corresponding image and label"""
    image_name = os.path.join(savepath, str(index) + name + '.tif')
    if phase=='image':
        tifffile.imsave(image_name, np.int32(img))
    else:
        tifffile.imsave(image_name,np.uint8(img * 255))


def ImageForVis(image_recon):
    """Just for Visualization"""
    image_recon = np.array(image_recon, np.int32)
    image_recon[image_recon > 350] = 350
    image_recon[image_recon < 100] = 100
    image_recon = (image_recon - np.min(image_recon)) / (np.max(image_recon) - np.min(image_recon))
    image_recon=np.uint8(image_recon * 255)
    return image_recon



def GetImageLabelList(prefix, phase='label_train'):
    # obtain the original image path
    data_path='./dataset/'
    # data_path = ''
    module_path = os.path.dirname(__file__)
    data_path = module_path + '/' + prefix + '/'

    image_list_name = data_path + phase + '_image_list.txt'
    label_list_name = data_path + phase + '_label_list.txt'

    image_list=ReadTxt2List(image_list_name)
    label_list=ReadTxt2List(label_list_name)

    return image_list, label_list



def GetImageLabeList(image_list, label_list,image_num=0):
    # load the image and labels
    image_name = image_list[image_num]
    image=tifffile.imread(image_name)

    label_name=label_list[image_num]
    label=tifffile.imread(label_name)

    image = np.array(image,dtype=np.int32)
    label = np.array(label, dtype=np.int32)

    return image, label











def GetImageLabel(prefix, phase='label_train',image_num=0):
    # obtain the original image path
    data_path='./dataset/'
    # data_path = ''
    module_path = os.path.dirname(__file__)
    data_path = module_path + '/' + prefix + '/'

    image_list_name = data_path + phase + '_image_list.txt'
    label_list_name = data_path + phase + '_label_list.txt'

    image_list=ReadTxt2List(image_list_name)
    label_list=ReadTxt2List(label_list_name)

    # load the image and labels
    image_name = image_list[image_num]
    image=ReadMatFile(image_name,'Image')

    label_name=label_list[image_num]
    label=ReadMatFile(label_name,'Label')

    image = np.array(image,dtype=np.int32)
    label = np.array(label, dtype=np.int32)

    return image, label



class GenerateDataset_ForNew():
    def __init__(self,image,image_shape, patch_size, overlap):
        self.n_patches = np.ceil(image_shape / (patch_size - overlap))
        self.patches_num = int(self.n_patches[0]*self.n_patches[1]*self.n_patches[2])
        self.patch_size=patch_size

        self.image=image.astype(np.float32)

        ## normalizae
        self.mean1 = np.array([150],dtype=np.float32)
        self.std1=np.array([350],dtype=np.float32)
        self.image=(self.image-self.mean1)/self.std1

        # calculate patch index
        self.patchindices = compute_patch_indices(image_shape, patch_size, overlap,start=0)

    def __len__(self):
        return self.patches_num


    def __getitem__(self, ind ):
        # get the patches of the image and label
        image_patch = get_patch_from_3d_data(self.image, patch_shape=self.patch_size, patch_index=self.patchindices[ind, :])

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)

        image_patch=torch.from_numpy(image_patch).float()

        return image_patch




class GenerateDataset_ForNew2():
    def __init__(self,image,label,image_shape, patch_size, overlap):
        self.n_patches = np.ceil(image_shape / (patch_size - overlap))
        self.patches_num = int(self.n_patches[0]*self.n_patches[1]*self.n_patches[2])
        self.patch_size=patch_size

        self.image=image.astype(np.float32)
        self.label=label.astype(np.int32)

        if np.max(self.label)>1:
            self.label=self.label>0

        self.label = self.label.astype(np.int32)

        ## normalizae
        self.mean1 = np.array([150],dtype=np.float32)
        self.std1=np.array([350],dtype=np.float32)

        self.image=(self.image-self.mean1)/self.std1

        # calculate patch index
        self.patchindices = compute_patch_indices(image_shape, patch_size, overlap,start=0)


    def __len__(self):
        return self.patches_num


    def __getitem__(self, ind ):
        # get the patches of the image and label
        image_patch = get_patch_from_3d_data(self.image, patch_shape=self.patch_size, patch_index=self.patchindices[ind, :])

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)

        image_patch=torch.from_numpy(image_patch).float()


        label_patch=get_patch_from_3d_data(self.label, patch_shape=self.patch_size, patch_index=self.patchindices[ind, :])
        label_patch = torch.from_numpy(label_patch).long()


        return image_patch,label_patch


class GenerateDataset_OnlyImage():
    def __init__(self,image,image_shape, patch_size, overlap):
        self.n_patches = np.ceil(image_shape / (patch_size - overlap))
        self.patches_num = int(self.n_patches[0]*self.n_patches[1]*self.n_patches[2])
        self.patch_size=patch_size

        self.image=image.astype(np.float32)

        ## normalizae
        self.mean1 = np.array([150],dtype=np.float32)
        self.std1=np.array([350],dtype=np.float32)

        # constrain the range
        # self.image = np.clip(self.image, opt.limited_range[0], opt.limited_range[1])

        self.image=(self.image-self.mean1)/self.std1

        # calculate patch index
        self.patchindices = compute_patch_indices(image_shape, patch_size, overlap,start=None)

    def __len__(self):
        return self.patches_num


    def __getitem__(self, ind ):
        # get the patches of the image and label
        image_patch = get_patch_from_3d_data(self.image, patch_shape=self.patch_size, patch_index=self.patchindices[ind, :])

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)

        image_patch=torch.from_numpy(image_patch).float()

        return image_patch



class GenerateDataset_ForNewImage():
    def __init__(self,image,image_shape, patch_size, overlap):
        self.n_patches = np.ceil(image_shape / (patch_size - overlap))
        self.patches_num = int(self.n_patches[0]*self.n_patches[1]*self.n_patches[2])
        self.patch_size=patch_size

        self.image=image.astype(np.float32)

        ## normalizae
        self.mean1 = np.array([150],dtype=np.float32)
        self.std1=np.array([350],dtype=np.float32)
        self.image=(self.image-self.mean1)/self.std1

        # calculate patch index
        self.patchindices = compute_patch_indices(image_shape, patch_size, overlap,start=None)

    def __len__(self):
        return self.patches_num


    def __getitem__(self, ind ):
        # get the patches of the image and label
        image_patch = get_patch_from_3d_data(self.image, patch_shape=self.patch_size, patch_index=self.patchindices[ind, :])

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)

        image_patch=torch.from_numpy(image_patch).float()


        return image_patch









class GenerateTestDataset():
    def __init__(self, prefix, phase='label_train', is_label=True, augument=None):
        # obtain the original image path
        # generate the list name of the file
        module_path = os.path.dirname(__file__)
        data_path=module_path+'/'+prefix+'/'

        image_list_name=data_path+phase+'_image_list.txt'
        label_list_name=data_path+phase+'_label_list.txt'

        self.image_list=ReadTxt2List(image_list_name)
        self.label_list=ReadTxt2List(label_list_name)
        self.is_label=is_label

        # get the patch index of the dataset
        self.augument=augument

        self.mean1=np.array([150],dtype=np.float32)
        self.std1=np.array([350],dtype=np.float32)

        self.patches_num = int (opt.patch_valnum * len(self.image_list))


    def __len__(self):
        return self.patches_num


    def __getitem__(self, ind ):
        # load the image and labels
        image_num = int( np.ceil( (ind+1)/opt.patch_valnum ) )

        image_name = self.image_list[image_num-1]
        image=ReadMatFile(image_name,'Image')

        label_name=self.label_list[image_num-1]
        label=ReadMatFile(label_name,'Label')

        image = np.array(image,dtype=np.float32)
        label = np.array(label, dtype=np.int32)

        ## normalizae
        image = (image - self.mean1) / self.std1

        if np.max(label)>1:
            label = label/np.max(label)

        if self.augument:
            image, label = self.augument(image, label)

        patchindices = compute_patch_indices(image_shape=opt.image_size, patch_size=opt.patch_size, overlap=opt.overlap,
                                             start=None)

        patchindex = int(ind - (image_num - 1) * opt.patch_valnum)
        # get the patches of the image and label
        image_patch = get_patch_from_3d_data(image, patch_shape=opt.patch_size, patch_index=patchindices[patchindex, :])

        label_patch = get_patch_from_3d_data(label, patch_shape=opt.patch_size, patch_index=patchindices[patchindex, :])

        # get the numpy data of the images: whether the type is ok??
        image_patch = np.array(image_patch, dtype=np.float32)
        label_patch = np.array(label_patch, dtype=np.int32)

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)

        image_patch=torch.from_numpy(image_patch).float()
        label_patch=torch.from_numpy(label_patch).long()

        # print(type(image_patch))

        return image_patch, label_patch





