from utils.show_results import Read_Tiff
from Common import *
from GetModel import *
from utils.tools import *
from utils.meters import AverageMeter
import torch.nn as nn
import torch
import time
from utils.eval_metric import ConfusionMeter
import os
import numpy as np
from dataset.generatePatches import *
from torch.utils.data import DataLoader
from utils.show_results import Get_Compared_MIP,Get_Joint_MIP,ImageForVis
import skimage.io as skio
from glob import glob
from dataset.tools import WriteList2Txt,ReadTxt2List


def Normalization(img):
    img=np.array(img,np.float)
    return (img-np.min(img))/(np.max(img)-np.min(img))


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



def test_model(val_loader,model):
    model.eval()

    pred_Patches = []
    prob_patches =[]
    # image_Patches = []

    soft_max = nn.Softmax(dim=1)

    start_time=time.time()
    for batch_ids, (image_patch) in enumerate(val_loader):

        print(batch_ids)


        if opt.use_cuda:
            image_patch=image_patch.cuda()

            output=model(image_patch)

            with torch.no_grad():
                # just 0 and 1
                _,pred_patch=torch.max(output,dim=1)

                # for prob
                prob_patch = soft_max(output)
                prob_patch=prob_patch[:,1,...]

                del output

                pred_patch=pred_patch.cpu().numpy()
                prob_patch = prob_patch.cpu().numpy()

                # image_patch=image_patch.cpu().numpy()


                for id1 in range(pred_patch.shape[0]):
                    # 0 and 1
                    pred1=np.array(pred_patch[id1,:,:,:], dtype=np.float32)
                    pred_Patches.append( pred1 )

                    # prob
                    prob1 = np.array(prob_patch[id1, :, :, :], dtype=np.float32)
                    prob_patches.append(prob1)


                    # image1=np.array(image_patch[image_num,0,:,:,:], dtype=np.float32)
                    # image_Patches.append(image1)

    # save the images into npy type
    run_time=time.time()-start_time
    print(run_time)

    return pred_Patches,prob_patches







##################################################################
####### This is used for Osten Dataset  ##########################
##################################################################



if __name__=="__main__":

    file_path='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

    file_names = glob(os.path.join(file_path, '*.tif'))

    # begin to predict
    recon_path='XXXXXXXXXXXXXXXXXXXXXXXx'
    recon_path_pred=recon_path+'/pred'
    make_dirs(recon_path_pred)

    # begin to test the big dataset
    model = GetModel(opt)

    parameters_name = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'


    model_CKPT = torch.load(parameters_name)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')


    for image_num in range(len(file_names)):
        # file_name1=file_names[image_num]
        file_name1 = os.path.join(file_path,file_names[image_num])

        save_num=file_name1.split('/')[-1].split('.')[0]


        # read the image and process
        start1=time.time()
        image=Read_Tiff(file_name1)
        print('reading time:{}'.format(time.time()-start1))

        # begin to generate patches
        patch_size=np.array([120,120,120])
        overlap=np.array([10,10,10])
        patch_indices=compute_patch_indices(image.shape, patch_size, overlap, start=0)


        Tdataset = GenerateDataset_ForNew(image, image.shape, patch_size, overlap)
        val_loader = DataLoader(Tdataset, batch_size=4, num_workers=5, shuffle=False)

        pred_Patches,prob_patches=test_model(val_loader, model)

        # begin to combine the images into one
        start=time.time()
        patchindices = compute_patch_indices(image.shape, patch_size, overlap,start=0)

        # # 0 and 1 recon
        pred_recon = reconstruct_from_patches(pred_Patches, patchindices, image.shape)
        pred_recon=pred_recon.astype(np.uint8)

        # prob recon (need)
        prob_recon = reconstruct_from_patches(prob_patches, patchindices, image.shape)

        run_time=time.time()-start
        print(run_time)

        tifffile.imsave(os.path.join(recon_path_pred, save_num + '_pred0.tif'), np.uint8(pred_recon * 255))
        # tifffile.imsave(os.path.join(recon_path_pred, save_num + '_prob0.tif'), np.uint8(prob_recon * 255))



        print('ok')

































