"""This is used for 3D show of the segmentation result"""

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from utils.tools import make_dirs
import skimage.io as skio
from utils.afterprocessing import denoise_by_connection

# def readTiffImage(image_name):
#     with tifffile.TiffFile(image_name) as tif:
#         image = tif.asarray()
#
#     return image


# read the mip images of the file
def Read_Tiff(tiff_name):
    img=tifffile.imread(tiff_name)
    return img

# get the mip images of the image
def Get_MIP_Image(img,direction):
    im1=np.max(img,axis=direction)
    return im1

def Get_Joint_MIP(img):
    if np.max(img)<=1:
        img=np.uint8(img*255)

    im0=np.max(img,0)
    im1 = np.max(img, 1)
    im2 = np.max(img, 2)
    imh = Hor_Cat(im0, im1, im2)
    return imh


def Get_MIP(img,axis):
    if np.max(img)<=1:
        img=np.uint8(img*255)

    im0=np.max(img,axis)
    return im0

def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()

def Hor_Cat(im1,im2,im3):
    out=im1
    out=np.hstack([out,im2,im3])
    return out

def Ver_Cat(im1,im2,im3,im4):
    out=im1
    out=np.vstack([out,im2,im3,im4])
    return out


def Cal_Seg_Diff(lb1,sg1):
    """calculate the difference of the Diff between label and seg"""
    df1=lb1.astype(np.int16)-sg1.astype(np.int16)

    # eliminate small noise caused by different edge effect
    df2=df1.copy()
    df2[df2<0]=1
    df2=denoise_by_connection(df2,16)

    df3=df1.copy()
    df3[df3>0]=1
    df3=denoise_by_connection(df3,16)

    df_out=np.array(np.zeros(df1.shape),np.uint8)
    df_out[df2>0]=100
    df_out[df3>0]=240

    return df_out

# calculate the compared mip images of all
def Get_Compared_MIP(img, label, pred):
    Diff1=Cal_Seg_Diff(label,pred)

    dmip=Get_Joint_MIP(img)
    lmip=Get_Joint_MIP(label)
    pmip=Get_Joint_MIP(pred)
    dmip1=Get_Joint_MIP(Diff1)

    out=dmip
    out=np.vstack([out,lmip,pmip,dmip1])
    return out


def ImageForVis(image_recon):
    """Just for Visualization"""
    image_recon = np.array(image_recon, np.int32)
    image_recon[image_recon > 350] = 350
    image_recon[image_recon < 100] = 100
    image_recon = (image_recon - np.min(image_recon)) / (np.max(image_recon) - np.min(image_recon))
    image_recon=np.uint8(image_recon * 255)
    return image_recon






if __name__=="__main__":

    # save the results of the images
    root_path='/home/hp/Neuro_Separate/Neuron_3D3/result/'
    save_path=root_path+'Recon_MIP'
    make_dirs(save_path)

    img_path=root_path+'3D_recon'

    total_num=90

    for bids in range(total_num):
        img_name='{}data.tif'.format(bids+1)
        label_name='{}label.tif'.format(bids+1)
        seg_name='{}pred.tif'.format(bids+1)

        img=Read_Tiff(os.path.join(img_path,img_name))
        label=Read_Tiff(os.path.join(img_path,label_name))
        seg=Read_Tiff(os.path.join(img_path,seg_name))

        # save the mip result of 2d
        mip_image = Get_Compared_MIP(img, label, seg)
        mip_name = os.path.join(save_path, '{}_mip.png'.format(bids+1))
        skio.imsave(mip_name, mip_image)


    # plt.figure()
    # plt.imshow(out)
    # plt.show()
    print('OK')
