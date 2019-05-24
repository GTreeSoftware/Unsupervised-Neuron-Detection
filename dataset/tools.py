import numpy as np
import scipy.io as io
import random
from config import opt
from utils.tools import *
import sys
from glob import glob
import tifffile

# read the image and label list of corresponding images
def GetImageLabelList_Path(data_dir_root):
    image_path=os.path.join(data_dir_root,'image')
    label_path=os.path.join(data_dir_root,'label')

    # image and label had corresponding name
    name_list = os.listdir(image_path)

    # random.shuffle(name_list)

    image_list=[os.path.join(image_path,name_num) for name_num in name_list]    # image and label had corresponding name
    label_list=[os.path.join(label_path,name_num) for name_num in name_list]

    return image_list,label_list



def GetImageLabelList_PathNum(data_dir_root,num1):
    image_path=os.path.join(data_dir_root,'image')
    label_path=os.path.join(data_dir_root,'label{}'.format(num1))

    # image and label had corresponding name
    name_list = os.listdir(image_path)

    # random.shuffle(name_list)

    image_list=[os.path.join(image_path,name_num) for name_num in name_list]    # image and label had corresponding name
    label_list=[os.path.join(label_path,name_num) for name_num in name_list]

    return image_list,label_list


def Read_ImageLabelList_PatchesInd(prefix,phase):
    # read the image label list and patches index
    module_path = os.path.dirname(__file__)
    data_path = module_path + '/' + prefix + '/'

    image_list_name = data_path + phase + '_image_list.txt'
    label_list_name = data_path + phase + '_label_list.txt'

    image_list = ReadTxt2List(image_list_name)
    label_list = ReadTxt2List(label_list_name)

    # get the patch index of the dataset
    index_list_name = data_path + phase + '_random_patch_ind.npy'
    index_list = np.load(index_list_name)

    return image_list,label_list,index_list



def GetList(prefix,phase1='train'):
    # generate the list name of the file
    # data_path = './dataset/'
    data_path=prefix+'/'

    image_list_name = data_path + phase1 + '_image_list.txt'
    label_list_name = data_path + phase1 + '_label_list.txt'

    image_list = ReadTxt2List(image_list_name)
    label_list = ReadTxt2List(label_list_name)

    return image_list,label_list



def RandomSelectPatches(patch_size,patch_num,prefix,threshold1=1200,phase='train'):
    # calculate how many patches to generate
    image_list, label_list = GetList(prefix,phase)

    new_patch_index=[]

    # decide to generate new patch
    count_num=0

    while count_num<patch_num:
        # random get image index
        inum = random.randint(0,len(image_list)-1)

        # calculate whether fit the choice
        label1 = tifffile.imread(label_list[inum])
        label_patch1,position1=RandomPatches(label1,patch_size)
        label_patch_indnum = LabelIndexNum(label_patch1)

        # fit a predefined rule
        threshold1=np.array(threshold1)

        # contain only one
        if threshold1.size==1:
            if label_patch_indnum>=threshold1[0]:
                new_position=np.hstack((position1,inum))

                new_patch_index.append(new_position)

                count_num+=1

                print('-----{} generated_num:{}'.format(phase,count_num))

                # save the index list
                np.save(prefix+'/{}_random_patch_ind.npy'.format(phase), new_patch_index)


        # contain 2 constrains:
        if threshold1.size==2:
            if (label_patch_indnum>=threshold1[0]) and (label_patch_indnum<=threshold1[1]):
                new_position=np.hstack((position1,inum))

                new_patch_index.append(new_position)

                count_num+=1

                print('-----{} generated_num:{}'.format(phase,count_num))

                # save the index list
                np.save(prefix+'/{}_random_patch_ind.npy'.format(phase), new_patch_index)


def Unite_2_List_PatchesInd(prefix0,phase0,prefix1,phase1,prefix2,phase2):
    image_list0, label_list0, index_list0=Read_ImageLabelList_PatchesInd(prefix0,phase0)
    image_list1, label_list1, index_list1 = Read_ImageLabelList_PatchesInd(prefix1, phase1)

    ######################## getting the new index and label list for train and val #####
    old_image_num=len(image_list0)

    new_image_list=image_list0+image_list1
    new_label_list=label_list0+label_list1


    # get the new index
    index_list1[:,3]+=old_image_num
    index_list2=np.vstack([index_list0,index_list1])

    ###### resave all these images for new ################################################
    new_ilist_name=prefix2+'/'+phase2+'_image_list.txt'
    WriteList2Txt(new_ilist_name,new_image_list)

    new_llist_name=prefix2+'/'+phase2+'_label_list.txt'
    WriteList2Txt(new_llist_name,new_label_list)

    new_patch_index_name=prefix2+'/'+phase2+'_random_patch_ind.npy'
    np.save(new_patch_index_name,index_list2)











def split_list(input_list, split=0.8, rseed=100, shuffle_list=True):
    random.seed(rseed)

    if shuffle_list:
        random.shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing




# save the filelist into txt
def WriteList2Txt(name1,ipTable,mode='w'):
    with open(name1,mode=mode) as fileObject:
        for ip in ipTable:
            fileObject.write(ip)
            fileObject.write('\n')


# Read the filelist to list
def ReadTxt2List(name1,mode='r'):
    result=[]
    with open(name1,mode=mode) as f:
        data = f.readlines()   #read all the trsing into data
        for line in data:
            word = line.strip()  # list
            result.append(word)
    return result


# get the random patches
def RandomPatches(image,patch_size):
    w,h,z = image.shape
    pw,ph,pz=patch_size

    # calculate the random patches index
    nw,nh,nz=w-pw,h-ph,z-pz
    iw=random.randint(0,nw-1)
    ih=random.randint(0,nh-1)
    iz=random.randint(0,nz-1)

    # at: different from matlab
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches,[iw,ih,iz]


def RandomJointPatches(image,label,patch_size):
    w,h,z = image.shape
    pw,ph,pz=patch_size

    # calculate the random patches index
    nw,nh,nz=w-pw,h-ph,z-pz
    iw=random.randint(0,nw-1)
    ih=random.randint(0,nh-1)
    iz=random.randint(0,nz-1)

    # at: different from matlab
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]
    label_patches=label[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches,label_patches


def GetPatches(image,patch_size,position):
    pw,ph,pz=patch_size
    iw,ih,iz=position
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches

# calculate the useful mask numbers
def LabelIndexNum(label_image):
    image1=label_image>0
    image1=image1.flatten()
    sum_index=np.sum(image1)

    return sum_index

def DecideRange(isize,psize,osize):
    i1=np.array(isize,np.int)
    p1=np.array(psize,np.int)
    o1=np.array(osize,np.int)

    index1=np.asarray(np.mgrid[0:i1:(p1-o1)])
    len1=len(index1)
    index1[len1-1]=i1-1-p1

    return index1



# calculate the ordered patch inds
def GetOrderedPatchInds(image_size,patch_size,overlap):
    indx=DecideRange(image_size[0],patch_size[0],overlap[0])
    indy=DecideRange(image_size[1],patch_size[1],overlap[1])
    indz = DecideRange(image_size[2], patch_size[2], overlap[2])

    Xout = [(xx, yy, zz) for xx in indx for yy in indy for zz in indz]

    return Xout


def Normalization(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))
