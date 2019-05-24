"""This is used to generate the train and val, test datasets for the images"""

from dataset.tools import *
from utils.tools import make_dirs
import shutil
import tifffile

def GenerateDataListUsePathForTrain(Image_Path,save_prex):
    """gennerate the namelist for train,val and test"""
    """image_list, label_list and range, prefix, phase """
    path_len=len(Image_Path)

    new_image_list=[]
    new_label_list=[]

    for i in range(path_len):
        tmp_path=Image_Path[i]
        tmp_image_list,tmp_label_list=GetImageLabelList_Path(tmp_path)

        new_image_list.extend(tmp_image_list)
        new_label_list.extend(tmp_label_list)

    # save the list name
    ilist_name=save_prex+'_image_list.txt'
    WriteList2Txt(ilist_name,new_image_list)

    llist_name=save_prex+'_label_list.txt'
    WriteList2Txt(llist_name,new_label_list)



def GetList(prefix,phase1='train'):
    # generate the list name of the file
    # data_path = './dataset/'
    data_path=prefix+'/'

    image_list_name = data_path + phase1 + '_image_list.txt'
    label_list_name = data_path + phase1 + '_label_list.txt'

    image_list = ReadTxt2List(image_list_name)
    label_list = ReadTxt2List(label_list_name)

    return image_list,label_list


def RandomSelectPatches(patch_size,patch_num,prefix,threshold1=900,phase='train'):
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
        if label_patch_indnum>=threshold1:
            new_position=np.hstack((position1,inum))

            new_patch_index.append(new_position)

            count_num+=1

            print('-----{} generated_num:{}'.format(phase,count_num))

            # save the index list
            np.save(prefix+'/{}_random_patch_ind.npy'.format(phase), new_patch_index)



if __name__=="__main__":
    ####### this is used to copy the sparse dataset to defined area ###
    save_dir = 'Unsupervised_Train'
    make_dirs(save_dir)

    # here to put your own dataset
    first_path_train='/media/hp/work/Unspervised_VRN/First_Selected/train'
    first_path_val = '/media/hp/work/Unspervised_VRN/First_Selected/val'


    Train_Path=[first_path_train]
    Val_Path=[first_path_val]

    GenerateDataListUsePathForTrain(Train_Path, save_dir+'/train')
    GenerateDataListUsePathForTrain(Val_Path, save_dir + '/val')


    # generate the patches according to the ranked orders
    patch_size = opt.patch_size

    # 35 and 15 are train and val patches respectively
    RandomSelectPatches(patch_size, 35*30, save_dir, threshold1=0.0005*120*120*120, phase='train')
    RandomSelectPatches(patch_size, 15*30, save_dir, threshold1=0.0005*120*120*120, phase='val')

    print('ok')





