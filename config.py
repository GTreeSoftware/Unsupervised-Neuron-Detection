from pprint import pprint
import torch
import numpy as np
import os
import time

class Config:
    ##############: for data prepare: no resize operation
    patch_size = [120,120,120]
    pixel_size=[0.2,0.2,1]

    ######################## for dataset
    dataset_prefix='Unsupervised_Train'

    # ##############: for training
    train_augument=True
    train_shuffile=True
    train_label_batch=3
    num_workers = 5

    use_cuda=True
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # user GPU or not

    model_choice='VoxResNet_3D'
    in_dim=1
    out_dim=2

    save_parameters_name = 'Un_Sup0'

    load_state=True

    # record the result
    log_path='result'
    # log_name=os.path.join(log_path,time.strftime('%m-%d-%H:%M:%S',time.localtime()))
    log_name = os.path.join(log_path, 'log_{}.txt'.format(save_parameters_name))

    optimizer='SGD'
    lr=0.01
    # lr = 0.001
    momentum=0.9
    weight_decay=0.0005

    scheduler='StepLR'
    step_size=4
    gamma=0.5

    # train parameters
    train_epoch=50
    train_plotfreq=2

    val_run=True
    val_plotfreq=2
    save_img=True

    ###################: for validatation
    val_batch=3
    test_batch=3

    ###################: for test
    image_size = [300, 300, 300]
    overlap = 10
    patch_valnum = np.ceil(image_size[0] / (patch_size[0] - overlap)) ** 3

    thred=32


    ###############: print the informations
    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}



###### build the instance
opt = Config()

