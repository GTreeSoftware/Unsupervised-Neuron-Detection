"""This is used to:
   1.generate the dataloader for train or validation
   2.get the optimizer
   """

from dataset.Get_Semi_Dataset import GetSemiDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.tools import make_dirs
import torch
import sys
import tifffile
import os
import numpy as np
import shutil
import random


def GetSemiDatasetLoader(opt,prefix='UnSupervised',phase= 'Train',phase_1=None,augument=None):
    # decide whether to augument or not
    if phase == 'Train':
        # dataset with label
        dataset1 = GetSemiDataset(prefix, phase=phase_1, is_label=True, augument=augument)
        dataloader1 = DataLoader(dataset1, batch_size=opt.train_label_batch, num_workers=opt.num_workers,
                                 shuffle=opt.train_shuffile)

    if phase == 'Val':
        dataset1=GetSemiDataset(prefix,phase=phase_1,augument=None)
        dataloader1 = DataLoader(dataset1, batch_size=opt.val_batch, num_workers=opt.num_workers, shuffle=False)

    if phase == 'Test':
        dataset1 = GetSemiDataset(prefix,phase=phase_1,augument=None)
        dataloader1 = DataLoader(dataset1, batch_size=opt.test_batch, num_workers=opt.num_workers, shuffle=False)

    return dataloader1





def GetOptimizer(opt,model):
    if opt.optimizer=='SGD':
        optimizer1=optim.SGD(
            model.parameters(),lr=opt.lr,momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )

    if opt.optimizer=='RMSp':
        optimizer1=optim.RMSprop(
            model.parameters(),lr=opt.lr,alpha=opt.alpha,
            weight_decay=opt.weight_decay,
        )

    return optimizer1


def GetScheduler(opt,optimizer1):
    scheduler1=optim.lr_scheduler.StepLR(
        optimizer1,step_size=opt.step_size,gamma=opt.gamma
    )
    return scheduler1


def adjust_lr(opt,optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# save the result
def save_parameters(state,best_value='VoxResNet'):
    save_path = 'checkpoints'
    make_dirs(save_path)

    # save_name = save_path + '/model_parameters_value{:.3f}.pth'.format(best_value)
    save_name = save_path + '/{}.ckpt'.format(best_value)
    torch.save(state, save_name)


def load_checkpoint(model, checkpoint_PATH, optimizer):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    start_epoch = model_CKPT['epoch'] + 1

    return model, optimizer,start_epoch



def save_img2tiff(img,file_name,save_path='result_img'):
    # make_dirs(save_path)
    # save_name=os.path.join(save_path,file_name)
    save_name=file_name
    tifffile.imsave(save_name,img,dtype=np.uint8)
    print('saved:',save_name)



def GetWeight(opt,target,slr=0,is_t=1):
    if target.device.type=='cuda':
        beta = target.sum().cpu().numpy().astype(np.float32) / (target.numel() + 1e-5)
    else:
        beta = target.sum().numpy().astype(np.float32) / (target.numel() + 1e-5)

    beta = beta + slr
    weight = np.array([beta, 1 - beta])

    if is_t:
        weight = torch.tensor(weight)
        if opt.use_cuda:
            weight = weight.float().cuda()

    return weight


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



#### write the log files
def WriteLog(opt1):
    log = Logger()
    make_dirs(opt1.log_path)
    log.open(opt1.log_name, mode='a')
    log.write('** experiment settings **\n')
    log.write('\toptimizer:               {}  \n'.format(opt1.optimizer))
    log.write('\tlearning rate:         {:.3f}\n'.format(opt1.lr))
    log.write('\tweight_decay:         {:.4f}\n'.format(opt1.weight_decay))
    log.write('\tepoches:               {:.3f}\n'.format(opt1.train_epoch))
    log.write('\tpatch_size:              {}  \n'.format(opt1.patch_size))
    log.write('\tmodel:                   {}  \n'.format(opt1.model_choice))
    log.write('\tsave_parameter:                   {}  \n'.format(opt1.save_parameters_name))

    return log


