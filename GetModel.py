from model.vnet import VNet as VNet_3D
import torch.nn as nn
import torch
from model.VoxResNet import VoxResNet_3D


model_choice={
    'VNet_3D':VNet_3D,
    'VoxResNet_3D':VoxResNet_3D,

}


# decide to use which Model: currently UNet
def GetModel(opt):

    if opt.model_choice == 'VoxResNet_3D':
        # act_fn=nn.LeakyReLU(0.01)
        model=model_choice[opt.model_choice](opt.in_dim,opt.out_dim)

    # decide whether to use cuda or not
    if opt.use_cuda:
        model=nn.DataParallel(model).cuda()

    return model


def GetSemiModel(opt,para_name,ema=False):
    """1. get the model"""
    model=GetModel(opt)

    """2. load the parameters"""
    # model_CKPT = torch.load(para_name)
    # model.load_state_dict(model_CKPT['state_dict'])

    model.load_state_dict(torch.load(para_name))

    """3. decide ema model or original model"""
    if ema:
        for param in model.parameters():
            param.detach_()

    return model







# if __name__=="__main__":
#     model=GetModel(opt)
#     print(model)

